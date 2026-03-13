/// Training data generation via self-play
///
/// Each thread plays games independently, recording positions with
/// search evaluations. Output: binary file for NNUE training.
///
/// Binary format per position (26 bytes):
///   pits[0][0..9]:  9 bytes (u8)
///   pits[1][0..9]:  9 bytes (u8)
///   kazan[0]:       1 byte  (u8)
///   kazan[1]:       1 byte  (u8)
///   tuzdyk[0]:      1 byte  (i8)
///   tuzdyk[1]:      1 byte  (i8)
///   side_to_move:   1 byte  (u8, 0=white 1=black)
///   eval:           2 bytes (i16, little-endian, from side-to-move perspective)
///   result:         1 byte  (0=white_loss, 1=draw, 2=white_win)

use std::fs::File;
use std::io::{BufWriter, Write, Read};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::board::{Board, Side, GameResult, NUM_PITS, WIN_THRESHOLD};
use crate::eval::evaluate;
use crate::nnue::NnueNetwork;
use crate::search::Searcher;

const RECORD_SIZE: usize = 26;
const TOTAL_STONES: u8 = 162;

/// Pack a position into 26 bytes
fn pack_position(board: &Board, eval: i16, result: u8) -> [u8; RECORD_SIZE] {
    let mut buf = [0u8; RECORD_SIZE];

    // Pits
    for i in 0..9 {
        buf[i] = board.pits[0][i];
        buf[9 + i] = board.pits[1][i];
    }

    // Kazan
    buf[18] = board.kazan[0];
    buf[19] = board.kazan[1];

    // Tuzdyk
    buf[20] = board.tuzdyk[0] as u8;
    buf[21] = board.tuzdyk[1] as u8;

    // Side to move
    buf[22] = board.side_to_move.index() as u8;

    // Eval (i16 little-endian)
    let eval_bytes = eval.to_le_bytes();
    buf[23] = eval_bytes[0];
    buf[24] = eval_bytes[1];

    // Result
    buf[25] = result;

    buf
}

/// Simple xorshift64 PRNG (no external dependency needed)
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed) }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Generate a random valid endgame position with a target number of board stones.
/// Returns None if the generated position is terminal or has no legal moves.
fn random_endgame_position(rng: &mut Rng, max_board_stones: u8) -> Option<Board> {
    // Pick random total stones on board: 3..=max_board_stones
    let n = 3 + rng.range((max_board_stones - 2) as usize) as u8;

    // Distribute remaining stones to kazans
    // Total in kazans = 162 - n. Split between two players, both must be < 82
    let kazan_total = TOTAL_STONES - n;
    // kw can range from max(0, kazan_total - 81) to min(81, kazan_total)
    let kw_min = if kazan_total > 81 { kazan_total - 81 } else { 0 };
    let kw_max = std::cmp::min(81, kazan_total);
    if kw_min > kw_max {
        return None;
    }
    let kw = kw_min + rng.range((kw_max - kw_min + 1) as usize) as u8;
    let kb = kazan_total - kw;

    // Distribute n stones randomly across 18 pits
    let mut pits = [[0u8; NUM_PITS]; 2];
    for _ in 0..n {
        let pit = rng.range(18);
        let side = pit / 9;
        let idx = pit % 9;
        pits[side][idx] += 1;
    }

    // Random tuzdyk (most positions don't have one)
    let mut tuzdyk = [-1i8; 2];
    let tuz_chance = rng.range(4); // ~25% chance of having tuzdyks
    if tuz_chance == 0 {
        // White's tuzdyk (on black's side): pits 1-7 (indices 1-7, not pit 8)
        let idx = 1 + rng.range(7);
        tuzdyk[0] = idx as i8;
    }
    let tuz_chance2 = rng.range(4);
    if tuz_chance2 == 0 {
        let idx = 1 + rng.range(7);
        // Can't be same pit as opponent's tuzdyk
        if tuzdyk[0] != idx as i8 {
            tuzdyk[1] = idx as i8;
        }
    }

    // Random side to move
    let side_to_move = if rng.range(2) == 0 { Side::White } else { Side::Black };

    let board = Board::from_parts(pits, [kw, kb], tuzdyk, side_to_move);

    // Validate: not terminal, has legal moves
    if board.is_terminal() {
        return None;
    }
    let mut moves = [0usize; NUM_PITS];
    let num_moves = board.valid_moves_array(&mut moves);
    if num_moves == 0 {
        return None;
    }

    Some(board)
}

/// Load starting positions from binary file (23 bytes each).
/// Format: u32 count + count * 23 bytes (pits0[9] + pits1[9] + kazan[2] + tuzdyk[2] + side[1])
pub fn load_starting_positions(path: &str) -> Vec<Board> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to load starting positions from {}: {}", path, e);
            return Vec::new();
        }
    };

    let mut count_buf = [0u8; 4];
    if file.read_exact(&mut count_buf).is_err() {
        return Vec::new();
    }
    let count = u32::from_le_bytes(count_buf) as usize;

    let mut positions = Vec::with_capacity(count);
    let mut buf = [0u8; 23];

    for _ in 0..count {
        if file.read_exact(&mut buf).is_err() {
            break;
        }

        let mut pits = [[0u8; NUM_PITS]; 2];
        for i in 0..9 {
            pits[0][i] = buf[i];
            pits[1][i] = buf[9 + i];
        }
        let kazan = [buf[18], buf[19]];
        let tuzdyk = [buf[20] as i8, buf[21] as i8];
        let side = if buf[22] == 0 { Side::White } else { Side::Black };

        let board = Board::from_parts(pits, kazan, tuzdyk, side);
        if !board.is_terminal() {
            positions.push(board);
        }
    }

    eprintln!("Loaded {} starting positions from {}", positions.len(), path);
    positions
}

/// Play one self-play game, return positions
/// When use_hce_labels=true, positions are labeled with HCE static eval
/// instead of search score (for hybrid NNUE-gameplay + HCE-label datagen)
fn play_game(
    searcher: &mut Searcher,
    search_depth: i32,
    search_time_ms: u64,
    rng: &mut Rng,
    adjudicate: bool,
    eval_scale: f64,
    use_hce_labels: bool,
    endgame_start: bool,
    expert_starts: &[Board],
) -> Vec<[u8; RECORD_SIZE]> {
    let mut board = if !expert_starts.is_empty() {
        // Pick a random expert midgame position
        let idx = rng.range(expert_starts.len());
        expert_starts[idx]
    } else if endgame_start {
        // Try up to 20 times to get a valid endgame position
        let mut pos = None;
        for _ in 0..20 {
            if let Some(b) = random_endgame_position(rng, 50) {
                pos = Some(b);
                break;
            }
        }
        match pos {
            Some(b) => b,
            None => Board::new(), // fallback
        }
    } else {
        Board::new()
    };
    let mut positions: Vec<(Board, i16)> = Vec::with_capacity(120);
    let mut ply = 0;

    // Random opening: first N plies are random moves (fewer for endgame starts)
    let random_plies: usize = if endgame_start { 2 } else { 8 };
    const MAX_PLIES: usize = 300; // prevent infinite games
    const ADJUDICATE_THRESHOLD: i32 = 200; // ~winning advantage in NNUE/64 scale
    const ADJUDICATE_COUNT: usize = 4; // consecutive plies

    let mut adjudicate_counter = 0usize;
    let mut adjudicate_side: Option<Side> = None;

    // Track game history for repetition detection
    searcher.game_history.clear();
    searcher.push_game_position(searcher.compute_hash(&board));

    // Play the game
    loop {
        if board.is_terminal() || ply >= MAX_PLIES {
            break;
        }

        let mut moves = [0usize; 9];
        let num_moves = board.valid_moves_array(&mut moves);
        if num_moves == 0 { break; }

        if ply < random_plies {
            // Random move for opening diversity
            let idx = rng.range(num_moves);
            board.make_move(moves[idx]);
            searcher.push_game_position(searcher.compute_hash(&board));
            ply += 1;
            continue;
        }

        // Search from current position
        let result = searcher.search(&board, search_depth, search_time_ms);

        // Label: HCE static eval (for hybrid mode) or search score
        let raw_score = if use_hce_labels {
            evaluate(&board)
        } else {
            result.score
        };
        let eval = (raw_score as f64 * eval_scale).clamp(-3000.0, 3000.0) as i16;
        if result.score.abs() < 50000 {
            positions.push((board, eval));
        }

        // Adjudication: stop if one side is clearly winning for several plies
        if adjudicate && (eval as i32).abs() > ADJUDICATE_THRESHOLD {
            let winning = if eval > 0 { board.side_to_move } else { board.side_to_move.opposite() };
            if adjudicate_side == Some(winning) {
                adjudicate_counter += 1;
            } else {
                adjudicate_side = Some(winning);
                adjudicate_counter = 1;
            }
            if adjudicate_counter >= ADJUDICATE_COUNT {
                break;
            }
        } else {
            adjudicate_counter = 0;
            adjudicate_side = None;
        }

        // Play the best move
        board.make_move(result.best_move);
        searcher.push_game_position(searcher.compute_hash(&board));
        ply += 1;
    }

    // Determine game result (from board or adjudication)
    let result_byte = if let Some(game_result) = board.game_result() {
        match game_result {
            GameResult::Win(Side::White) => 2u8,
            GameResult::Win(Side::Black) => 0u8,
            GameResult::Draw => 1u8,
        }
    } else if let Some(winner) = adjudicate_side {
        // Adjudicated result
        if winner == Side::White { 2u8 } else { 0u8 }
    } else {
        1u8 // draw by max plies
    };

    // Pack all positions with the game result
    positions
        .iter()
        .map(|(b, eval)| pack_position(b, *eval, result_byte))
        .collect()
}

/// Run data generation
pub fn run_datagen(num_games: u32, depth: i32, time_ms: u64, num_threads: u32, nnue: Option<NnueNetwork>, output_prefix: &str, use_hce_labels: bool, endgame: bool, starts_file: Option<&str>) {
    let has_nnue = nnue.is_some();

    // With NNUE/64 normalization, search scores are already in centipawn-like scale.
    // No scaling needed — store raw search scores as training evals.
    let eval_scale = 1.0;

    println!("NNUE Training Data Generation");
    println!("=============================");
    println!("Games: {}", num_games);
    println!("Search: depth {} / {}ms", depth, time_ms);
    println!("Threads: {}", num_threads);
    println!("Eval: {}", if has_nnue { "NNUE" } else { "Handcrafted" });
    if use_hce_labels {
        println!("Labels: HCE static eval (hybrid mode)");
    }
    if endgame {
        println!("Mode: ENDGAME (random endgame starting positions, 3-50 stones)");
    }

    // Load expert starting positions if provided
    let expert_starts: Arc<Vec<Board>> = Arc::new(match starts_file {
        Some(path) => {
            let starts = load_starting_positions(path);
            println!("Mode: EXPERT STARTS ({} positions from {})", starts.len(), path);
            starts
        }
        None => Vec::new(),
    });

    println!("Eval scale: {:.4}", eval_scale);
    println!("Output prefix: {}", output_prefix);
    println!();

    let games_done = Arc::new(AtomicU64::new(0));
    let positions_total = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let start_time = Instant::now();
    let prefix = output_prefix.to_string();

    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let games_done = games_done.clone();
        let positions_total = positions_total.clone();
        let stop = stop.clone();
        let nnue_clone = nnue.clone();
        let prefix = prefix.clone();
        let expert_starts = expert_starts.clone();
        let games_per_thread = num_games / num_threads
            + if thread_id < num_games % num_threads { 1 } else { 0 };

        handles.push(std::thread::spawn(move || {
            let filename = format!("{}_thread_{}.bin", prefix, thread_id);
            let file = File::create(&filename).expect("Failed to create output file");
            let mut writer = BufWriter::with_capacity(1024 * 1024, file);
            let mut searcher = Searcher::new(4); // 4MB TT per thread
            searcher.silent = true;
            if let Some(net) = nnue_clone {
                searcher.set_nnue(net);
            }
            let mut rng = Rng::new(thread_id as u64 * 6364136223846793005 + 1442695040888963407);
            let mut local_positions = 0u64;

            for _ in 0..games_per_thread {
                if stop.load(Ordering::Relaxed) {
                    break;
                }

                let records = play_game(&mut searcher, depth, time_ms, &mut rng, false, eval_scale, use_hce_labels, endgame, &expert_starts);
                local_positions += records.len() as u64;

                for record in &records {
                    writer.write_all(record).expect("Write failed");
                }

                let done = games_done.fetch_add(1, Ordering::Relaxed) + 1;
                positions_total.store(
                    positions_total.load(Ordering::Relaxed) + records.len() as u64,
                    Ordering::Relaxed,
                );

                if done % 50 == 0 {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let pct = done as f64 / num_games as f64;
                    let games_per_min = done as f64 / elapsed * 60.0;
                    let eta_secs = if pct > 0.0 { elapsed / pct * (1.0 - pct) } else { 0.0 };
                    let eta_h = eta_secs as u64 / 3600;
                    let eta_m = (eta_secs as u64 % 3600) / 60;
                    let pos = positions_total.load(Ordering::Relaxed);

                    // Progress bar: 30 chars wide
                    let filled = (pct * 30.0) as usize;
                    let bar: String = (0..30).map(|i| if i < filled { '█' } else { '░' }).collect();

                    eprint!(
                        "\r[{}] {:>3.0}% | {}/{} | {:.1}M pos | {:.0} g/min | ETA: {}h {:02}m  ",
                        bar, pct * 100.0, done, num_games,
                        pos as f64 / 1e6, games_per_min, eta_h, eta_m,
                    );
                }
            }

            writer.flush().expect("Flush failed");
            (filename, local_positions)
        }));
    }

    // Wait for all threads
    let mut all_files = Vec::new();
    let mut total_pos = 0u64;
    for handle in handles {
        let (filename, count) = handle.join().expect("Thread panicked");
        all_files.push(filename);
        total_pos += count;
    }

    eprintln!(); // newline after progress bar
    let elapsed = start_time.elapsed();

    // Merge all thread files into one
    let output_file = format!("{}_training_data.bin", prefix);
    {
        let out = File::create(&output_file).expect("Failed to create merged file");
        let mut writer = BufWriter::new(out);

        for filename in &all_files {
            let data = std::fs::read(filename).expect("Failed to read thread file");
            writer.write_all(&data).expect("Write failed");
            std::fs::remove_file(filename).ok(); // cleanup thread files
        }
        writer.flush().expect("Flush failed");
    }

    println!("\n=============================");
    println!("Data generation complete!");
    println!("Games played: {}", games_done.load(Ordering::Relaxed));
    println!("Positions: {}", total_pos);
    println!("Time: {:.1}s", elapsed.as_secs_f64());
    println!(
        "Speed: {:.1} games/min",
        games_done.load(Ordering::Relaxed) as f64 / elapsed.as_secs_f64() * 60.0
    );
    println!("Output: {} ({:.1} MB)", output_file, total_pos as f64 * RECORD_SIZE as f64 / 1e6);
}

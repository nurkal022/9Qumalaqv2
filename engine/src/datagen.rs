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
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::board::{Board, Side, GameResult};
use crate::nnue::NnueNetwork;
use crate::search::Searcher;

const RECORD_SIZE: usize = 26;

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

/// Play one self-play game, return positions
fn play_game(
    searcher: &mut Searcher,
    search_depth: i32,
    search_time_ms: u64,
    rng: &mut Rng,
    adjudicate: bool,
    eval_scale: f64,
) -> Vec<[u8; RECORD_SIZE]> {
    let mut board = Board::new();
    let mut positions: Vec<(Board, i16)> = Vec::with_capacity(120);
    let mut ply = 0;

    // Random opening: first 8 plies are random moves
    const RANDOM_PLIES: usize = 8;
    const MAX_PLIES: usize = 300; // prevent infinite games
    const ADJUDICATE_THRESHOLD: i32 = 8500; // near-terminal
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

        if ply < RANDOM_PLIES {
            // Random move for opening diversity
            let idx = rng.range(num_moves);
            board.make_move(moves[idx]);
            searcher.push_game_position(searcher.compute_hash(&board));
            ply += 1;
            continue;
        }

        // Search from current position
        let result = searcher.search(&board, search_depth, search_time_ms);

        // Scale and clamp eval to trainable range
        let raw_score = result.score;
        let eval = (raw_score as f64 * eval_scale).clamp(-3000.0, 3000.0) as i16;
        if raw_score.abs() < 50000 {
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
pub fn run_datagen(num_games: u32, depth: i32, time_ms: u64, num_threads: u32, nnue: Option<NnueNetwork>, output_prefix: &str) {
    let has_nnue = nnue.is_some();

    // Auto-detect eval scale: search starting position, target mean_abs ~500
    let eval_scale = if has_nnue {
        let mut calibration_searcher = Searcher::new(4);
        calibration_searcher.silent = true;
        if let Some(ref net) = nnue {
            calibration_searcher.set_nnue(net.clone());
        }
        let board = Board::new();
        let result = calibration_searcher.search(&board, depth, time_ms);
        let abs_score = result.score.unsigned_abs().max(1) as f64;
        let target_abs = 500.0;
        let scale = (target_abs / abs_score).min(1.0); // never amplify, only shrink
        eprintln!("Eval calibration: start_pos score={}, scale={:.4}", result.score, scale);
        scale
    } else {
        1.0
    };

    println!("NNUE Training Data Generation");
    println!("=============================");
    println!("Games: {}", num_games);
    println!("Search: depth {} / {}ms", depth, time_ms);
    println!("Threads: {}", num_threads);
    println!("Eval: {}", if has_nnue { "NNUE" } else { "Handcrafted" });
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

                let records = play_game(&mut searcher, depth, time_ms, &mut rng, false, eval_scale);
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

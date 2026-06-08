pub(crate) use togyzkumalaq_core as board;
mod book;
mod datagen;
mod egtb;
mod eval;
mod nnue;
mod search;
mod texel;
mod tt;
mod zobrist;

use std::io::{self, BufRead, Write};
use std::sync::Arc;
use board::{Board, Side, GameResult, NUM_PITS};
use book::OpeningBook;
use nnue::NnueNetwork;
use search::Searcher;

const NNUE_PATH: &str = "nnue_weights.bin";
const BOOK_PATH: &str = "opening_book.txt";
const EGTB_PATH: &str = "egtb.bin";

/// Resolve an asset file: prefer next to the executable (so a deployed binary finds
/// its NNUE/EGTB/book regardless of CWD), else fall back to the CWD-relative name
/// (preserves the engine/-dir and ab_match workflows).
fn resolve_asset(name: &str) -> String {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join(name);
            if p.exists() {
                return p.to_string_lossy().into_owned();
            }
        }
    }
    name.to_string()
}

fn load_book() -> Option<OpeningBook> {
    OpeningBook::load(&resolve_asset(BOOK_PATH))
}

/// Simple xor-shift PRNG seeded from system time. We don't need crypto-grade
/// randomness — just enough variation between game positions.
fn rand_u64() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static SEED: AtomicU64 = AtomicU64::new(0);
    let mut s = SEED.load(Ordering::Relaxed);
    if s == 0 {
        s = SystemTime::now().duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64).unwrap_or(0xDEADBEEF) | 1;
    }
    // xorshift64
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    SEED.store(s, Ordering::Relaxed);
    s
}

/// Pick a move from the scored candidates with creativity-controlled randomness.
///   level 1: light  — uniform sample among moves within 30 cp of best
///   level 2: medium — softmax sample among moves within 60 cp, temperature 30
///   level 3: high   — softmax sample among moves within 100 cp, temperature 60
/// Mate-zone scores (|score| > EVAL_MATE - 200) are NEVER randomized — we play strict
/// to convert wins and not blunder mate.
fn pick_creative(scored: &[(usize, i32)], level: u32, best_score: i32) -> usize {
    if scored.is_empty() {
        return 0;
    }
    // Mate-zone: don't gamble with wins/losses
    if best_score.abs() > 90_000 - 200 {
        return scored[0].0;
    }
    let (window_cp, temp_cp): (i32, f64) = match level {
        1 => (30, 0.0),     // uniform within window
        2 => (60, 30.0),    // softmax, modest temperature
        3 => (100, 60.0),   // softmax, hotter
        _ => return scored[0].0,
    };
    let top_score = scored[0].1;
    // Candidates within window of best
    let candidates: Vec<&(usize, i32)> = scored.iter()
        .filter(|(_, s)| top_score - s <= window_cp)
        .collect();
    if candidates.len() == 1 {
        return candidates[0].0;
    }

    let r = (rand_u64() as f64) / (u64::MAX as f64);  // [0, 1)

    if temp_cp <= 0.0 {
        // Uniform among candidates
        let idx = (r * candidates.len() as f64) as usize;
        return candidates[idx.min(candidates.len() - 1)].0;
    }

    // Softmax weights based on score gap
    let mut weights: Vec<f64> = candidates.iter()
        .map(|(_, s)| ((-(top_score - *s) as f64) / temp_cp).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    for w in weights.iter_mut() { *w /= total; }

    let mut acc = 0.0;
    for (i, w) in weights.iter().enumerate() {
        acc += w;
        if r <= acc {
            return candidates[i].0;
        }
    }
    candidates[candidates.len() - 1].0
}

fn load_egtb() -> Option<Arc<egtb::EndgameTablebase>> {
    let egtb_path = resolve_asset(EGTB_PATH);
    if std::path::Path::new(&egtb_path).exists() {
        match egtb::EndgameTablebase::load(&egtb_path) {
            Ok(tb) => {
                eprintln!("EGTB loaded: {} entries, max {} stones", tb.len(), tb.max_stones);
                Some(Arc::new(tb))
            }
            Err(e) => {
                eprintln!("Warning: failed to load EGTB: {}", e);
                None
            }
        }
    } else {
        None
    }
}

fn load_nnue() -> Option<NnueNetwork> {
    let nnue_path = resolve_asset(NNUE_PATH);
    if std::path::Path::new(&nnue_path).exists() {
        match NnueNetwork::load(&nnue_path) {
            Ok(net) => {
                eprintln!("NNUE loaded from {}", nnue_path);
                Some(net)
            }
            Err(e) => {
                eprintln!("Warning: failed to load NNUE: {}", e);
                None
            }
        }
    } else {
        eprintln!("No NNUE weights found, using handcrafted eval");
        None
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "bench" => run_bench(),
            "play" => play_interactive(),
            "perft" => run_perft(),
            "selfplay" => run_selfplay(),
            "texel" => texel::run_texel_tuning(),
            "match" => {
                let num_games: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
                let time_ms: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(500);
                run_match(num_games, time_ms);
            }
            "match-nnue" => {
                // match-nnue <weights_a> <weights_b> [games] [time_ms]
                let weights_a = args.get(2).map(|s| s.as_str()).unwrap_or("nnue_weights.bin");
                let weights_b = args.get(3).map(|s| s.as_str()).unwrap_or("nnue_weights_champion_backup.bin");
                let num_games: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
                let time_ms: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(500);
                run_match_nnue(weights_a, weights_b, num_games, time_ms);
            }
            "datagen" => {
                let num_games: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10000);
                let depth: i32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
                let threads: u32 = args.get(4).and_then(|s| s.parse().ok())
                    .unwrap_or(std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(4));
                let prefix = args.get(5).map(|s| s.as_str()).unwrap_or("local");
                // Use --hce flag or "hce" prefix to force HCE eval (needed for K=1050 training)
                let use_hce = args.iter().any(|a| a == "--hce") || prefix.contains("hce");
                // --hybrid: NNUE plays games, HCE static eval provides labels
                let use_hybrid = args.iter().any(|a| a == "--hybrid");
                let nnue = if use_hce && !use_hybrid {
                    eprintln!("Datagen: using HCE eval (for K=1050 training compatibility)");
                    None
                } else {
                    load_nnue()
                };
                let use_hce_labels = use_hce || use_hybrid;
                let endgame = args.iter().any(|a| a == "--endgame");
                let starts_file = args.iter().position(|a| a == "--starts")
                    .and_then(|i| args.get(i + 1).map(|s| s.as_str()));
                datagen::run_datagen(num_games, depth, 500, threads, nnue, prefix, use_hce_labels, endgame, starts_file);
            }
            "egtb-gen" => {
                let max_stones: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
                let output = args.get(3).map(|s| s.as_str()).unwrap_or(EGTB_PATH);
                egtb::generate_egtb(max_stones, output);
            }
            "egtb-verify" => {
                let num_tests: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10000);
                if let Some(tb) = load_egtb() {
                    egtb::verify_egtb(&tb, num_tests);
                } else {
                    eprintln!("No EGTB file found at {}", EGTB_PATH);
                }
            }
            "analyze" => {
                // Format: analyze "w0,w1,...,w8/b0,...,b8/kw,kb/tw,tb/side" [time_ms]
                let pos = args.get(2).map(|s| s.as_str()).unwrap_or("");
                let time_ms: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3000);
                run_analyze(pos, time_ms);
            }
            "serve" => run_serve(),
            _ => print_usage(),
        }
    } else {
        print_usage();
    }
}

fn print_usage() {
    println!("Togyzkumalaq Championship Engine v1.0");
    println!();
    println!("Usage:");
    println!("  togyzkumalaq-engine play      - Play against the engine");
    println!("  togyzkumalaq-engine bench      - Run benchmark");
    println!("  togyzkumalaq-engine perft      - Count nodes at each depth");
    println!("  togyzkumalaq-engine selfplay   - Engine plays against itself");
    println!("  togyzkumalaq-engine texel      - Tune eval weights (Texel method)");
    println!("  togyzkumalaq-engine match [games] [time_ms]");
    println!("                                 - NNUE vs Handcrafted eval match");
    println!("  togyzkumalaq-engine datagen [games] [depth] [threads] [prefix]");
    println!("                                 - Generate NNUE training data");
    println!("  togyzkumalaq-engine analyze <position> [time_ms]");
    println!("                                 - Analyze position (JSON output)");
    println!("  togyzkumalaq-engine serve      - Persistent stdin/stdout protocol");
    println!("    Position format: w0,w1,...,w8/b0,...,b8/kw,kb/tw,tb/side");
}

fn get_num_threads() -> usize {
    // ENGINE_THREADS env var allows pool-friendly limiting
    if let Ok(s) = std::env::var("ENGINE_THREADS") {
        if let Ok(n) = s.parse::<usize>() {
            if n >= 1 { return n; }
        }
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn play_interactive() {
    let mut board = Board::new();
    let mut searcher = Searcher::new(64);
    if let Some(nnue) = load_nnue() {
        searcher.set_nnue(nnue);
    }
    if let Some(tb) = load_egtb() {
        searcher.set_egtb(tb);
    }
    if let Some(book) = load_book() {
        searcher.set_book(book);
    }
    let search_time_ms: u64 = 3000;
    let max_depth: i32 = 30;
    let num_threads = get_num_threads();

    println!("Togyzkumalaq Engine v0.1");
    println!("You play White. Enter pit number 1-9.");
    println!("Type 'quit' to exit, 'undo' to take back.\n");

    let stdin = io::stdin();
    let mut move_history: Vec<(board::UndoInfo, Board)> = Vec::new();

    // Track positions for repetition detection
    searcher.push_game_position(searcher.compute_hash(&board));

    loop {
        println!("{}", board);
        println!();

        if let Some(result) = board.game_result() {
            match result {
                GameResult::Win(Side::White) => println!("White wins!"),
                GameResult::Win(Side::Black) => println!("Black wins!"),
                GameResult::Draw => println!("Draw!"),
            }
            break;
        }

        if board.side_to_move == Side::White {
            // Human's turn
            print!("Your move (pit 1-9): ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            stdin.lock().read_line(&mut input).unwrap();
            let input = input.trim();

            if input == "quit" || input == "q" {
                break;
            }
            if input == "undo" || input == "u" {
                if move_history.len() >= 2 {
                    move_history.pop();
                    let (_, prev_board) = move_history.pop().unwrap();
                    board = prev_board;
                    // Rebuild game history
                    searcher.game_history.truncate(searcher.game_history.len().saturating_sub(2));
                    println!("Undone 2 moves.");
                } else {
                    println!("Nothing to undo.");
                }
                continue;
            }

            let pit: usize = match input.parse::<usize>() {
                Ok(p) if (1..=9).contains(&p) => p - 1,
                _ => {
                    println!("Invalid input. Enter 1-9.");
                    continue;
                }
            };

            if !board.is_valid_move(pit) {
                println!("Invalid move! Pit {} is empty or blocked.", pit + 1);
                continue;
            }

            let saved = board;
            let undo = board.make_move(pit);
            move_history.push((undo, saved));
            searcher.push_game_position(searcher.compute_hash(&board));
            println!("You played pit {}.", pit + 1);
        } else {
            // Engine's turn
            println!("Engine thinking ({} threads)...", num_threads);
            let result = searcher.search_smp(&board, max_depth, search_time_ms, num_threads);

            println!(
                "Engine plays pit {} (score: {}, depth: {}, nodes: {}, time: {}ms)",
                result.best_move + 1,
                result.score,
                result.depth,
                result.nodes,
                result.time_ms,
            );

            let saved = board;
            let undo = board.make_move(result.best_move);
            move_history.push((undo, saved));
            searcher.push_game_position(searcher.compute_hash(&board));
        }
    }
}

fn run_bench() {
    println!("Running benchmark...\n");

    let mut searcher = Searcher::new(64);
    if let Some(nnue) = load_nnue() {
        searcher.set_nnue(nnue);
    }
    let positions = vec![
        ("Initial", Board::new()),
        ("Midgame", {
            let mut b = Board::new();
            let moves = [6, 8, 5, 7, 0, 6, 1, 7, 3, 5];
            for &m in &moves {
                if b.is_valid_move(m) {
                    b.make_move(m);
                }
            }
            b
        }),
        ("Endgame", {
            let mut b = Board::new();
            b.pits[0] = [0, 3, 0, 5, 0, 2, 0, 1, 0];
            b.pits[1] = [2, 0, 4, 0, 1, 0, 3, 0, 2];
            b.kazan = [70, 69];
            b.tuzdyk = [3, 5];
            b
        }),
    ];

    let mut total_nodes = 0u64;
    let mut total_time = 0u64;

    for (name, pos) in &positions {
        println!("Position: {}", name);
        println!("{}", pos);

        let result = searcher.search(pos, 15, 5000);
        total_nodes += result.nodes;
        total_time += result.time_ms;

        println!(
            "Best: pit {}, Score: {}, Depth: {}, Nodes: {}, Time: {}ms, NPS: {}",
            result.best_move + 1,
            result.score,
            result.depth,
            result.nodes,
            result.time_ms,
            if result.time_ms > 0 { result.nodes * 1000 / result.time_ms } else { result.nodes },
        );
        println!();
        searcher.clear();
    }

    println!(
        "Total: {} nodes in {}ms ({} nps)",
        total_nodes,
        total_time,
        if total_time > 0 { total_nodes * 1000 / total_time } else { total_nodes },
    );
}

fn run_perft() {
    println!("Perft (move generation test):\n");

    let board = Board::new();

    for depth in 1..=8 {
        let start = std::time::Instant::now();
        let nodes = perft(&board, depth);
        let elapsed = start.elapsed().as_millis();
        let nps = if elapsed > 0 { nodes * 1000 / elapsed as u64 } else { nodes };
        println!("Depth {}: {} nodes, {}ms, {} nps", depth, nodes, elapsed, nps);
    }
}

fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if board.is_terminal() {
        return 1;
    }

    let mut moves = [0usize; NUM_PITS];
    let num_moves = board.valid_moves_array(&mut moves);
    let mut nodes = 0u64;

    for i in 0..num_moves {
        let mut new_board = *board;
        new_board.make_move(moves[i]);
        nodes += perft(&new_board, depth - 1);
    }
    nodes
}

fn run_selfplay() {
    println!("Engine self-play:\n");

    let mut board = Board::new();
    let mut searcher = Searcher::new(64);
    if let Some(nnue) = load_nnue() {
        searcher.set_nnue(nnue);
    }
    let search_time = 1000u64;
    let max_depth = 20;
    let mut move_num = 0;

    // Track positions for repetition detection
    searcher.push_game_position(searcher.compute_hash(&board));

    loop {
        if let Some(result) = board.game_result() {
            println!("\n{}", board);
            match result {
                GameResult::Win(Side::White) => println!("\nResult: White wins!"),
                GameResult::Win(Side::Black) => println!("\nResult: Black wins!"),
                GameResult::Draw => println!("\nResult: Draw!"),
            }
            println!("Total moves: {}", move_num);
            break;
        }

        let result = searcher.search(&board, max_depth, search_time);

        move_num += 1;
        let side = if board.side_to_move == Side::White { "W" } else { "B" };
        print!("{}. {}{} ({}) ", (move_num + 1) / 2, side, result.best_move + 1, result.score);

        if move_num % 4 == 0 {
            println!();
        }

        board.make_move(result.best_move);
        searcher.push_game_position(searcher.compute_hash(&board));
        io::stdout().flush().unwrap();
    }
}

fn run_match(num_games: u32, time_ms: u64) {
    println!("NNUE vs Handcrafted Eval Match");
    println!("==============================");
    println!("Games: {} (alternating colors)", num_games);
    println!("Time per move: {}ms", time_ms);
    println!();

    let nnue = match load_nnue() {
        Some(n) => n,
        None => {
            println!("Error: nnue_weights.bin not found!");
            return;
        }
    };

    let max_depth = 20;
    let mut nnue_wins = 0u32;
    let mut hce_wins = 0u32;
    let mut draws = 0u32;

    // Load EGTB once, share across all games
    let egtb = load_egtb();

    // Shared zobrist keys for game history (deterministic, same as searcher's)
    let zobrist = search::Searcher::new(1).zobrist;

    for game_num in 0..num_games {
        let nnue_is_white = game_num % 2 == 0;
        let mut board = Board::new();

        let mut nnue_searcher = Searcher::new(16);
        nnue_searcher.set_nnue(NnueNetwork::load(NNUE_PATH).unwrap());
        if let Some(ref tb) = egtb {
            nnue_searcher.set_egtb(tb.clone());
        }
        // Opening book disabled: human moves are weaker than engine search
        nnue_searcher.silent = true;

        let mut hce_searcher = Searcher::new(16);
        hce_searcher.silent = true;

        // Track game positions for repetition detection
        let mut game_hashes: Vec<u64> = Vec::new();
        game_hashes.push(zobrist.hash(&board));

        loop {
            if board.is_terminal() {
                break;
            }

            let is_white_turn = board.side_to_move == Side::White;
            let use_nnue = is_white_turn == nnue_is_white;

            let result = if use_nnue {
                nnue_searcher.game_history = game_hashes.clone();
                nnue_searcher.search(&board, max_depth, time_ms)
            } else {
                hce_searcher.game_history = game_hashes.clone();
                hce_searcher.search(&board, max_depth, time_ms)
            };

            board.make_move(result.best_move);
            game_hashes.push(zobrist.hash(&board));
        }

        match board.game_result() {
            Some(GameResult::Win(Side::White)) => {
                if nnue_is_white { nnue_wins += 1; } else { hce_wins += 1; }
            }
            Some(GameResult::Win(Side::Black)) => {
                if !nnue_is_white { nnue_wins += 1; } else { hce_wins += 1; }
            }
            Some(GameResult::Draw) | None => { draws += 1; }
        }

        let total = game_num + 1;
        if total % 10 == 0 || total == num_games {
            println!(
                "Game {}/{}: NNUE {}-{}-{} HCE ({:.1}%)",
                total, num_games, nnue_wins, draws, hce_wins,
                (nnue_wins as f64 + draws as f64 * 0.5) / total as f64 * 100.0,
            );
        }
    }

    println!("\n==============================");
    println!("Final: NNUE {} - {} - {} HCE", nnue_wins, draws, hce_wins);
    let score = (nnue_wins as f64 + draws as f64 * 0.5) / num_games as f64;
    println!("NNUE score: {:.1}%", score * 100.0);
    if score > 0.5 {
        let elo = -400.0 * (1.0 / score - 1.0).ln() / std::f64::consts::LN_10;
        println!("NNUE Elo advantage: +{:.0}", elo);
    } else if score < 0.5 {
        let elo = -400.0 * (1.0 / (1.0 - score) - 1.0).ln() / std::f64::consts::LN_10;
        println!("HCE Elo advantage: +{:.0}", elo);
    }
}

fn run_match_nnue(weights_a: &str, weights_b: &str, num_games: u32, time_ms: u64) {
    println!("NNUE vs NNUE Match");
    println!("==============================");
    println!("Engine A: {}", weights_a);
    println!("Engine B: {}", weights_b);
    println!("Games: {} (alternating colors, 4 random opening plies)", num_games);
    println!("Time per move: {}ms", time_ms);
    println!();

    let _nnue_a = match NnueNetwork::load(weights_a) {
        Ok(n) => n,
        Err(e) => { println!("Error loading {}: {}", weights_a, e); return; }
    };
    let _nnue_b = match NnueNetwork::load(weights_b) {
        Ok(n) => n,
        Err(e) => { println!("Error loading {}: {}", weights_b, e); return; }
    };

    let max_depth = 20;
    let mut a_wins = 0u32;
    let mut b_wins = 0u32;
    let mut draws = 0u32;

    let egtb = load_egtb();
    let zobrist = search::Searcher::new(1).zobrist;

    // Simple RNG for random openings
    let mut rng_state: u64 = 0xDEADBEEF12345678;
    let mut rng_next = |state: &mut u64| -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    };

    // Games are played in pairs: same opening, alternating colors
    for game_num in 0..num_games {
        let a_is_white = game_num % 2 == 0;
        let mut board = Board::new();

        // Random opening: 4 plies of random moves (same opening for each pair)
        let opening_seed = if game_num % 2 == 0 {
            rng_next(&mut rng_state)
        } else {
            // Reuse same seed as previous game for same opening
            rng_state // state hasn't changed since the paired game used it
        };
        let mut opening_rng = opening_seed;
        for _ in 0..4 {
            if board.is_terminal() { break; }
            let mut moves = [0usize; 9];
            let num_moves = board.valid_moves_array(&mut moves);
            if num_moves == 0 { break; }
            let idx = (rng_next(&mut opening_rng) % num_moves as u64) as usize;
            board.make_move(moves[idx]);
        }

        let mut searcher_a = Searcher::new(16);
        searcher_a.set_nnue(NnueNetwork::load(weights_a).expect("Failed to load A"));
        if let Some(ref tb) = egtb {
            searcher_a.set_egtb(tb.clone());
        }
        searcher_a.silent = true;

        let mut searcher_b = Searcher::new(16);
        searcher_b.set_nnue(NnueNetwork::load(weights_b).expect("Failed to load B"));
        if let Some(ref tb) = egtb {
            searcher_b.set_egtb(tb.clone());
        }
        searcher_b.silent = true;

        let mut game_hashes: Vec<u64> = Vec::new();
        game_hashes.push(zobrist.hash(&board));
        let mut move_count = 0u32;

        loop {
            if board.is_terminal() { break; }
            // Safety: max 300 moves per game to prevent infinite loops
            if move_count >= 200 { break; }
            move_count += 1;

            let is_white_turn = board.side_to_move == Side::White;
            let use_a = is_white_turn == a_is_white;

            let result = if use_a {
                searcher_a.game_history = game_hashes.clone();
                searcher_a.search(&board, max_depth, time_ms)
            } else {
                searcher_b.game_history = game_hashes.clone();
                searcher_b.search(&board, max_depth, time_ms)
            };

            board.make_move(result.best_move);
            game_hashes.push(zobrist.hash(&board));
        }

        match board.game_result() {
            Some(GameResult::Win(Side::White)) => {
                if a_is_white { a_wins += 1; } else { b_wins += 1; }
            }
            Some(GameResult::Win(Side::Black)) => {
                if !a_is_white { a_wins += 1; } else { b_wins += 1; }
            }
            Some(GameResult::Draw) | None => { draws += 1; }
        }

        let total = game_num + 1;
        if total % 10 == 0 || total == num_games {
            let score_a = (a_wins as f64 + draws as f64 * 0.5) / total as f64 * 100.0;
            println!(
                "Game {}/{}: A {}-{}-{} B ({:.1}%)",
                total, num_games, a_wins, draws, b_wins, score_a,
            );
        }
    }

    println!("\n==============================");
    println!("Final: A {} - {} - {} B", a_wins, draws, b_wins);
    let score = (a_wins as f64 + draws as f64 * 0.5) / num_games as f64;
    println!("A score: {:.1}%", score * 100.0);
    if score > 0.5 {
        let elo = -400.0 * (1.0 / score - 1.0).ln() / std::f64::consts::LN_10;
        println!("A Elo advantage: +{:.0}", elo);
    } else if score < 0.5 {
        let elo = -400.0 * (1.0 / (1.0 - score) - 1.0).ln() / std::f64::consts::LN_10;
        println!("B Elo advantage: +{:.0}", elo);
    }
}

/// Persistent stdin/stdout protocol for web server integration.
/// Keeps the Searcher alive between moves so TT and game history persist.
fn run_serve() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let num_threads = get_num_threads();

    // Pool-friendly: 1GB TT per engine (3 engines × 1GB = 3GB total)
    // Reads TT_SIZE_MB env var for override (default 1024 MB)
    let tt_mb: usize = std::env::var("TT_SIZE_MB")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let mut searcher = Searcher::new(tt_mb);
    if let Some(nnue) = load_nnue() {
        searcher.set_nnue(nnue);
    }
    if let Some(tb) = load_egtb() {
        searcher.set_egtb(tb);
    }
    if let Some(book) = load_book() {
        searcher.set_book(book);
    }
    searcher.silent = true;

    // Signal ready
    {
        let mut out = stdout.lock();
        writeln!(out, "ready").unwrap();
        out.flush().unwrap();
    }

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts[0] {
            "newgame" => {
                searcher.clear();
                let mut out = stdout.lock();
                writeln!(out, "ready").unwrap();
                out.flush().unwrap();
            }
            "position" => {
                if parts.len() >= 2 {
                    match board::parse_position(parts[1]) {
                        Ok(board) => {
                            let hash = searcher.compute_hash(&board);
                            searcher.push_game_position(hash);
                            let mut out = stdout.lock();
                            writeln!(out, "ready").unwrap();
                            out.flush().unwrap();
                        }
                        Err(e) => {
                            let mut out = stdout.lock();
                            writeln!(out, "error {}", e).unwrap();
                            out.flush().unwrap();
                        }
                    }
                }
            }
            "go" => {
                let mut time_ms: u64 = 3000;
                let mut pos_str = "";
                let mut nobook: bool = false;
                let mut i = 1;
                while i < parts.len() {
                    match parts[i] {
                        "time" => {
                            if i + 1 < parts.len() {
                                time_ms = parts[i + 1].parse().unwrap_or(3000);
                            }
                            i += 2;
                        }
                        "pos" => {
                            if i + 1 < parts.len() {
                                pos_str = parts[i + 1];
                            }
                            i += 2;
                        }
                        "nobook" => {
                            nobook = true;
                            i += 1;
                        }
                        _ => {
                            i += 1;
                        }
                    }
                }

                match board::parse_position(pos_str) {
                    Ok(board) => {
                        if board.is_terminal() {
                            let result = board.game_result();
                            let result_str = match result {
                                Some(GameResult::Win(Side::White)) => "white_win",
                                Some(GameResult::Win(Side::Black)) => "black_win",
                                Some(GameResult::Draw) => "draw",
                                None => "unknown",
                            };
                            let mut out = stdout.lock();
                            writeln!(out, "terminal {}", result_str).unwrap();
                            out.flush().unwrap();
                        } else {
                            searcher.skip_book = nobook;
                            let result = searcher.search_smp(&board, 30, time_ms, num_threads);
                            searcher.skip_book = false;
                            let chosen_move = result.best_move;

                            // Push resulting position to game history
                            let mut new_board = board;
                            new_board.make_move(chosen_move);
                            let new_hash = searcher.compute_hash(&new_board);
                            searcher.push_game_position(new_hash);

                            let nps = if result.time_ms > 0 {
                                result.nodes * 1000 / result.time_ms
                            } else {
                                result.nodes
                            };

                            let mut out = stdout.lock();
                            writeln!(
                                out,
                                "bestmove {} score {} depth {} nodes {} time {} nps {}",
                                chosen_move,
                                result.score,
                                result.depth,
                                result.nodes,
                                result.time_ms,
                                nps,
                            )
                            .unwrap();
                            out.flush().unwrap();
                        }
                    }
                    Err(e) => {
                        let mut out = stdout.lock();
                        writeln!(out, "error {}", e).unwrap();
                        out.flush().unwrap();
                    }
                }
            }
            "ping" => {
                let mut out = stdout.lock();
                writeln!(out, "pong").unwrap();
                out.flush().unwrap();
            }
            "quit" => break,
            _ => {
                let mut out = stdout.lock();
                writeln!(out, "error unknown command: {}", parts[0]).unwrap();
                out.flush().unwrap();
            }
        }
    }
}

fn run_analyze(pos: &str, time_ms: u64) {
    let board = match board::parse_position(pos) {
        Ok(b) => b,
        Err(e) => {
            println!("{{\"error\":\"{}\"}}", e);
            return;
        }
    };

    if board.is_terminal() {
        let result = board.game_result();
        let result_str = match result {
            Some(GameResult::Win(Side::White)) => "white_win",
            Some(GameResult::Win(Side::Black)) => "black_win",
            Some(GameResult::Draw) => "draw",
            None => "unknown",
        };
        println!("{{\"terminal\":true,\"result\":\"{}\"}}", result_str);
        return;
    }

    let mut searcher = Searcher::new(64);
    searcher.silent = true;
    if let Some(nnue) = load_nnue() {
        searcher.set_nnue(nnue);
    }
    if let Some(tb) = load_egtb() {
        searcher.set_egtb(tb);
    }

    let result = searcher.search(&board, 30, time_ms);

    // Output JSON
    println!(
        "{{\"bestmove\":{},\"score\":{},\"depth\":{},\"nodes\":{},\"time_ms\":{},\"nps\":{}}}",
        result.best_move,
        result.score,
        result.depth,
        result.nodes,
        result.time_ms,
        if result.time_ms > 0 { result.nodes * 1000 / result.time_ms } else { result.nodes },
    );
}

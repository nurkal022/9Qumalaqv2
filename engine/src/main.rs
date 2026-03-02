mod board;
mod datagen;
mod eval;
mod nnue;
mod search;
mod texel;
mod tt;
mod zobrist;

use std::io::{self, BufRead, Write};
use board::{Board, Side, GameResult, NUM_PITS};
use nnue::NnueNetwork;
use search::Searcher;

const NNUE_PATH: &str = "nnue_weights.bin";

fn load_nnue() -> Option<NnueNetwork> {
    if std::path::Path::new(NNUE_PATH).exists() {
        match NnueNetwork::load(NNUE_PATH) {
            Ok(net) => {
                eprintln!("NNUE loaded from {}", NNUE_PATH);
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
            "datagen" => {
                let num_games: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10000);
                let depth: i32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
                let threads: u32 = args.get(4).and_then(|s| s.parse().ok())
                    .unwrap_or(std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(4));
                let prefix = args.get(5).map(|s| s.as_str()).unwrap_or("local");
                let nnue = load_nnue();
                datagen::run_datagen(num_games, depth, 500, threads, nnue, prefix);
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

fn play_interactive() {
    let mut board = Board::new();
    let mut searcher = Searcher::new(64);
    if let Some(nnue) = load_nnue() {
        searcher.set_nnue(nnue);
    }
    let search_time_ms: u64 = 3000;
    let max_depth: i32 = 30;

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
            println!("Engine thinking...");
            let result = searcher.search(&board, max_depth, search_time_ms);

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

    // Shared zobrist keys for game history (deterministic, same as searcher's)
    let zobrist = search::Searcher::new(1).zobrist;

    for game_num in 0..num_games {
        let nnue_is_white = game_num % 2 == 0;
        let mut board = Board::new();

        let mut nnue_searcher = Searcher::new(16);
        nnue_searcher.set_nnue(NnueNetwork::load(NNUE_PATH).unwrap());
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

/// Parse position string: "w0,w1,...,w8/b0,...,b8/kw,kb/tw,tb/side"
fn parse_position(pos: &str) -> Result<Board, String> {
    let parts: Vec<&str> = pos.split('/').collect();
    if parts.len() != 5 {
        return Err(format!("Expected 5 parts separated by '/', got {}", parts.len()));
    }

    let white_pits: Vec<u8> = parts[0].split(',').map(|s| s.parse().unwrap_or(0)).collect();
    let black_pits: Vec<u8> = parts[1].split(',').map(|s| s.parse().unwrap_or(0)).collect();
    let kazans: Vec<u8> = parts[2].split(',').map(|s| s.parse().unwrap_or(0)).collect();
    let tuzdyks: Vec<i8> = parts[3].split(',').map(|s| s.parse().unwrap_or(-1)).collect();
    let side: u8 = parts[4].parse().unwrap_or(0);

    if white_pits.len() != 9 || black_pits.len() != 9 || kazans.len() != 2 || tuzdyks.len() != 2 {
        return Err("Invalid array lengths".into());
    }

    let mut board = Board::new();
    for i in 0..9 {
        board.pits[0][i] = white_pits[i];
        board.pits[1][i] = black_pits[i];
    }
    board.kazan[0] = kazans[0];
    board.kazan[1] = kazans[1];
    board.tuzdyk[0] = tuzdyks[0];
    board.tuzdyk[1] = tuzdyks[1];
    board.side_to_move = if side == 0 { Side::White } else { Side::Black };

    Ok(board)
}

/// Persistent stdin/stdout protocol for web server integration.
/// Keeps the Searcher alive between moves so TT and game history persist.
fn run_serve() {
    let stdin = io::stdin();
    let stdout = io::stdout();

    let mut searcher = Searcher::new(64);
    if let Some(nnue) = load_nnue() {
        searcher.set_nnue(nnue);
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
                    match parse_position(parts[1]) {
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
                        _ => {
                            i += 1;
                        }
                    }
                }

                match parse_position(pos_str) {
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
                            let result = searcher.search(&board, 30, time_ms);

                            // Push resulting position to game history
                            let mut new_board = board;
                            new_board.make_move(result.best_move);
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
                                result.best_move,
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
    let board = match parse_position(pos) {
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

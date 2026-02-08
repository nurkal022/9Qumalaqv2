mod board;
mod eval;
mod search;
mod tt;
mod zobrist;

use std::io::{self, BufRead, Write};
use board::{Board, Side, GameResult, NUM_PITS};
use search::Searcher;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "bench" => run_bench(),
            "play" => play_interactive(),
            "perft" => run_perft(),
            "selfplay" => run_selfplay(),
            _ => print_usage(),
        }
    } else {
        print_usage();
    }
}

fn print_usage() {
    println!("Togyzkumalaq Engine v0.1");
    println!();
    println!("Usage:");
    println!("  togyzkumalaq-engine play      - Play against the engine");
    println!("  togyzkumalaq-engine bench      - Run benchmark");
    println!("  togyzkumalaq-engine perft      - Count nodes at each depth");
    println!("  togyzkumalaq-engine selfplay   - Engine plays against itself");
}

fn play_interactive() {
    let mut board = Board::new();
    let mut searcher = Searcher::new(64);
    let search_time_ms: u64 = 3000;
    let max_depth: i32 = 30;

    println!("Togyzkumalaq Engine v0.1");
    println!("You play White. Enter pit number 1-9.");
    println!("Type 'quit' to exit, 'undo' to take back.\n");

    let stdin = io::stdin();
    let mut move_history: Vec<(board::UndoInfo, Board)> = Vec::new();

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
        }
    }
}

fn run_bench() {
    println!("Running benchmark...\n");

    let mut searcher = Searcher::new(64);
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
    let search_time = 1000u64;
    let max_depth = 20;
    let mut move_num = 0;

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
        io::stdout().flush().unwrap();
    }
}

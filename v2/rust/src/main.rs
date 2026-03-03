/// V2 Gumbel AlphaZero Engine for Тоғызқұмалақ
///
/// CLI commands:
///   selfplay  - Generate self-play training data
///   arena     - Match between two models
///   serve     - Persistent protocol for web integration
///   test      - Run a quick smoke test

mod arena;
mod board;
mod data_writer;
mod features;
mod gumbel;
mod mcts;
mod network;
mod selfplay;
mod serve;
mod zobrist;

use clap::{Parser, Subcommand};
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "v2_engine")]
#[command(about = "Gumbel AlphaZero engine for Togyz Kumalak")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate self-play games for training
    Selfplay {
        /// Path to ONNX model
        #[arg(long)]
        model: String,

        /// Number of games to generate
        #[arg(long, default_value = "1000")]
        games: u32,

        /// MCTS simulations per move
        #[arg(long, default_value = "100")]
        simulations: u32,

        /// Initial Gumbel candidates
        #[arg(long, default_value = "8")]
        candidates: usize,

        /// Number of threads
        #[arg(long, default_value = "4")]
        threads: usize,

        /// Output data file path
        #[arg(long)]
        output: String,

        /// Temperature moves (first N moves with temperature)
        #[arg(long, default_value = "15")]
        temperature_moves: usize,

        /// Sigma scale for Q-value transformation
        #[arg(long, default_value = "50.0")]
        sigma_scale: f32,
    },

    /// Run arena match between two models
    Arena {
        /// Path to new ONNX model
        #[arg(long)]
        model_new: String,

        /// Path to old ONNX model
        #[arg(long)]
        model_old: String,

        /// Number of games
        #[arg(long, default_value = "200")]
        games: u32,

        /// MCTS simulations per move
        #[arg(long, default_value = "100")]
        simulations: u32,

        /// Initial Gumbel candidates
        #[arg(long, default_value = "8")]
        candidates: usize,

        /// Sigma scale
        #[arg(long, default_value = "50.0")]
        sigma_scale: f32,
    },

    /// Persistent engine protocol (stdin/stdout JSON)
    Serve {
        /// Path to ONNX model
        #[arg(long)]
        model: String,

        /// MCTS simulations per move
        #[arg(long, default_value = "100")]
        simulations: u32,

        /// Initial Gumbel candidates
        #[arg(long, default_value = "8")]
        candidates: usize,

        /// Sigma scale
        #[arg(long, default_value = "50.0")]
        sigma_scale: f32,
    },

    /// Quick smoke test (no model needed)
    Test,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Selfplay {
            model,
            games,
            simulations,
            candidates,
            threads,
            output,
            temperature_moves,
            sigma_scale,
        } => {
            eprintln!("=== V2 Self-Play ===");
            eprintln!("Model: {}", model);
            eprintln!("Games: {}, Sims: {}, Candidates: {}", games, simulations, candidates);
            eprintln!("Threads: {}, Output: {}", threads, output);

            let network = Arc::new(
                network::Network::load(&model).expect("Failed to load ONNX model"),
            );

            selfplay::generate_games(
                network,
                games,
                simulations,
                candidates,
                sigma_scale,
                threads,
                temperature_moves,
                &output,
            );
        }

        Commands::Arena {
            model_new,
            model_old,
            games,
            simulations,
            candidates,
            sigma_scale,
        } => {
            eprintln!("=== V2 Arena ===");
            eprintln!("New: {} vs Old: {}", model_new, model_old);
            eprintln!("Games: {}, Sims: {}", games, simulations);

            let new_net = Arc::new(
                network::Network::load(&model_new).expect("Failed to load new model"),
            );
            let old_net = Arc::new(
                network::Network::load(&model_old).expect("Failed to load old model"),
            );

            let (new_wins, old_wins, draws) = arena::run_arena(
                new_net, old_net, games, simulations, candidates, sigma_scale,
            );

            let total = new_wins + old_wins + draws;
            let winrate = (new_wins as f64 + 0.5 * draws as f64) / total as f64;
            let elo = arena::winrate_to_elo(winrate);

            println!("=== Arena Results ===");
            println!(
                "New: {} wins | Old: {} wins | Draws: {}",
                new_wins, old_wins, draws
            );
            println!("Winrate: {:.1}% | Elo diff: {:+.0}", winrate * 100.0, elo);

            if winrate >= 0.55 {
                println!("ACCEPTED: New model is stronger.");
            } else {
                println!("REJECTED: New model not significantly stronger.");
            }
        }

        Commands::Serve {
            model,
            simulations,
            candidates,
            sigma_scale,
        } => {
            eprintln!("V2 Engine serving on stdin/stdout...");
            let network = Arc::new(
                network::Network::load(&model).expect("Failed to load ONNX model"),
            );
            serve::serve_loop(network, simulations, candidates, sigma_scale);
        }

        Commands::Test => {
            run_smoke_test();
        }
    }
}

fn run_smoke_test() {
    println!("=== V2 Smoke Test ===");

    // Test 1: Board rules
    println!("\n[1] Board rules...");
    let mut b = board::Board::new();
    assert_eq!(b.legal_moves().len(), 9);
    assert_eq!(b.stones_on_board(), 162);
    b.make_move(6); // White plays pit 7
    assert_eq!(b.kazan[0], 10); // capture
    assert_eq!(b.side_to_move, board::Side::Black);
    println!("  Board rules: OK");

    // Test 2: Features
    println!("\n[2] Feature extraction...");
    let board = board::Board::new();
    let features = features::board_to_features(&board);
    assert_eq!(features.len(), features::FEATURE_SIZE);
    println!("  Features ({} dims): OK", features::FEATURE_SIZE);

    // Test 3: MCTS with random network
    println!("\n[3] MCTS (random network, 50 sims)...");
    let board = board::Board::new();
    let (move_visits, value) = mcts::mcts_search_random(&board, 50);
    println!("  Move visits: {:?}", move_visits);
    println!("  Root value: {:.3}", value);
    assert!(!move_visits.is_empty());
    println!("  MCTS: OK");

    // Test 4: Gumbel search with random
    println!("\n[4] Gumbel search (random, 32 sims, 4 candidates)...");
    let board = board::Board::new();
    let results = gumbel::gumbel_search_random(&board, 32, 4);
    println!("  Results: {:?}", results);
    assert!(!results.is_empty());
    println!("  Gumbel: OK");

    // Test 5: Play a full game with random MCTS
    println!("\n[5] Full random game...");
    let mut board = board::Board::new();
    let mut moves = 0;
    while !board.is_terminal() && moves < 300 {
        let (move_visits, _) = mcts::mcts_search_random(&board, 10);
        if move_visits.is_empty() {
            break;
        }
        let best = move_visits
            .iter()
            .max_by_key(|(_, v)| *v)
            .unwrap()
            .0;
        board.make_move(best);
        moves += 1;
    }
    let total = board.stones_on_board() + board.kazan[0] as u32 + board.kazan[1] as u32;
    println!(
        "  Game ended after {} moves. Kazan: {} - {}. Terminal: {}",
        moves,
        board.kazan[0],
        board.kazan[1],
        board.is_terminal()
    );
    assert_eq!(total, 162, "Stone conservation violated!");
    println!("  Full game: OK");

    // Test 6: Zobrist hashing
    println!("\n[6] Zobrist hashing...");
    let keys = zobrist::ZobristKeys::new();
    let b1 = board::Board::new();
    let mut b2 = board::Board::new();
    b2.pits[0][0] = 10;
    assert_ne!(keys.hash(&b1), keys.hash(&b2));
    assert_eq!(keys.hash(&b1), keys.hash(&board::Board::new()));
    println!("  Zobrist: OK");

    println!("\n=== All tests passed! ===");
}

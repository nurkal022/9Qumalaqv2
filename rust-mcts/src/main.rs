mod board;
mod encoding;
mod evaluator;
mod mcts;
mod replay_buffer;
mod self_play;

use clap::Parser;
use crossbeam_channel;
use std::thread;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "rust-mcts")]
#[command(about = "Fast MCTS self-play for Togyz Kumalak")]
struct Args {
    /// Path to ONNX model file
    #[arg(long, default_value = "model.onnx")]
    model: String,

    /// Number of games to play
    #[arg(long, default_value_t = 100)]
    games: u32,

    /// MCTS simulations per move
    #[arg(long, default_value_t = 800)]
    sims: u32,

    /// Number of worker threads
    #[arg(long, default_value_t = 20)]
    workers: u32,

    /// GPU inference batch size
    #[arg(long, default_value_t = 128)]
    batch_size: usize,

    /// Output replay buffer file
    #[arg(long, default_value = "replay_buffer.bin")]
    output: String,

    /// PUCT exploration constant
    #[arg(long, default_value_t = 1.5)]
    cpuct: f32,

    /// Dirichlet noise alpha
    #[arg(long, default_value_t = 0.3)]
    dirichlet_alpha: f32,

    /// Temperature threshold (moves before switching to greedy)
    #[arg(long, default_value_t = 15)]
    temp_threshold: u32,

    /// Use CPU-only inference (no GPU)
    #[arg(long)]
    cpu: bool,

    /// Use dummy evaluator (uniform policy, for testing)
    #[arg(long)]
    dummy: bool,
}

fn main() {
    let args = Args::parse();

    eprintln!("=== Rust MCTS Self-Play ===");
    eprintln!("Model: {}", args.model);
    eprintln!("Games: {}, Sims: {}, Workers: {}", args.games, args.sims, args.workers);
    eprintln!("Batch size: {}, c_puct: {}", args.batch_size, args.cpuct);
    eprintln!("Output: {}", args.output);
    eprintln!();

    let mcts_config = mcts::MctsConfig {
        num_simulations: args.sims,
        c_puct: args.cpuct,
        dirichlet_alpha: args.dirichlet_alpha,
        dirichlet_epsilon: 0.25,
        temperature_threshold: args.temp_threshold,
        virtual_batch: 128,
    };

    // Channel for eval requests (workers -> evaluator)
    let (eval_tx, eval_rx) = crossbeam_channel::unbounded::<evaluator::EvalRequest>();

    // Channel for game results (workers -> main)
    let (result_tx, result_rx) = crossbeam_channel::unbounded::<Vec<replay_buffer::TrainingRecord>>();

    // Start evaluator thread
    let eval_config = evaluator::EvaluatorConfig {
        model_path: args.model.clone(),
        batch_size: args.batch_size,
        max_wait_us: 200,
        use_gpu: !args.cpu,
    };

    let eval_handle = if args.dummy {
        thread::spawn(move || evaluator::dummy_evaluator_loop(eval_rx))
    } else {
        thread::spawn(move || evaluator::evaluator_loop(eval_rx, eval_config))
    };

    // Distribute games across workers
    let games_per_worker = args.games / args.workers;
    let remainder = args.games % args.workers;

    let start = Instant::now();

    // Spawn worker threads
    let mut worker_handles = Vec::new();
    for w in 0..args.workers {
        let config = mcts_config.clone();
        let tx = eval_tx.clone();
        let res_tx = result_tx.clone();
        let n = games_per_worker + if w < remainder { 1 } else { 0 };

        let handle = thread::spawn(move || {
            self_play::worker_loop(config, tx, res_tx, n, w);
        });
        worker_handles.push(handle);
    }

    // Drop our copy of senders so evaluator/collector know when done
    drop(eval_tx);
    drop(result_tx);

    // Collect results
    let mut all_records: Vec<replay_buffer::TrainingRecord> = Vec::new();
    let mut games_completed = 0u32;

    for game_records in result_rx {
        games_completed += 1;
        let n = game_records.len();
        all_records.extend(game_records);

        if games_completed % 10 == 0 || games_completed == args.games {
            let elapsed = start.elapsed().as_secs_f64();
            let gps = games_completed as f64 / elapsed;
            eprintln!(
                "[{}/{}] games | {} positions | {:.1} games/sec | {:.0}s elapsed",
                games_completed, args.games, all_records.len(), gps, elapsed,
            );
        }
    }

    // Wait for workers
    for h in worker_handles {
        let _ = h.join();
    }

    // Write replay buffer
    if let Err(e) = replay_buffer::write_records(&args.output, &all_records) {
        eprintln!("Error writing replay buffer: {}", e);
        std::process::exit(1);
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!();
    eprintln!("=== Done ===");
    eprintln!("Games: {}", games_completed);
    eprintln!("Positions: {}", all_records.len());
    eprintln!("Time: {:.1}s ({:.1} games/sec)", elapsed, games_completed as f64 / elapsed);
    eprintln!("Output: {}", args.output);

    // Evaluator thread will exit when all eval_tx clones are dropped
    let _ = eval_handle.join();
}

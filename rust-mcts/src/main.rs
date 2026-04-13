mod board;
mod encoding;
mod eval_vs_engine;
mod evaluator;
mod gumbel;
mod league;
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

    /// MCTS simulations per move (full search; fast search uses sims/5)
    #[arg(long, default_value_t = 200)]
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

    /// Dirichlet noise alpha (10/branching_factor ≈ 1.1 for 9 moves)
    #[arg(long, default_value_t = 1.1)]
    dirichlet_alpha: f32,

    /// Temperature threshold (moves at τ=1.0, then linear decay to 0.3)
    #[arg(long, default_value_t = 25)]
    temp_threshold: u32,

    /// Use CPU-only inference (no GPU)
    #[arg(long)]
    cpu: bool,

    /// Use dummy evaluator (uniform policy, for testing)
    #[arg(long)]
    dummy: bool,

    /// Eval mode: play MCTS vs engine instead of selfplay
    #[arg(long)]
    eval: bool,

    /// Path to engine binary (for eval mode)
    #[arg(long, default_value = "")]
    engine: String,

    /// Engine time per move in ms (for eval mode)
    #[arg(long, default_value_t = 500)]
    engine_time: u64,

    /// Simulations for eval (can differ from selfplay sims)
    #[arg(long, default_value_t = 0)]
    eval_sims: u32,

    /// Serve mode: stdin/stdout protocol (drop-in replacement for NNUE engine)
    #[arg(long)]
    serve: bool,

    /// League mode: mixed selfplay + engine games
    #[arg(long)]
    league: bool,

    /// Number of engine games (league mode)
    #[arg(long, default_value_t = 40)]
    engine_games: u32,

    /// Number of parallel engine subprocess instances (league mode)
    #[arg(long, default_value_t = 4)]
    engine_workers: u32,

    /// Output file for engine game records (league mode)
    #[arg(long, default_value = "engine_buffer.bin")]
    engine_output: String,
}

fn main() {
    let args = Args::parse();

    if args.serve {
        run_serve(&args);
        return;
    }
    if args.eval {
        run_eval(&args);
        return;
    }
    if args.league {
        run_league(&args);
        return;
    }

    eprintln!("=== Rust Gumbel MCTS Self-Play ===");
    eprintln!("Model: {}", args.model);
    eprintln!("Games: {}, Workers: {}", args.games, args.workers);
    eprintln!("Batch size: {}, Temp threshold: {}", args.batch_size, args.temp_threshold);
    eprintln!("Playout cap: 25% full Gumbel, 75% raw policy");
    eprintln!("Output: {}", args.output);
    eprintln!();

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
    let temp_threshold = args.temp_threshold;

    // Spawn worker threads (Gumbel selfplay)
    let mut worker_handles = Vec::new();
    for w in 0..args.workers {
        let tx = eval_tx.clone();
        let res_tx = result_tx.clone();
        let n = games_per_worker + if w < remainder { 1 } else { 0 };

        let handle = thread::spawn(move || {
            self_play::worker_loop(tx, res_tx, n, w, temp_threshold);
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

/// Serve mode: stdin/stdout protocol, drop-in replacement for NNUE engine.
/// Accepts same commands: newgame, position, go pos ... time ...
fn run_serve(args: &Args) {
    use std::io::{self, BufRead, Write};

    eprintln!("MCTS serve mode starting...");
    eprintln!("Model: {}", args.model);

    // Start evaluator thread (CPU or GPU)
    let (eval_tx, eval_rx) = crossbeam_channel::unbounded::<evaluator::EvalRequest>();
    let eval_config = evaluator::EvaluatorConfig {
        model_path: args.model.clone(),
        batch_size: args.batch_size,
        max_wait_us: 5000, // longer wait for serve (low throughput)
        use_gpu: !args.cpu,
    };
    let _eval_handle = thread::spawn(move || evaluator::evaluator_loop(eval_rx, eval_config));

    let ctx = eval_vs_engine::EvalContext::new(eval_tx);

    let stdin = io::stdin();
    let stdout = io::stdout();

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
        if line.is_empty() { continue; }

        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts[0] {
            "newgame" => {
                let mut out = stdout.lock();
                writeln!(out, "ready").unwrap();
                out.flush().unwrap();
            }
            "position" => {
                let mut out = stdout.lock();
                writeln!(out, "ready").unwrap();
                out.flush().unwrap();
            }
            "go" => {
                // Parse: go pos W_PITS/B_PITS/KW,KB/TW,TB/SIDE time MS
                let mut pos_str = "";
                let mut _time_ms: u64 = 1000;
                let mut i = 1;
                while i < parts.len() {
                    match parts[i] {
                        "pos" => {
                            if i + 1 < parts.len() { pos_str = parts[i + 1]; }
                            i += 2;
                        }
                        "time" => {
                            if i + 1 < parts.len() {
                                _time_ms = parts[i + 1].parse().unwrap_or(1000);
                            }
                            i += 2;
                        }
                        _ => { i += 1; }
                    }
                }

                // Parse position
                match parse_position(pos_str) {
                    Ok(board) => {
                        if board.is_terminal() {
                            let result_str = match board.game_result() {
                                Some(board::GameResult::Win(board::Side::White)) => "white_win",
                                Some(board::GameResult::Win(board::Side::Black)) => "black_win",
                                Some(board::GameResult::Draw) => "draw",
                                None => "unknown",
                            };
                            let mut out = stdout.lock();
                            writeln!(out, "terminal {}", result_str).unwrap();
                            out.flush().unwrap();
                        } else {
                            // Use Gumbel 1-ply to find best move
                            let t0 = Instant::now();
                            let action = ctx.best_move(&board);
                            let elapsed_ms = t0.elapsed().as_millis() as u64;

                            let mut out = stdout.lock();
                            writeln!(out, "bestmove {} score 0 depth 1 nodes 10 time {} nps 100",
                                action, elapsed_ms).unwrap();
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
            "quit" => break,
            _ => {}
        }
    }
}

/// Parse position string: "w0,w1,...,w8/b0,...,b8/kw,kb/tw,tb/side"
fn parse_position(pos: &str) -> Result<board::Board, String> {
    let parts: Vec<&str> = pos.split('/').collect();
    if parts.len() != 5 {
        return Err(format!("Expected 5 parts, got {}", parts.len()));
    }
    let white_pits: Vec<u8> = parts[0].split(',').map(|s| s.parse().unwrap_or(0)).collect();
    let black_pits: Vec<u8> = parts[1].split(',').map(|s| s.parse().unwrap_or(0)).collect();
    let kazans: Vec<u8> = parts[2].split(',').map(|s| s.parse().unwrap_or(0)).collect();
    let tuzdyks: Vec<i8> = parts[3].split(',').map(|s| s.parse().unwrap_or(-1)).collect();
    let side: u8 = parts[4].parse().unwrap_or(0);

    if white_pits.len() != 9 || black_pits.len() != 9 {
        return Err("Invalid pit array".into());
    }

    let mut b = board::Board::new();
    for i in 0..9 {
        b.pits[0][i] = white_pits[i];
        b.pits[1][i] = black_pits[i];
    }
    b.kazan[0] = kazans.get(0).copied().unwrap_or(0);
    b.kazan[1] = kazans.get(1).copied().unwrap_or(0);
    b.tuzdyk[0] = tuzdyks.get(0).copied().unwrap_or(-1);
    b.tuzdyk[1] = tuzdyks.get(1).copied().unwrap_or(-1);
    b.side_to_move = if side == 0 { board::Side::White } else { board::Side::Black };
    Ok(b)
}

/// League mode: mixed selfplay + engine games, two output files
fn run_league(args: &Args) {
    let selfplay_games = args.games;
    let engine_games = args.engine_games;

    eprintln!("=== League Training ===");
    eprintln!("Model: {}", args.model);
    eprintln!("Selfplay: {} games, {} workers", selfplay_games, args.workers);
    eprintln!("Engine: {} games, {} workers, {}ms/move", engine_games, args.engine_workers, args.engine_time);
    eprintln!();

    // Shared evaluator
    let (eval_tx, eval_rx) = crossbeam_channel::unbounded::<evaluator::EvalRequest>();
    let eval_config = evaluator::EvaluatorConfig {
        model_path: args.model.clone(),
        batch_size: args.batch_size,
        max_wait_us: 200,
        use_gpu: !args.cpu,
    };
    let eval_handle = thread::spawn(move || evaluator::evaluator_loop(eval_rx, eval_config));

    // Separate result channels for selfplay and engine
    let (sp_result_tx, sp_result_rx) = crossbeam_channel::unbounded::<Vec<replay_buffer::TrainingRecord>>();
    let (eng_result_tx, eng_result_rx) = crossbeam_channel::unbounded::<Vec<replay_buffer::TrainingRecord>>();

    let start = Instant::now();
    let temp_threshold = args.temp_threshold;

    // Spawn selfplay workers
    let sp_per_worker = selfplay_games / args.workers;
    let sp_remainder = selfplay_games % args.workers;
    let mut handles = Vec::new();

    for w in 0..args.workers {
        let tx = eval_tx.clone();
        let res_tx = sp_result_tx.clone();
        let n = sp_per_worker + if w < sp_remainder { 1 } else { 0 };
        handles.push(thread::spawn(move || {
            self_play::worker_loop(tx, res_tx, n, w, temp_threshold);
        }));
    }

    // Spawn engine workers
    let engine_path = args.engine.clone();
    let engine_dir = std::path::Path::new(&engine_path)
        .parent().unwrap_or(std::path::Path::new("."))
        .join("..").join("..");
    let resource_dir = engine_dir.canonicalize().unwrap_or(engine_dir);
    let engine_time = args.engine_time;

    let eng_per_worker = engine_games / args.engine_workers;
    let eng_remainder = engine_games % args.engine_workers;

    for w in 0..args.engine_workers {
        let tx = eval_tx.clone();
        let res_tx = eng_result_tx.clone();
        let n = eng_per_worker + if w < eng_remainder { 1 } else { 0 };
        let ep = engine_path.clone();
        let rd = resource_dir.clone();
        handles.push(thread::spawn(move || {
            league::engine_worker_loop(tx, res_tx, &ep, &rd, engine_time, n, w, temp_threshold);
        }));
    }

    // Drop sender clones
    drop(eval_tx);
    drop(sp_result_tx);
    drop(eng_result_tx);

    // Collect selfplay results
    let mut sp_records: Vec<replay_buffer::TrainingRecord> = Vec::new();
    let mut sp_games = 0u32;
    for game_records in sp_result_rx {
        sp_games += 1;
        sp_records.extend(game_records);
        if sp_games % 50 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!("[selfplay] {}/{} games, {} positions, {:.0}s",
                sp_games, selfplay_games, sp_records.len(), elapsed);
        }
    }

    // Collect engine results
    let mut eng_records: Vec<replay_buffer::TrainingRecord> = Vec::new();
    let mut eng_games = 0u32;
    for game_records in eng_result_rx {
        eng_games += 1;
        eng_records.extend(game_records);
    }

    // Wait for all threads
    for h in handles {
        let _ = h.join();
    }

    // Write two output files
    if let Err(e) = replay_buffer::write_records(&args.output, &sp_records) {
        eprintln!("Error writing selfplay buffer: {}", e);
    }
    if let Err(e) = replay_buffer::write_records(&args.engine_output, &eng_records) {
        eprintln!("Error writing engine buffer: {}", e);
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!();
    eprintln!("=== League Done ===");
    eprintln!("Selfplay: {} games, {} positions", sp_games, sp_records.len());
    eprintln!("Engine:   {} games, {} positions", eng_games, eng_records.len());
    eprintln!("Time: {:.1}s", elapsed);

    let _ = eval_handle.join();
}

/// Eval mode: play MCTS model vs engine, report results as JSON to stdout
fn run_eval(args: &Args) {
    use std::process::{Command, Stdio};
    use std::io::{BufRead, BufReader};

    let num_pairs = args.games; // games = number of color pairs
    let eval_sims = if args.eval_sims > 0 { args.eval_sims } else { args.sims };

    eprintln!("=== Rust MCTS Eval vs Engine ===");
    eprintln!("Model: {}, Sims: {}, Pairs: {}", args.model, eval_sims, num_pairs);
    eprintln!("Engine: {}, Time: {}ms", args.engine, args.engine_time);

    let eval_config = mcts::MctsConfig {
        num_simulations: eval_sims,
        c_puct: 2.5, // match Python eval c_puct
        dirichlet_alpha: 0.0, // no noise for eval
        dirichlet_epsilon: 0.0,
        temperature_threshold: 0, // always greedy
        virtual_batch: 128,
    };

    // Start evaluator thread
    let (eval_tx, eval_rx) = crossbeam_channel::unbounded::<evaluator::EvalRequest>();
    let model_path = args.model.clone();
    let batch_size = args.batch_size;
    let use_gpu = !args.cpu;

    let eval_handle = thread::spawn(move || {
        evaluator::evaluator_loop(eval_rx, evaluator::EvaluatorConfig {
            model_path,
            batch_size,
            max_wait_us: 2000,
            use_gpu,
        })
    });

    let ctx = eval_vs_engine::EvalContext::new(eval_tx.clone());

    // Find engine binary and its resource directory
    let engine_path = if args.engine.is_empty() {
        // Try to find engine relative to binary
        let mut p = std::env::current_exe().unwrap_or_default();
        p.pop(); p.pop(); p.pop(); // target/release/ -> project root
        p.push("engine/target/release/togyzkumalaq-engine");
        if p.exists() {
            p.to_string_lossy().to_string()
        } else {
            eprintln!("ERROR: No engine specified. Use --engine <path>");
            std::process::exit(1);
        }
    } else {
        args.engine.clone()
    };

    let engine_dir = std::path::Path::new(&engine_path).parent().unwrap_or(std::path::Path::new("."));
    // Engine resources are in engine/ dir (3 dirs up from target/release/binary)
    let resource_dir = engine_dir.join("..").join("..").canonicalize().unwrap_or_else(|_| engine_dir.to_path_buf());
    eprintln!("Engine resources dir: {:?}", resource_dir);

    // Create temp dir with symlinks to engine resources
    let tmpdir = tempfile::tempdir().expect("Failed to create temp dir");
    for f in &["nnue_weights.bin", "egtb.bin", "opening_book.txt"] {
        let src = resource_dir.join(f);
        if src.exists() {
            let _ = std::os::unix::fs::symlink(&src, tmpdir.path().join(f));
        }
    }

    // Start engine process
    let mut engine_proc = Command::new(&engine_path)
        .arg("serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .current_dir(tmpdir.path())
        .spawn()
        .expect("Failed to start engine");

    let mut engine_stdin = engine_proc.stdin.take().unwrap();
    let mut engine_stdout = BufReader::new(engine_proc.stdout.take().unwrap());

    // Wait for "ready"
    let mut ready_line = String::new();
    let _ = engine_stdout.read_line(&mut ready_line);

    let start = Instant::now();
    let mut wins = 0u32;
    let mut draws = 0u32;
    let mut losses = 0u32;
    let mut pair_wins = 0u32;
    let mut pair_draws = 0u32;
    let mut pair_losses = 0u32;

    for pair_id in 0..num_pairs {
        let mut pair_score: f32 = 0.0;

        for color in 0..2 {
            let mcts_is_white = color == 0;
            let outcome = eval_vs_engine::play_eval_game(
                &ctx, eval_sims,
                &mut engine_stdin, &mut engine_stdout,
                mcts_is_white, args.engine_time,
            );

            match outcome {
                eval_vs_engine::GameOutcome::MctsWin => { wins += 1; pair_score += 1.0; }
                eval_vs_engine::GameOutcome::Draw => { draws += 1; pair_score += 0.5; }
                eval_vs_engine::GameOutcome::MctsLoss => { losses += 1; }
            }
        }

        if pair_score > 1.0 { pair_wins += 1; }
        else if pair_score == 1.0 { pair_draws += 1; }
        else { pair_losses += 1; }

        let total = wins + draws + losses;
        let wr = (wins as f32 + 0.5 * draws as f32) / total as f32 * 100.0;

        if (pair_id + 1) % 5 == 0 || pair_id + 1 == num_pairs {
            eprintln!(
                "  [{}/{}] pairs | {}W-{}D-{}L = {:.1}% | pairs: {}W-{}D-{}L",
                pair_id + 1, num_pairs, wins, draws, losses, wr,
                pair_wins, pair_draws, pair_losses,
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total = wins + draws + losses;
    let wr = (wins as f32 + 0.5 * draws as f32) / total as f32 * 100.0;

    // Print JSON result to stdout (for Python to parse)
    println!(
        "{{\"wins\":{},\"draws\":{},\"losses\":{},\"winrate\":{:.1},\"pair_wins\":{},\"pair_draws\":{},\"pair_losses\":{},\"time\":{:.0}}}",
        wins, draws, losses, wr, pair_wins, pair_draws, pair_losses, elapsed
    );

    eprintln!("=== Eval Done in {:.0}s ===", elapsed);

    let _ = engine_proc.kill();
    // Drop all senders so evaluator thread exits
    drop(ctx);
    drop(eval_tx);
    let _ = eval_handle.join();
}

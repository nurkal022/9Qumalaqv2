/// League training: model plays against engine subprocess.
/// Each engine worker spawns its own engine process and plays games sequentially.
/// Uses Gumbel MCTS for model moves, engine serve protocol for opponent moves.

use crate::board::{Board, GameResult, Side, NUM_PITS};
use crate::encoding::encode_state;
use crate::evaluator::EvalRequest;
use crate::gumbel::{self, GumbelContext};
use crate::replay_buffer::TrainingRecord;
use crossbeam_channel::Sender;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};

/// Engine subprocess wrapper with serve protocol
pub struct EngineProcess {
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    proc: Child,
}

impl EngineProcess {
    pub fn spawn(engine_path: &str, resource_dir: &std::path::Path) -> Self {
        // Create temp dir with symlinks to resources
        let tmpdir = tempfile::tempdir().expect("Failed to create temp dir");
        for f in &["nnue_weights.bin", "egtb.bin", "opening_book.txt"] {
            let src = resource_dir.join(f);
            if src.exists() {
                let _ = std::os::unix::fs::symlink(&src, tmpdir.path().join(f));
            }
        }

        let mut proc = Command::new(engine_path)
            .arg("serve")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .current_dir(tmpdir.path())
            .spawn()
            .expect("Failed to start engine");

        let stdin = proc.stdin.take().unwrap();
        let mut stdout = BufReader::new(proc.stdout.take().unwrap());

        // Wait for "ready"
        let mut line = String::new();
        let _ = stdout.read_line(&mut line);

        // Leak tmpdir so it persists (cleaned up when process exits)
        std::mem::forget(tmpdir);

        EngineProcess { stdin, stdout, proc }
    }

    pub fn new_game(&mut self) {
        writeln!(self.stdin, "newgame").unwrap();
        self.stdin.flush().unwrap();
        let mut line = String::new();
        let _ = self.stdout.read_line(&mut line); // "ready"
    }

    pub fn get_move(&mut self, board: &Board, time_ms: u64) -> Option<usize> {
        let pos_str = board_to_position_string(board);
        writeln!(self.stdin, "go pos {} time {}", pos_str, time_ms).unwrap();
        self.stdin.flush().unwrap();

        let mut line = String::new();
        let _ = self.stdout.read_line(&mut line);
        let response = line.trim();

        if response.starts_with("bestmove") {
            let parts: Vec<&str> = response.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(m) = parts[1].parse::<usize>() {
                    return Some(m);
                }
            }
        }
        None
    }
}

impl Drop for EngineProcess {
    fn drop(&mut self) {
        let _ = self.proc.kill();
    }
}

fn board_to_position_string(board: &Board) -> String {
    let white: Vec<String> = (0..NUM_PITS).map(|i| board.pits[0][i].to_string()).collect();
    let black: Vec<String> = (0..NUM_PITS).map(|i| board.pits[1][i].to_string()).collect();
    format!(
        "{}/{}/{},{}/{},{}/{}",
        white.join(","),
        black.join(","),
        board.kazan[0], board.kazan[1],
        board.tuzdyk[0], board.tuzdyk[1],
        board.side_to_move.index(),
    )
}

/// Play one game: model (Gumbel) vs engine.
/// Only records positions where MODEL moves (not engine moves).
pub fn play_engine_game(
    ctx: &GumbelContext,
    engine: &mut EngineProcess,
    model_is_white: bool,
    engine_time_ms: u64,
    temp_threshold: u32,
) -> Vec<TrainingRecord> {
    let mut board = Board::new();
    let mut pending: Vec<PendingRecord> = Vec::with_capacity(100);
    let mut move_count = 0u32;

    engine.new_game();

    while board.game_result().is_none() && move_count < 200 {
        let is_model_turn = (board.side_to_move == Side::White) == model_is_white;

        if is_model_turn {
            // Model move: Gumbel search
            let temperature = if move_count < temp_threshold {
                1.0
            } else if move_count < temp_threshold + 15 {
                let t = (move_count - temp_threshold) as f32 / 15.0;
                1.0 - 0.7 * t
            } else {
                0.3
            };

            // Always full Gumbel for engine games (quality matters more than speed)
            let result = gumbel::gumbel_search(&board, ctx, true, temperature);

            // Record position with improved policy
            pending.push(PendingRecord {
                board,
                policy: result.improved_policy,
                side_to_move: board.side_to_move.index() as u8,
            });

            let action = result.action;
            if board.is_valid_move(action) {
                board.make_move(action);
            } else {
                let mut moves = [0usize; NUM_PITS];
                let n = board.valid_moves_array(&mut moves);
                if n > 0 { board.make_move(moves[0]); } else { break; }
            }
        } else {
            // Engine move
            if let Some(action) = engine.get_move(&board, engine_time_ms) {
                if board.is_valid_move(action) {
                    board.make_move(action);
                } else {
                    break; // Engine returned invalid move
                }
            } else {
                break; // Engine error
            }
        }

        move_count += 1;
    }

    // Game outcome: score-proportional values
    let result = board.game_result();
    let white_kazan = board.kazan[0] as f32;
    let black_kazan = board.kazan[1] as f32;
    let model_side = if model_is_white { Side::White } else { Side::Black };

    let mut records = Vec::with_capacity(pending.len());
    for p in pending {
        let value = match result {
            Some(GameResult::Win(side)) => {
                let diff = if side == Side::White {
                    (white_kazan - black_kazan) / 82.0
                } else {
                    (black_kazan - white_kazan) / 82.0
                };
                let magnitude = diff.abs().min(1.0).max(0.3);
                if side.index() as u8 == p.side_to_move {
                    magnitude
                } else {
                    -magnitude
                }
            }
            Some(GameResult::Draw) | None => 0.0,
        };

        records.push(TrainingRecord {
            board: p.board,
            policy: p.policy,
            value,
        });
    }

    records
}

struct PendingRecord {
    board: Board,
    policy: [f32; 9],
    side_to_move: u8,
}

/// Engine worker: plays multiple games against engine, sends results
pub fn engine_worker_loop(
    eval_tx: Sender<EvalRequest>,
    result_tx: Sender<Vec<TrainingRecord>>,
    engine_path: &str,
    resource_dir: &std::path::Path,
    engine_time_ms: u64,
    num_games: u32,
    worker_id: u32,
    temp_threshold: u32,
) {
    let ctx = GumbelContext::new(eval_tx);
    let mut engine = EngineProcess::spawn(engine_path, resource_dir);

    for game_id in 0..num_games {
        let model_is_white = game_id % 2 == 0; // alternate colors
        let records = play_engine_game(
            &ctx, &mut engine, model_is_white, engine_time_ms, temp_threshold,
        );

        let n = records.len();
        if result_tx.send(records).is_err() {
            break;
        }

        if (game_id + 1) % 5 == 0 || game_id + 1 == num_games {
            eprintln!(
                "Engine worker {}: {}/{} games ({} positions)",
                worker_id, game_id + 1, num_games, n,
            );
        }
    }
}

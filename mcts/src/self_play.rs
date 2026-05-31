/// Self-play worker using Gumbel MCTS (1-ply lookahead).
///
/// Playout cap randomization:
/// - 25% of moves: full Gumbel search (root + children) → improved policy for training
/// - 75% of moves: fast move (root only, raw policy) → value training only

use crate::board::{Board, GameResult, NUM_PITS};
use crate::evaluator::EvalRequest;
use crate::gumbel::{self, GumbelContext};
use crate::replay_buffer::TrainingRecord;
use crossbeam_channel::Sender;
use rand::Rng;

/// Play one complete game, returning training records.
pub fn play_one_game(
    ctx: &GumbelContext,
    temp_threshold: u32,
    _game_id: u32,
) -> Vec<TrainingRecord> {
    let mut board = Board::new();
    let mut pending: Vec<PendingRecord> = Vec::with_capacity(200);
    let mut move_number: u32 = 0;
    let max_moves: u32 = 300;
    let mut rng = rand::thread_rng();

    let full_search_prob = 0.25;

    while board.game_result().is_none() && move_number < max_moves {
        // Temperature schedule: τ=1.0 for first N moves, decay to 0.3
        let temperature = if move_number < temp_threshold {
            1.0
        } else if move_number < temp_threshold + 15 {
            let t = (move_number - temp_threshold) as f32 / 15.0;
            1.0 - 0.7 * t
        } else {
            0.3
        };

        // Playout cap: 25% full Gumbel, 75% fast (raw policy)
        let is_full = rng.gen::<f32>() < full_search_prob;

        let result = if is_full {
            gumbel::gumbel_search(&board, ctx, true, temperature)
        } else {
            gumbel::fast_move(&board, ctx, temperature)
        };

        // Record position
        pending.push(PendingRecord {
            board,
            policy: result.improved_policy, // [0,0,...] for fast moves
            side_to_move: board.side_to_move.index() as u8,
        });

        // Make move
        let action = result.action;
        if !board.is_valid_move(action) {
            let mut moves = [0usize; NUM_PITS];
            let n = board.valid_moves_array(&mut moves);
            if n > 0 {
                board.make_move(moves[0]);
            } else {
                break;
            }
        } else {
            board.make_move(action);
        }

        move_number += 1;
    }

    // Game outcome: score-proportional values
    let result = board.game_result();
    let white_kazan = board.kazan[0] as f32;
    let black_kazan = board.kazan[1] as f32;

    let mut records = Vec::with_capacity(pending.len());
    for p in pending {
        let value = match result {
            Some(GameResult::Win(side)) => {
                let diff = if side == crate::board::Side::White {
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

/// Worker loop: play multiple games
pub fn worker_loop(
    eval_tx: Sender<EvalRequest>,
    result_tx: Sender<Vec<TrainingRecord>>,
    num_games: u32,
    worker_id: u32,
    temp_threshold: u32,
) {
    let ctx = GumbelContext::new(eval_tx);
    for game_id in 0..num_games {
        let records = play_one_game(&ctx, temp_threshold, worker_id * 10000 + game_id);
        if result_tx.send(records).is_err() {
            break;
        }
        if (game_id + 1) % 10 == 0 {
            eprintln!("Worker {}: completed {}/{} games", worker_id, game_id + 1, num_games);
        }
    }
}

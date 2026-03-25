/// Self-play worker: plays complete games using MCTS, produces training records.

use crate::board::{Board, GameResult, NUM_PITS};
use crate::evaluator::EvalRequest;
use crate::mcts::{self, MctsConfig, EvalContext};
use crate::replay_buffer::TrainingRecord;
use crossbeam_channel::Sender;

/// Play one complete game, returning training records.
pub fn play_one_game(
    config: &MctsConfig,
    ctx: &EvalContext,
    _game_id: u32,
) -> Vec<TrainingRecord> {
    let mut board = Board::new();
    let mut pending: Vec<PendingRecord> = Vec::with_capacity(200);
    let mut move_number: u32 = 0;
    let max_moves: u32 = 300;

    while board.game_result().is_none() && move_number < max_moves {
        // Run MCTS search
        let add_noise = true;
        let (policy, _value) = mcts::search(&board, config, ctx, add_noise);

        // Record position
        pending.push(PendingRecord {
            board,
            policy,
            side_to_move: board.side_to_move.index() as u8,
        });

        // Select action with temperature
        let temperature = if move_number < config.temperature_threshold {
            1.0
        } else {
            0.0 // greedy
        };

        let mut moves_arr = [0usize; NUM_PITS];
        let num_moves = board.valid_moves_array(&mut moves_arr);
        let valid_moves = &moves_arr[..num_moves];

        let action = mcts::select_action(&policy, temperature, valid_moves);

        // Validate and make move
        if !board.is_valid_move(action) {
            // Fallback to first valid move
            if num_moves > 0 {
                board.make_move(valid_moves[0]);
            } else {
                break;
            }
        } else {
            board.make_move(action);
        }

        move_number += 1;
    }

    // Determine game outcome
    let result = board.game_result();

    // Convert pending records to training records with values
    let mut records = Vec::with_capacity(pending.len());
    for p in pending {
        let value = match result {
            Some(GameResult::Win(side)) => {
                if side.index() as u8 == p.side_to_move {
                    1.0
                } else {
                    -1.0
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

/// Play multiple games on this worker thread
pub fn worker_loop(
    config: MctsConfig,
    eval_tx: Sender<EvalRequest>,
    result_tx: Sender<Vec<TrainingRecord>>,
    num_games: u32,
    worker_id: u32,
) {
    let ctx = EvalContext::new(eval_tx);
    for game_id in 0..num_games {
        let records = play_one_game(&config, &ctx, worker_id * 10000 + game_id);
        let n = records.len();
        if result_tx.send(records).is_err() {
            break; // Main thread dropped receiver
        }
        if (game_id + 1) % 10 == 0 {
            eprintln!("Worker {}: completed {}/{} games", worker_id, game_id + 1, num_games);
        }
    }
}

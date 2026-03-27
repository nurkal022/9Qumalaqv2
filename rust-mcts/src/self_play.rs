/// Self-play worker: plays complete games using MCTS, produces training records.
///
/// Implements playout cap randomization (KataGo):
/// - 25% of moves: full search (num_simulations) → record for policy+value training
/// - 75% of moves: fast search (num_simulations/5) → record for value training only
///   (policy set to all zeros so trainer can distinguish)

use crate::board::{Board, GameResult, NUM_PITS};
use crate::evaluator::EvalRequest;
use crate::mcts::{self, MctsConfig, EvalContext};
use crate::replay_buffer::TrainingRecord;
use crossbeam_channel::Sender;
use rand::Rng;

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
    let mut rng = rand::thread_rng();

    // Playout cap randomization config
    let fast_sims = std::cmp::max(config.num_simulations / 5, 20);
    let full_search_prob = 0.25; // 25% full search, 75% fast

    // Create fast config (no noise for fast moves)
    let fast_config = MctsConfig {
        num_simulations: fast_sims,
        dirichlet_alpha: 0.0, // no noise for fast moves
        dirichlet_epsilon: 0.0,
        ..config.clone()
    };

    while board.game_result().is_none() && move_number < max_moves {
        // Playout cap randomization: decide full vs fast search
        let is_full_search = rng.gen::<f32>() < full_search_prob;

        let (search_config, add_noise) = if is_full_search {
            (config, true)
        } else {
            (&fast_config, false)
        };

        // Run MCTS search
        let (policy, _value) = mcts::search(&board, search_config, ctx, add_noise);

        // Record position
        // For fast search: zero out policy so trainer knows to use only for value
        let record_policy = if is_full_search {
            policy
        } else {
            [0.0f32; 9] // marker: value-only training record
        };

        pending.push(PendingRecord {
            board,
            policy: record_policy,
            side_to_move: board.side_to_move.index() as u8,
        });

        // Temperature schedule: τ=1.0 for first 25 moves, linear decay to 0.3 over next 15
        let temperature = if move_number < config.temperature_threshold {
            1.0
        } else if move_number < config.temperature_threshold + 15 {
            let t = (move_number - config.temperature_threshold) as f32 / 15.0;
            1.0 - 0.7 * t // 1.0 → 0.3
        } else {
            0.3 // soft greedy (not hard 0.0 — keeps some diversity)
        };

        let mut moves_arr = [0usize; NUM_PITS];
        let num_moves = board.valid_moves_array(&mut moves_arr);
        let valid_moves = &moves_arr[..num_moves];

        let action = mcts::select_action(&policy, temperature, valid_moves);

        // Validate and make move
        if !board.is_valid_move(action) {
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

/// Self-play worker.
///
/// Playout cap randomization (the "full search" fraction produces the policy
/// training target; the rest is fast/raw and trains value only):
/// - full search: either Gumbel 1-ply (root + children, default) OR — when
///   `full_mcts` is set — a full PUCT tree search (`mcts::search`, sims-deep,
///   the now-fixed value-sign tree). Full MCTS gives a genuinely *deeper*
///   visit-count policy target, which is what enables the AlphaZero virtuous
///   cycle (improve → better targets → improve). Gumbel-1-ply targets are
///   barely better than the raw policy, so they only sustain, not improve.
/// - fast move: root only, raw policy → value training only.

use crate::board::{Board, GameResult, NUM_PITS};
use crate::evaluator::EvalRequest;
use crate::gumbel::{self, GumbelContext};
use crate::mcts::{self, EvalContext, MctsConfig};
use crate::replay_buffer::TrainingRecord;
use crossbeam_channel::Sender;
use rand::Rng;

/// Self-play search configuration (per worker).
#[derive(Clone)]
pub struct SelfPlayConfig {
    /// Use the full PUCT tree (mcts::search) for full-search moves instead of Gumbel 1-ply.
    pub full_mcts: bool,
    /// MCTS config (sims, c_puct, Dirichlet) used when full_mcts is set.
    pub mcts: MctsConfig,
    /// Probability a move uses full search (vs a fast raw-policy move).
    pub full_search_prob: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig { full_mcts: false, mcts: MctsConfig::default(), full_search_prob: 0.25 }
    }
}

/// Play one complete game, returning training records.
pub fn play_one_game(
    gctx: &GumbelContext,
    ectx: &EvalContext,
    cfg: &SelfPlayConfig,
    temp_threshold: u32,
    _game_id: u32,
) -> Vec<TrainingRecord> {
    let mut board = Board::new();
    let mut pending: Vec<PendingRecord> = Vec::with_capacity(200);
    let mut move_number: u32 = 0;
    let max_moves: u32 = 300;
    let mut rng = rand::thread_rng();

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

        // Playout cap: `full_search_prob` of moves get a full search (policy target),
        // the rest are fast raw-policy moves (value target only, zero policy).
        let is_full = rng.gen::<f32>() < cfg.full_search_prob;

        let (policy, action) = if is_full {
            if cfg.full_mcts {
                // Full PUCT tree: deeper visit-count policy target (add_noise=true
                // for root Dirichlet exploration). Action sampled by temperature.
                let (vp, _root_val) = mcts::search(&board, &cfg.mcts, ectx, true);
                let mut moves = [0usize; NUM_PITS];
                let n = board.valid_moves_array(&mut moves);
                let action = mcts::select_action(&vp, temperature, &moves[..n]);
                (vp, action)
            } else {
                let r = gumbel::gumbel_search(&board, gctx, true, temperature);
                (r.improved_policy, r.action)
            }
        } else {
            let r = gumbel::fast_move(&board, gctx, temperature);
            (r.improved_policy, r.action) // [0,0,...] policy for fast moves
        };

        // Record position
        pending.push(PendingRecord {
            board,
            policy,
            side_to_move: board.side_to_move.index() as u8,
        });

        // Make move
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

    // Game outcome: score-proportional values.
    // Use SWEPT final totals (kazan + own remaining board stones), consistent with
    // the end-game sweep rule that game_result() now uses to decide the winner.
    // (Previously used raw kazans, which mis-scaled the magnitude on empty-side
    // terminals — the same bug class as the wrong-winner label.)
    let result = board.game_result();
    let white_kazan = board.kazan[0] as f32 + board.stones_on_side(crate::board::Side::White) as f32;
    let black_kazan = board.kazan[1] as f32 + board.stones_on_side(crate::board::Side::Black) as f32;

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
    cfg: SelfPlayConfig,
) {
    let gctx = GumbelContext::new(eval_tx.clone());
    let ectx = EvalContext::new(eval_tx);
    for game_id in 0..num_games {
        let records = play_one_game(&gctx, &ectx, &cfg, temp_threshold, worker_id * 10000 + game_id);
        if result_tx.send(records).is_err() {
            break;
        }
        if (game_id + 1) % 10 == 0 {
            eprintln!("Worker {}: completed {}/{} games", worker_id, game_id + 1, num_games);
        }
    }
}

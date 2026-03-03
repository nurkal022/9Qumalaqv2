/// Gumbel AlphaZero with Sequential Halving
///
/// Key differences from standard MCTS:
/// 1. Gumbel noise added to logits for stochastic action selection
/// 2. Top-m candidates selected (Gumbel-Top-m trick)
/// 3. Sequential Halving eliminates half the candidates each round
/// 4. Guarantees policy improvement even with small simulation budget
///
/// Reference: "Policy improvement by planning with Gumbel" (Danihelka et al., 2022)

use rand::Rng;

use crate::board::Board;
use crate::features::board_to_features;
use crate::mcts::mcts_search;
use crate::network::Network;

/// Generate Gumbel(0,1) noise: g = -ln(-ln(U)), U ~ Uniform(0,1)
fn sample_gumbel(rng: &mut impl Rng) -> f32 {
    let u: f64 = rng.gen_range(1e-10..1.0 - 1e-10);
    -((-u.ln()).ln()) as f32
}

/// Monotone transformation sigma: maps Q-values [-1,1] to logit scale.
/// sigma(q) = scale * q
fn sigma(q: f32, scale: f32) -> f32 {
    scale * q
}

/// Candidate action with Gumbel noise for Sequential Halving
struct GumbelCandidate {
    mov: usize,
    gumbel_logit: f32,   // g(a) + logits(a)
    q_estimate: f32,     // Mean Q from simulations
    simulations: u32,
    combined_score: f32, // gumbel_logit + sigma(q)
}

/// Gumbel AlphaZero search with Sequential Halving.
///
/// Returns a list of (move, score) pairs for all surviving candidates.
/// The scores can be used to construct training targets.
pub fn gumbel_alphazero_search(
    board: &Board,
    network: &Network,
    total_simulations: u32,
    initial_candidates: usize,
    sigma_scale: f32,
) -> Vec<(usize, f32)> {
    let mut rng = rand::thread_rng();

    // 1. Get policy logits from network
    let features = board_to_features(board);
    let (logits, _root_value) = network.predict(&features);

    let legal_moves = board.legal_moves();
    if legal_moves.is_empty() {
        return vec![];
    }
    if legal_moves.len() == 1 {
        return vec![(legal_moves[0], 0.0)];
    }

    // 2. For each legal move: score = gumbel(a) + logits(a)
    let mut candidates: Vec<GumbelCandidate> = legal_moves
        .iter()
        .map(|&mov| {
            let g = sample_gumbel(&mut rng);
            GumbelCandidate {
                mov,
                gumbel_logit: g + logits[mov],
                q_estimate: 0.0,
                simulations: 0,
                combined_score: 0.0,
            }
        })
        .collect();

    // 3. Gumbel-Top-m: keep m best candidates
    candidates.sort_by(|a, b| {
        b.gumbel_logit
            .partial_cmp(&a.gumbel_logit)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let m = initial_candidates.min(candidates.len());
    candidates.truncate(m);

    // 4. Sequential Halving
    let mut sims_used: u32 = 0;

    while candidates.len() > 1 && sims_used < total_simulations {
        let rounds_left = (candidates.len() as f32).log2().ceil().max(1.0) as u32;
        let budget_per_action =
            ((total_simulations - sims_used) / rounds_left) / candidates.len() as u32;
        let budget_per_action = budget_per_action.max(1);

        // Give each candidate budget_per_action simulations
        for candidate in candidates.iter_mut() {
            let mut child_board = board.clone();
            child_board.make_move(candidate.mov);

            if child_board.is_terminal() {
                candidate.q_estimate =
                    -child_board.outcome_for_side(child_board.side_to_move_u8());
                candidate.simulations += 1;
                sims_used += 1;
            } else {
                // Run mini-MCTS from the position after the move
                let (_move_visits, avg_value) =
                    mcts_search(&child_board, network, budget_per_action);
                let new_q = -avg_value;

                // Update running average
                let total_sims = candidate.simulations + budget_per_action;
                candidate.q_estimate = (candidate.q_estimate * candidate.simulations as f32
                    + new_q * budget_per_action as f32)
                    / total_sims as f32;
                candidate.simulations = total_sims;
                sims_used += budget_per_action;
            }
        }

        // Combined score: gumbel_logit + sigma(q)
        for c in candidates.iter_mut() {
            c.combined_score = c.gumbel_logit + sigma(c.q_estimate, sigma_scale);
        }

        // Eliminate worst half
        candidates.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let keep = (candidates.len() + 1) / 2;
        candidates.truncate(keep);
    }

    candidates
        .iter()
        .map(|c| (c.mov, c.combined_score))
        .collect()
}

/// Simplified Gumbel search using random network (for testing)
pub fn gumbel_search_random(
    board: &Board,
    total_simulations: u32,
    initial_candidates: usize,
) -> Vec<(usize, f32)> {
    let mut rng = rand::thread_rng();
    let legal_moves = board.legal_moves();

    if legal_moves.is_empty() {
        return vec![];
    }
    if legal_moves.len() == 1 {
        return vec![(legal_moves[0], 0.0)];
    }

    // Uniform logits + Gumbel noise
    let mut candidates: Vec<GumbelCandidate> = legal_moves
        .iter()
        .map(|&mov| {
            let g = sample_gumbel(&mut rng);
            GumbelCandidate {
                mov,
                gumbel_logit: g,
                q_estimate: 0.0,
                simulations: 0,
                combined_score: g,
            }
        })
        .collect();

    candidates.sort_by(|a, b| {
        b.gumbel_logit
            .partial_cmp(&a.gumbel_logit)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let m = initial_candidates.min(candidates.len());
    candidates.truncate(m);

    // Sequential Halving with random rollouts
    let mut sims_used: u32 = 0;

    while candidates.len() > 1 && sims_used < total_simulations {
        let rounds_left = (candidates.len() as f32).log2().ceil().max(1.0) as u32;
        let budget_per_action =
            ((total_simulations - sims_used) / rounds_left) / candidates.len() as u32;
        let budget_per_action = budget_per_action.max(1);

        for candidate in candidates.iter_mut() {
            let mut child_board = board.clone();
            child_board.make_move(candidate.mov);

            if child_board.is_terminal() {
                candidate.q_estimate =
                    -child_board.outcome_for_side(child_board.side_to_move_u8());
                candidate.simulations += 1;
                sims_used += 1;
            } else {
                let (_visits, avg_value) =
                    crate::mcts::mcts_search_random(&child_board, budget_per_action);
                let new_q = -avg_value;
                let total_sims = candidate.simulations + budget_per_action;
                candidate.q_estimate = (candidate.q_estimate * candidate.simulations as f32
                    + new_q * budget_per_action as f32)
                    / total_sims as f32;
                candidate.simulations = total_sims;
                sims_used += budget_per_action;
            }
        }

        for c in candidates.iter_mut() {
            c.combined_score = c.gumbel_logit + sigma(c.q_estimate, 50.0);
        }

        candidates.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let keep = (candidates.len() + 1) / 2;
        candidates.truncate(keep);
    }

    candidates
        .iter()
        .map(|c| (c.mov, c.combined_score))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gumbel_noise_distribution() {
        // Gumbel noise should have mean ~0.577 (Euler-Mascheroni constant)
        let mut rng = rand::thread_rng();
        let n = 10000;
        let sum: f32 = (0..n).map(|_| sample_gumbel(&mut rng)).sum();
        let mean = sum / n as f32;
        assert!(
            (mean - 0.577).abs() < 0.1,
            "Gumbel mean = {}, expected ~0.577",
            mean
        );
    }

    #[test]
    fn test_gumbel_search_random_returns_valid() {
        let board = Board::new();
        let results = gumbel_search_random(&board, 32, 4);
        assert!(!results.is_empty());

        let legal = board.legal_moves();
        for (mov, _score) in &results {
            assert!(legal.contains(mov));
        }
    }

    #[test]
    fn test_gumbel_search_single_move() {
        // When only one move is legal, return it immediately
        let mut board = Board::new();
        board.pits[0] = [0, 0, 0, 0, 5, 0, 0, 0, 0];
        let results = gumbel_search_random(&board, 32, 4);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 4);
    }

    #[test]
    fn test_sigma_monotone() {
        let scale = 50.0;
        assert!(sigma(0.5, scale) > sigma(0.0, scale));
        assert!(sigma(0.0, scale) > sigma(-0.5, scale));
        assert!(sigma(1.0, scale) > sigma(-1.0, scale));
    }
}

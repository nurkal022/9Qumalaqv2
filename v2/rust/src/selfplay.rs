/// Self-play game generation for training data.
///
/// Plays complete games using Gumbel AlphaZero search, recording
/// (features, policy, value) tuples for neural network training.

use std::sync::{Arc, Mutex};
use std::thread;

use rand::Rng;

use crate::board::Board;
use crate::data_writer::{DataWriter, TrainingSample};
use crate::features::board_to_features;
use crate::gumbel::gumbel_alphazero_search;
use crate::network::{Network, NUM_ACTIONS};

/// Play one self-play game and return training samples.
pub fn play_one_game(
    network: &Network,
    simulations: u32,
    candidates: usize,
    sigma_scale: f32,
    temperature_moves: usize,
) -> Vec<TrainingSample> {
    let mut board = Board::new();
    let mut history: Vec<(Board, [f32; NUM_ACTIONS])> = Vec::new();
    let mut rng = rand::thread_rng();
    let mut move_count = 0;

    while !board.is_terminal() && move_count < 300 {
        // Gumbel AlphaZero search
        let results =
            gumbel_alphazero_search(&board, network, simulations, candidates, sigma_scale);

        if results.is_empty() {
            break;
        }

        // Normalize scores into a policy distribution
        let mut policy = [0.0f32; NUM_ACTIONS];
        let max_score = results
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = results.iter().map(|(_, s)| (s - max_score).exp()).sum();
        for (mov, score) in &results {
            policy[*mov] = (score - max_score).exp() / sum;
        }

        // Save position + policy for training
        history.push((board.clone(), policy));

        // Select move
        let selected_move = if move_count < temperature_moves {
            // Temperature 1.0: proportional selection for opening diversity
            select_proportional(&policy, &mut rng)
        } else {
            // Greedy: pick highest score
            results
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0
        };

        board.make_move(selected_move);
        move_count += 1;
    }

    // Determine game outcome
    let outcome = board.outcome(); // +1 White, -1 Black, 0 draw

    // Build training samples with correct value signs
    history
        .iter()
        .map(|(pos, policy)| {
            let side = pos.side_to_move_u8();
            let value = if side == 0 { outcome } else { -outcome };
            TrainingSample {
                features: board_to_features(pos),
                policy: *policy,
                value,
            }
        })
        .collect()
}

/// Generate multiple self-play games across threads.
pub fn generate_games(
    network: Arc<Network>,
    num_games: u32,
    simulations: u32,
    candidates: usize,
    sigma_scale: f32,
    num_threads: usize,
    temperature_moves: usize,
    output_path: &str,
) {
    let writer = Arc::new(Mutex::new(DataWriter::new(output_path)));
    let games_per_thread = num_games / num_threads as u32;
    let total_samples = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let total_games_done = Arc::new(std::sync::atomic::AtomicU32::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_thread_id| {
            let net = Arc::clone(&network);
            let writer = Arc::clone(&writer);
            let samples_counter = Arc::clone(&total_samples);
            let games_counter = Arc::clone(&total_games_done);

            thread::spawn(move || {
                for _game_idx in 0..games_per_thread {
                    let samples = play_one_game(
                        &net,
                        simulations,
                        candidates,
                        sigma_scale,
                        temperature_moves,
                    );

                    let num_samples = samples.len();

                    {
                        let mut w = writer.lock().unwrap();
                        for sample in samples {
                            w.write_sample(&sample);
                        }
                    }

                    samples_counter
                        .fetch_add(num_samples as u64, std::sync::atomic::Ordering::Relaxed);
                    let done =
                        games_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

                    if done % 100 == 0 || done == num_games {
                        let total_s =
                            samples_counter.load(std::sync::atomic::Ordering::Relaxed);
                        eprintln!(
                            "  Games: {}/{} | Samples: {} | Avg moves/game: {:.1}",
                            done,
                            num_games,
                            total_s,
                            total_s as f64 / done as f64
                        );
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let mut w = writer.lock().unwrap();
    let count = w.finish();
    eprintln!("Self-play complete. Total samples: {}", count);
}

/// Select a move proportionally to the policy distribution
fn select_proportional(policy: &[f32; NUM_ACTIONS], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in policy.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return i;
        }
    }
    // Fallback: pick the highest probability move
    policy
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_proportional() {
        let mut rng = rand::thread_rng();
        let policy = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        // With probability 1.0 at index 4, should always select 4
        for _ in 0..100 {
            assert_eq!(select_proportional(&policy, &mut rng), 4);
        }
    }

    #[test]
    fn test_select_proportional_distribution() {
        let mut rng = rand::thread_rng();
        let policy = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut counts = [0u32; NUM_ACTIONS];
        let n = 10000;
        for _ in 0..n {
            let m = select_proportional(&policy, &mut rng);
            counts[m] += 1;
        }
        // Both moves 0 and 1 should get ~50%
        let ratio = counts[0] as f64 / n as f64;
        assert!(ratio > 0.45 && ratio < 0.55, "ratio = {}", ratio);
    }
}

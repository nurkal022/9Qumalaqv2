/// Texel tuning for evaluation weights
///
/// Method: optimize eval weights to minimize mean squared error
/// between sigmoid(eval) and actual game results.
///
/// Reference: https://www.chessprogramming.org/Texel%27s_Tuning_Method

use std::fs;
use crate::board::{Board, Side, NUM_PITS};

/// Number of tunable weights
const NUM_WEIGHTS: usize = 11;

/// Weight names for display
const WEIGHT_NAMES: [&str; NUM_WEIGHTS] = [
    "material",
    "tuzdyk_base",
    "tuzdyk_center",
    "threat",
    "threat_opp_discount",
    "pit_stones",
    "mobility",
    "empty_pit",
    "large_pit",
    "endgame_material",
    "capture_opp",
];

/// A tuning position: board + expected result (0.0 to 1.0 from side-to-move)
struct TunePosition {
    board: Board,
    result: f64, // 1.0 = side-to-move wins, 0.0 = loses, 0.5 = draw
}

/// Parameterized evaluation (mirrors eval.rs but with tunable weights)
fn evaluate_with_weights(board: &Board, w: &[f64; NUM_WEIGHTS]) -> f64 {
    let me = board.side_to_move.index();
    let opp = board.side_to_move.opposite().index();

    // Terminal check
    if let Some(result) = board.game_result() {
        return match result {
            crate::board::GameResult::Win(winner) => {
                if winner == board.side_to_move { 90000.0 } else { -90000.0 }
            }
            crate::board::GameResult::Draw => 0.0,
        };
    }

    let mut score: f64 = 0.0;

    // 1. Material (kazan difference)
    let material_diff = board.kazan[me] as f64 - board.kazan[opp] as f64;
    score += material_diff * w[0]; // material

    // Endgame boost
    let total_kazan = board.kazan[0] as f64 + board.kazan[1] as f64;
    if total_kazan > 100.0 {
        score += material_diff * w[9]; // endgame_material
    }

    // Proximity to win
    if board.kazan[me] >= 70 {
        score += (board.kazan[me] as f64 - 70.0) * 30.0;
    }
    if board.kazan[opp] >= 70 {
        score -= (board.kazan[opp] as f64 - 70.0) * 30.0;
    }

    // 2. Tuzdyk
    if board.tuzdyk[me] >= 0 {
        let pos = board.tuzdyk[me] as usize;
        score += w[1]; // tuzdyk_base
        let center = match pos {
            3 | 4 | 5 => 3.0,
            2 | 6 => 2.0,
            1 | 7 => 1.0,
            _ => 0.0,
        };
        score += center * w[2]; // tuzdyk_center
    }
    if board.tuzdyk[opp] >= 0 {
        let pos = board.tuzdyk[opp] as usize;
        score -= w[1];
        let center = match pos {
            3 | 4 | 5 => 3.0,
            2 | 6 => 2.0,
            1 | 7 => 1.0,
            _ => 0.0,
        };
        score -= center * w[2];
    }

    // 3. Tuzdyk threats
    if board.tuzdyk[me] == -1 {
        for i in 0..8 {
            if board.pits[opp][i] == 2 && board.tuzdyk[opp] != i as i8 {
                score += w[3]; // threat
            }
        }
    }
    if board.tuzdyk[opp] == -1 {
        for i in 0..8 {
            if board.pits[me][i] == 2 && board.tuzdyk[me] != i as i8 {
                score -= w[3] * w[4] / 100.0; // threat * discount
            }
        }
    }

    // 4. Pit stones
    let my_stones: f64 = board.pits[me].iter().map(|&x| x as f64).sum();
    let opp_stones: f64 = board.pits[opp].iter().map(|&x| x as f64).sum();
    score += (my_stones - opp_stones) * w[5]; // pit_stones

    // 5. Mobility
    let opp_tuzdyk = board.tuzdyk[opp];
    let me_tuzdyk = board.tuzdyk[me];
    let mut my_moves = 0.0f64;
    let mut opp_moves = 0.0f64;
    for i in 0..NUM_PITS {
        if board.pits[me][i] > 0 && opp_tuzdyk != i as i8 {
            my_moves += 1.0;
        }
        if board.pits[opp][i] > 0 && me_tuzdyk != i as i8 {
            opp_moves += 1.0;
        }
    }
    score += (my_moves - opp_moves) * w[6]; // mobility

    // 6. Empty pit penalty
    for i in 0..NUM_PITS {
        if board.pits[me][i] == 0 && opp_tuzdyk != i as i8 {
            score += w[7]; // empty_pit (should be negative)
        }
    }

    // 7. Large pit bonus
    for i in 0..NUM_PITS {
        if board.pits[me][i] >= 10 {
            score += (board.pits[me][i] as f64 - 9.0) * w[8]; // large_pit
        }
    }

    // 8. Capture opportunities
    for i in 0..NUM_PITS {
        if board.pits[me][i] > 0 && opp_tuzdyk != i as i8 {
            if let Some((side, pit)) = predict_landing(i, board.pits[me][i], me) {
                if side == opp {
                    let new_count = board.pits[opp][pit] + 1;
                    if new_count % 2 == 0 && new_count > 0 {
                        score += new_count as f64 * w[10]; // capture_opp
                    }
                }
            }
        }
    }
    for i in 0..NUM_PITS {
        if board.pits[opp][i] > 0 && me_tuzdyk != i as i8 {
            if let Some((side, pit)) = predict_landing(i, board.pits[opp][i], opp) {
                if side == me {
                    let new_count = board.pits[me][pit] + 1;
                    if new_count % 2 == 0 && new_count > 0 {
                        score -= new_count as f64 * w[10];
                    }
                }
            }
        }
    }

    score
}

fn predict_landing(pit: usize, stones: u8, side: usize) -> Option<(usize, usize)> {
    if stones == 0 {
        return None;
    }
    if stones == 1 {
        let next = pit + 1;
        if next > 8 {
            return Some((1 - side, 0));
        }
        return Some((side, next));
    }
    let remaining = stones as usize - 1;
    let mut pos = pit + remaining;
    let mut landing_side = side;
    while pos > 8 {
        pos -= 9;
        landing_side = 1 - landing_side;
    }
    Some((landing_side, pos))
}

/// Sigmoid: maps eval to win probability
/// S(e) = 1 / (1 + 10^(-e/K))
fn sigmoid(eval: f64, k: f64) -> f64 {
    1.0 / (1.0 + 10.0_f64.powf(-eval / k))
}

/// Mean squared error over all positions
fn compute_error(positions: &[TunePosition], weights: &[f64; NUM_WEIGHTS], k: f64) -> f64 {
    let mut total_error = 0.0;
    for pos in positions {
        let eval = evaluate_with_weights(&pos.board, weights);
        let predicted = sigmoid(eval, k);
        let diff = pos.result - predicted;
        total_error += diff * diff;
    }
    total_error / positions.len() as f64
}

/// Find optimal K value (scaling factor for sigmoid)
fn find_optimal_k(positions: &[TunePosition], weights: &[f64; NUM_WEIGHTS]) -> f64 {
    let mut best_k = 400.0;
    let mut best_error = f64::MAX;

    // Coarse search
    for k_int in (50..=1000).step_by(50) {
        let k = k_int as f64;
        let error = compute_error(positions, weights, k);
        if error < best_error {
            best_error = error;
            best_k = k;
        }
    }

    // Fine search around best
    let start = (best_k - 50.0).max(10.0);
    let end = best_k + 50.0;
    let mut k = start;
    while k <= end {
        let error = compute_error(positions, weights, k);
        if error < best_error {
            best_error = error;
            best_k = k;
        }
        k += 5.0;
    }

    eprintln!("Optimal K = {:.1} (error = {:.6})", best_k, best_error);
    best_k
}

/// Load positions from file
fn load_positions(path: &str) -> Vec<TunePosition> {
    let content = fs::read_to_string(path).expect("Failed to read positions file");
    let mut positions = Vec::new();

    for line in content.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 24 {
            continue;
        }

        let mut pits = [[0u8; NUM_PITS]; 2];
        for i in 0..9 {
            pits[0][i] = parts[i].parse().unwrap_or(0);
        }
        for i in 0..9 {
            pits[1][i] = parts[9 + i].parse().unwrap_or(0);
        }

        let kazan_w: u8 = parts[18].parse().unwrap_or(0);
        let kazan_b: u8 = parts[19].parse().unwrap_or(0);
        let tuzdyk_w: i8 = parts[20].parse().unwrap_or(-1);
        let tuzdyk_b: i8 = parts[21].parse().unwrap_or(-1);
        let side: u8 = parts[22].parse().unwrap_or(0);
        let result: f64 = parts[23].parse().unwrap_or(0.5);

        let board = Board {
            pits,
            kazan: [kazan_w, kazan_b],
            tuzdyk: [tuzdyk_w, tuzdyk_b],
            side_to_move: if side == 0 { Side::White } else { Side::Black },
            move_count: 0,
        };

        positions.push(TunePosition { board, result });
    }

    positions
}

/// Run Texel tuning
pub fn run_texel_tuning() {
    println!("Texel Tuning for Тоғызқұмалақ Evaluation");
    println!("=========================================\n");

    // Load positions
    let positions = load_positions("positions.txt");
    println!("Loaded {} positions\n", positions.len());

    if positions.is_empty() {
        println!("No positions found! Run export_positions.py first.");
        return;
    }

    // Initial weights (current eval.rs values)
    // [material, tuzdyk_base, tuzdyk_center, threat, threat_opp_discount,
    //  pit_stones, mobility, empty_pit, large_pit, endgame_material, capture_opp]
    let mut weights: [f64; NUM_WEIGHTS] = [
        100.0,  // material
        800.0,  // tuzdyk_base
        50.0,   // tuzdyk_center
        150.0,  // threat
        50.0,   // threat_opp_discount (as percentage, so 50 = half)
        10.0,   // pit_stones
        20.0,   // mobility
        -15.0,  // empty_pit
        5.0,    // large_pit
        50.0,   // endgame_material
        8.0,    // capture_opp
    ];

    // Find optimal K
    let k = find_optimal_k(&positions, &weights);

    let initial_error = compute_error(&positions, &weights, k);
    println!("Initial error: {:.6}", initial_error);
    println!("Initial weights:");
    for i in 0..NUM_WEIGHTS {
        println!("  {:>22}: {:>8.1}", WEIGHT_NAMES[i], weights[i]);
    }
    println!();

    // Local search optimization
    // Try adjusting each weight by delta, keep if error improves
    let mut best_error = initial_error;
    let mut improved = true;
    let mut iteration = 0;

    // Start with large steps, decrease
    let deltas = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0];

    for &delta in &deltas {
        improved = true;
        while improved {
            improved = false;
            iteration += 1;

            for i in 0..NUM_WEIGHTS {
                // Try +delta
                weights[i] += delta;
                let error_plus = compute_error(&positions, &weights, k);

                if error_plus < best_error {
                    best_error = error_plus;
                    improved = true;
                    continue; // keep the change
                }

                // Try -delta (from +delta position, so -2*delta from original)
                weights[i] -= 2.0 * delta;
                let error_minus = compute_error(&positions, &weights, k);

                if error_minus < best_error {
                    best_error = error_minus;
                    improved = true;
                    continue;
                }

                // Neither improved, revert
                weights[i] += delta;
            }

            if iteration % 10 == 0 {
                eprintln!(
                    "Iteration {}, delta={:.0}, error={:.6}",
                    iteration, delta, best_error
                );
            }
        }

        println!(
            "Delta {:.0} done after {} iterations, error={:.6}",
            delta, iteration, best_error
        );
    }

    println!("\n=========================================");
    println!("Tuning complete!");
    println!("=========================================\n");
    println!("Error: {:.6} -> {:.6} ({:.1}% improvement)",
        initial_error, best_error,
        (1.0 - best_error / initial_error) * 100.0
    );
    println!("\nOptimal weights (for eval.rs):\n");

    // Print as Rust constants
    let rust_names = [
        "MATERIAL_WEIGHT",
        "TUZDYK_BASE",
        "TUZDYK_CENTER_BONUS",
        "THREAT_WEIGHT",
        "THREAT_OPP_DISCOUNT_PCT",
        "PIT_STONES_WEIGHT",
        "MOBILITY_WEIGHT",
        "EMPTY_PIT_PENALTY",
        "LARGE_PIT_BONUS",
        "ENDGAME_MATERIAL_BOOST",
        "CAPTURE_OPP_WEIGHT",
    ];

    for i in 0..NUM_WEIGHTS {
        let old = match i {
            0 => 100.0, 1 => 800.0, 2 => 50.0, 3 => 150.0, 4 => 50.0,
            5 => 10.0, 6 => 20.0, 7 => -15.0, 8 => 5.0, 9 => 50.0, 10 => 8.0,
            _ => 0.0,
        };
        let change = if old != 0.0 {
            format!("{:+.0}%", (weights[i] / old - 1.0) * 100.0)
        } else {
            "new".to_string()
        };
        println!(
            "const {}: i32 = {};  // was {:.0} ({})",
            rust_names[i],
            weights[i].round() as i32,
            old,
            change
        );
    }
}

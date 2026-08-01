/// Handcrafted evaluation function for Тоғызқұмалақ
///
/// Returns score in centipawns from the side-to-move's perspective.
/// Positive = good for side to move, negative = bad.

use crate::board::{Board, NUM_PITS};

/// Evaluation weights — Texel-tuned from 77K PlayOK positions (44% error reduction)
/// `pub(crate)`: this is also the legacy eval's per-stone (kazan-difference) centipawn
/// weight, referenced by `nnue.rs`'s NNU2 v4 margin blend (`STONE_CP`) so the two stay
/// in lockstep instead of duplicating the magic number.
pub(crate) const MATERIAL_WEIGHT: i32 = 21;
/// Position-specific tuzdyk values from PlayOK 533K game winrate analysis
/// Pit7=62.2%, Pit5=60.1%, Pit6=58.5%, Pit4=55.2%, Pit2=53.1%, Pit3=52.0%, Pit8=51.5%, Pit1=50.3%
const TUZDYK_VALUE: [i32; 9] = [350, 420, 400, 450, 550, 530, 560, 380, 0];
const THREAT_WEIGHT: i32 = 11;        // tuzdyk creation threat
const PIT_STONES_WEIGHT: i32 = 3;
const MOBILITY_WEIGHT: i32 = 124;     // most important positional factor
const EMPTY_PIT_PENALTY: i32 = -5;
const LARGE_PIT_BONUS: i32 = 1;
const ENDGAME_MATERIAL_BOOST: i32 = -6;
const CAPTURE_OPP_WEIGHT: i32 = 4;
const STARVATION_WEIGHT: i32 = 15;      // quadratic bonus for starving opponent
const STARVATION_FINISH: i32 = 200;     // extra bonus to finish off nearly-empty opponent

/// Maximum possible eval (for mate scores)
pub const EVAL_INF: i32 = 100_000;
pub const EVAL_MATE: i32 = 90_000;

/// Evaluate a position from side-to-move's perspective
pub fn evaluate(board: &Board) -> i32 {
    let me = board.side_to_move.index();
    let opp = board.side_to_move.opposite().index();

    // Terminal check
    if let Some(result) = board.game_result() {
        return match result {
            crate::board::GameResult::Win(winner) => {
                if winner == board.side_to_move {
                    EVAL_MATE - board.move_count as i32 // prefer faster wins
                } else {
                    -EVAL_MATE + board.move_count as i32 // prefer slower losses
                }
            }
            crate::board::GameResult::Draw => 0,
        };
    }

    let mut score: i32 = 0;

    // 1. Material (kazan difference) — most important
    let material_diff = board.kazan[me] as i32 - board.kazan[opp] as i32;
    score += material_diff * MATERIAL_WEIGHT;

    // Endgame: boost material importance when close to winning
    let total_kazan = board.kazan[0] as i32 + board.kazan[1] as i32;
    if total_kazan > 100 {
        score += material_diff * ENDGAME_MATERIAL_BOOST;
    }

    // Proximity to win bonus
    if board.kazan[me] >= 70 {
        score += (board.kazan[me] as i32 - 70) * 30;
    }
    if board.kazan[opp] >= 70 {
        score -= (board.kazan[opp] as i32 - 70) * 30;
    }

    // 2. Tuzdyk evaluation — position-specific values from 533K game analysis
    if board.tuzdyk[me] >= 0 {
        score += TUZDYK_VALUE[board.tuzdyk[me] as usize];
    }
    if board.tuzdyk[opp] >= 0 {
        score -= TUZDYK_VALUE[board.tuzdyk[opp] as usize];
    }

    // 3. Tuzdyk threats (opponent pits with 2 stones = one move from 3 = tuzdyk opportunity)
    if board.tuzdyk[me] == -1 {
        for i in 0..8 {
            if board.pits[opp][i] == 2 && board.tuzdyk[opp] != i as i8 {
                score += THREAT_WEIGHT;
            }
        }
    }

    // 4. Stones in own pits (potential material)
    let my_pit_stones: i32 = board.pits[me].iter().map(|&x| x as i32).sum();
    let opp_pit_stones: i32 = board.pits[opp].iter().map(|&x| x as i32).sum();
    score += (my_pit_stones - opp_pit_stones) * PIT_STONES_WEIGHT;

    // 5. Mobility — critical in late game (tempo play)
    let mut my_moves = 0i32;
    let mut opp_moves = 0i32;
    let opp_tuzdyk = board.tuzdyk[opp];
    let me_tuzdyk = board.tuzdyk[me];

    for i in 0..NUM_PITS {
        if board.pits[me][i] > 0 && opp_tuzdyk != i as i8 {
            my_moves += 1;
        }
        if board.pits[opp][i] > 0 && me_tuzdyk != i as i8 {
            opp_moves += 1;
        }
    }

    // Mobility weight scales with stones-on-board (data-tuned from 360K PlayOK games):
    //   ≤ 60 stones: ×2 — entering tempo phase (median ply 47, 92% of pro games reach this)
    //   ≤ 40 stones: ×3 — main tempo battle (61% of pro games, last-capture median: 37 stones)
    //   ≤ 25 stones: ×5 — critical zugzwang risk (rare but decisive)
    // Note: ply count not used; stones-on-board is what actually matters.
    let total_board = (my_pit_stones + opp_pit_stones) as i32;
    let mobility_mult = if total_board <= 25 {
        5
    } else if total_board <= 40 {
        3
    } else if total_board <= 60 {
        2
    } else {
        1
    };
    score += (my_moves - opp_moves) * MOBILITY_WEIGHT * mobility_mult;

    // Late-phase pit-asymmetry bonus (from 360K analysis):
    // 21% of pro games end with kazan spread ≤5 — decided purely by who emptied first.
    // Stones on YOUR pits at game end almost always become yours (sweep rule).
    // So in tempo phase, a stone on my side is worth more than a stone on opp's side.
    if total_board <= 60 {
        let pit_asymmetry = my_pit_stones as i32 - opp_pit_stones as i32;
        let asym_weight = if total_board <= 25 { 12 }
                          else if total_board <= 40 { 8 }
                          else { 4 };
        score += pit_asymmetry * asym_weight;
    }

    // CRITICAL: heavy penalty if I have very few moves vs opponent
    // This prevents engine from "winning material but losing on moves"
    if my_moves <= 3 && opp_moves > my_moves {
        let deficit = opp_moves - my_moves;
        // Quadratic penalty when low on moves
        score -= deficit * deficit * 80;
    }
    if opp_moves <= 3 && my_moves > opp_moves {
        let advantage = my_moves - opp_moves;
        score += advantage * advantage * 80;
    }

    // EXTREMELY CRITICAL: I'm about to run out of moves
    // When opponent forces me to empty my last pits while keeping their reserves
    if my_moves <= 1 && total_board > 5 {
        score -= 600;  // about to lose by zugzwang
    }
    if opp_moves <= 1 && total_board > 5 {
        score += 600;  // opponent about to lose by zugzwang
    }

    // 6. Empty pit penalty
    for i in 0..NUM_PITS {
        if board.pits[me][i] == 0 && opp_tuzdyk != i as i8 {
            score += EMPTY_PIT_PENALTY;
        }
    }

    // 7. Large pit bonus (stones concentrated = capture potential)
    for i in 0..NUM_PITS {
        if board.pits[me][i] >= 10 {
            score += (board.pits[me][i] as i32 - 9) * LARGE_PIT_BONUS;
        }
    }

    // 8. Capture opportunities
    for i in 0..NUM_PITS {
        if board.pits[me][i] > 0 && opp_tuzdyk != i as i8 {
            if let Some((side, pit)) = predict_landing(i, board.pits[me][i], me) {
                if side == opp {
                    let new_count = board.pits[opp][pit] + 1;
                    if new_count % 2 == 0 && new_count > 0 {
                        score += new_count as i32 * CAPTURE_OPP_WEIGHT;
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
                        score -= new_count as i32 * CAPTURE_OPP_WEIGHT;
                    }
                }
            }
        }
    }

    // 9. Starvation pressure — keeping opponent's side empty wins by zugzwang.
    //    Two-stage: gradual linear ramp (kicks in at 18 stones), then quadratic explosion (≤9).
    //    Data: 21% of pro games end with kazan spread ≤5 → decided by who emptied first.
    //    Median end-of-material has 37 total stones on board, so ~18-19 per side is "warning zone".
    let opp_p = opp_pit_stones as i32;
    let my_p = my_pit_stones as i32;
    if opp_p <= 18 {
        // Linear pre-warning gradient
        score += (18 - opp_p) * 6;
    }
    if my_p <= 18 {
        score -= (18 - my_p) * 6;
    }
    if opp_p <= 9 {
        // Quadratic explosion — close to terminal
        let pressure = 10 - opp_p;
        score += pressure * pressure * STARVATION_WEIGHT;
    }
    if my_p <= 9 {
        let pressure = 10 - my_p;
        score -= pressure * pressure * STARVATION_WEIGHT;
    }

    // 10. Finishing bonus — when we're winning on material and opponent is nearly empty,
    //     give a huge bonus to push for the kill instead of giving stones back
    if material_diff > 5 && opp_pit_stones <= 3 {
        score += (4 - opp_pit_stones) * STARVATION_FINISH;
    }
    if material_diff < -5 && my_pit_stones <= 3 {
        score -= (4 - my_pit_stones) * STARVATION_FINISH;
    }

    // 11. Endgame right-pit bonus (pits 7-9 dominate endgame: 50% of expert moves)
    let total_stones = board.total_board_stones();
    if total_stones <= 40 {
        let my_right: i32 = board.pits[me][6..9].iter().map(|&x| x as i32).sum();
        let opp_right: i32 = board.pits[opp][6..9].iter().map(|&x| x as i32).sum();
        score += (my_right - opp_right) * 3;
    }

    // 12. Midgame positional eval (stones 50-100: heavy pits, scatter penalty, right-pit)
    //     Switched from ply count to stones-on-board: data shows ply 30 ≈ 80 stones,
    //     ply 60 ≈ 52 stones (median). Stone-based threshold is more robust.
    //     Pit-asymmetry is already handled in section 5 (≤60), so not duplicated here.
    if total_board >= 50 && total_board <= 100 {
        // Heavy pit bonus: pits with 10+ stones are tactical weapons
        for i in 0..NUM_PITS {
            if board.pits[me][i] >= 10 {
                score += 5;
            }
            if board.pits[opp][i] >= 10 {
                score -= 5;
            }
        }

        // Scatter penalty: many pits with 1-2 stones = weak, fragmented position
        let mut my_scattered = 0i32;
        let mut opp_scattered = 0i32;
        for i in 0..NUM_PITS {
            if board.pits[me][i] >= 1 && board.pits[me][i] <= 2 {
                my_scattered += 1;
            }
            if board.pits[opp][i] >= 1 && board.pits[opp][i] <= 2 {
                opp_scattered += 1;
            }
        }
        if my_scattered >= 5 {
            score -= (my_scattered - 4) * 10;
        }
        if opp_scattered >= 5 {
            score += (opp_scattered - 4) * 10;
        }

        // Right-pit bonus in midgame (lower weight than endgame)
        let my_right: i32 = board.pits[me][6..9].iter().map(|&x| x as i32).sum();
        let opp_right: i32 = board.pits[opp][6..9].iter().map(|&x| x as i32).sum();
        score += (my_right - opp_right) * 2;
    }

    score
}

/// Predict where the last stone lands (approximate — doesn't account for tuzdyk skipping)
fn predict_landing(pit: usize, stones: u8, side: usize) -> Option<(usize, usize)> {
    if stones == 0 {
        return None;
    }

    if stones == 1 {
        let next_pit = pit + 1;
        if next_pit > 8 {
            return Some((1 - side, 0));
        }
        return Some((side, next_pit));
    }

    // For stones > 1: first stone stays, remaining distributed
    let remaining = stones as usize - 1;
    let mut pos = pit + remaining;
    let mut landing_side = side;

    // Simple calculation (ignoring tuzdyk skipping)
    if pos > 8 {
        pos -= 9;
        landing_side = 1 - landing_side;
        if pos > 8 {
            pos -= 9;
            landing_side = 1 - landing_side;
            // Could wrap more but rare
            while pos > 8 {
                pos -= 9;
                landing_side = 1 - landing_side;
            }
        }
    }

    Some((landing_side, pos))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_eval_near_zero() {
        let b = Board::new();
        let score = evaluate(&b);
        // Initial position should be roughly equal
        assert!(score.abs() < 100, "Initial eval too far from 0: {}", score);
    }

    #[test]
    fn test_material_advantage() {
        let mut b = Board::new();
        b.kazan[0] = 50;
        b.kazan[1] = 10;
        let score = evaluate(&b);
        assert!(score > 0, "White should have positive eval with more material");
    }

    #[test]
    fn test_winning_position() {
        let mut b = Board::new();
        b.kazan[0] = 82;
        let score = evaluate(&b);
        assert!(score > EVAL_MATE / 2, "Winning position should have very high eval");
    }
}

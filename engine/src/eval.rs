/// Handcrafted evaluation function for Тоғызқұмалақ
///
/// Returns score in centipawns from the side-to-move's perspective.
/// Positive = good for side to move, negative = bad.

use crate::board::{Board, NUM_PITS};

/// Evaluation weights — Texel-tuned from 77K PlayOK positions (44% error reduction)
const MATERIAL_WEIGHT: i32 = 21;
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

    // 5. Mobility
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
    score += (my_moves - opp_moves) * MOBILITY_WEIGHT;

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

    // 9. Starvation pressure — keeping opponent's side empty is critical
    //    When opponent has few stones, each stone less is exponentially more valuable
    //    because at 0 stones the game ends (we win if ahead in kazan)
    if opp_pit_stones <= 9 {
        let pressure = 10 - opp_pit_stones;
        score += pressure * pressure * STARVATION_WEIGHT;
    }
    if my_pit_stones <= 9 {
        let pressure = 10 - my_pit_stones;
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

    // 12. Midgame positional eval (ply 30-60: board mass, heavy pits, scatter penalty)
    let ply = board.move_count;
    if ply >= 30 && ply <= 60 {
        // Board mass bonus: keeping stones on our side = more options
        score += (my_pit_stones - opp_pit_stones) * 2;

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

        // Right-pit bonus in midgame too (lower weight)
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

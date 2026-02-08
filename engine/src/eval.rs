/// Handcrafted evaluation function for Тоғызқұмалақ
///
/// Returns score in centipawns from the side-to-move's perspective.
/// Positive = good for side to move, negative = bad.

use crate::board::{Board, NUM_PITS};

/// Evaluation weights (will be tuned via Texel tuning later)
const MATERIAL_WEIGHT: i32 = 100;     // per stone in kazan
const TUZDYK_BASE: i32 = 800;         // having a tuzdyk
const TUZDYK_CENTER_BONUS: i32 = 50;  // per position towards center (pits 3-7)
const THREAT_WEIGHT: i32 = 150;       // opponent pit with 2 stones (potential tuzdyk threat)
const PIT_STONES_WEIGHT: i32 = 10;    // stones in own pits (potential material)
const MOBILITY_WEIGHT: i32 = 20;      // per available move
const EMPTY_PIT_PENALTY: i32 = -15;   // per empty pit on own side
const LARGE_PIT_BONUS: i32 = 5;       // per stone in pits with 10+ stones
const ENDGAME_MATERIAL_BOOST: i32 = 50; // extra material weight in endgame

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

    // 2. Tuzdyk evaluation
    if board.tuzdyk[me] >= 0 {
        let pos = board.tuzdyk[me] as usize;
        score += TUZDYK_BASE;
        // Center tuzdyks are stronger (collect more stones passing through)
        let center_bonus = match pos {
            3 | 4 | 5 => 3,
            2 | 6 => 2,
            1 | 7 => 1,
            _ => 0,
        };
        score += center_bonus * TUZDYK_CENTER_BONUS;
    }
    if board.tuzdyk[opp] >= 0 {
        let pos = board.tuzdyk[opp] as usize;
        score -= TUZDYK_BASE;
        let center_bonus = match pos {
            3 | 4 | 5 => 3,
            2 | 6 => 2,
            1 | 7 => 1,
            _ => 0,
        };
        score -= center_bonus * TUZDYK_CENTER_BONUS;
    }

    // 3. Tuzdyk threats (opponent pits with 2 stones = one move from becoming 3)
    if board.tuzdyk[me] == -1 {
        // We don't have a tuzdyk yet — opponent pits with 2 are threats for US to create one
        for i in 0..8 {
            // pit 9 (index 8) can't become tuzdyk
            if board.pits[opp][i] == 2 && board.tuzdyk[opp] != i as i8 {
                score += THREAT_WEIGHT;
            }
        }
    }
    if board.tuzdyk[opp] == -1 {
        // Opponent doesn't have tuzdyk — our pits with 2 are threats for THEM
        for i in 0..8 {
            if board.pits[me][i] == 2 && board.tuzdyk[me] != i as i8 {
                score -= THREAT_WEIGHT / 2; // less weight since it's opponent's opportunity
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

    // 8. Capture opportunities: check if moves lead to immediate capture
    // Our capture opportunities
    for i in 0..NUM_PITS {
        if board.pits[me][i] > 0 && opp_tuzdyk != i as i8 {
            let landing = predict_landing(i, board.pits[me][i], me);
            if let Some((side, pit)) = landing {
                if side == opp {
                    let new_count = board.pits[opp][pit] + 1;
                    if new_count % 2 == 0 && new_count > 0 {
                        score += new_count as i32 * 8;
                    }
                }
            }
        }
    }
    // Opponent's capture opportunities
    for i in 0..NUM_PITS {
        if board.pits[opp][i] > 0 && me_tuzdyk != i as i8 {
            let landing = predict_landing(i, board.pits[opp][i], opp);
            if let Some((side, pit)) = landing {
                if side == me {
                    let new_count = board.pits[me][pit] + 1;
                    if new_count % 2 == 0 && new_count > 0 {
                        score -= new_count as i32 * 8;
                    }
                }
            }
        }
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

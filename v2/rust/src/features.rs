/// Feature extraction: Board -> tensor for neural network input
///
/// Produces a 70-dimensional feature vector normalized to [0,1] or [-1,1].
/// Features are always from the perspective of the current side to move.

use crate::board::{Board, NUM_PITS};

/// Total feature vector size
pub const FEATURE_SIZE: usize = 70;

/// Extract feature vector from a board position.
/// All values are normalized. Features are relative to the side to move.
pub fn board_to_features(board: &Board) -> [f32; FEATURE_SIZE] {
    let mut features = [0.0f32; FEATURE_SIZE];
    let side = board.side_to_move_u8();
    let opp = 1 - side;

    let my_pits = board.pits_for_side(side);
    let opp_pits = board.pits_for_side(opp);
    let my_kazan = board.kazan_for_side(side);
    let opp_kazan = board.kazan_for_side(opp);
    let my_tuzdyk = board.tuzdyk_for_side(side);
    let opp_tuzdyk = board.tuzdyk_for_side(opp);

    // === Basic features (40) — similar to V1 NNUE ===

    // [0-8] Stones in my pits, normalized by /50.0
    for i in 0..9 {
        features[i] = my_pits[i] as f32 / 50.0;
    }

    // [9-17] Stones in opponent's pits, /50.0
    for i in 0..9 {
        features[9 + i] = opp_pits[i] as f32 / 50.0;
    }

    // [18] My kazan, /82.0
    features[18] = my_kazan as f32 / 82.0;

    // [19] Opponent's kazan, /82.0
    features[19] = opp_kazan as f32 / 82.0;

    // [20-29] My tuzdyk (one-hot: 10 slots, index 9 = no tuzdyk)
    match my_tuzdyk {
        Some(idx) => features[20 + idx as usize] = 1.0,
        None => features[29] = 1.0,
    }

    // [30-39] Opponent's tuzdyk (one-hot)
    match opp_tuzdyk {
        Some(idx) => features[30 + idx as usize] = 1.0,
        None => features[39] = 1.0,
    }

    // === Strategic features (30) ===

    // [40-41] Mobility (number of non-empty pits / 9)
    let my_mobility = my_pits.iter().filter(|&&x| x > 0).count() as f32 / 9.0;
    let opp_mobility = opp_pits.iter().filter(|&&x| x > 0).count() as f32 / 9.0;
    features[40] = my_mobility;
    features[41] = opp_mobility;

    // [42-50] Capture threats: 1.0 if a move exists that would create even count in opp pit
    for i in 0..9 {
        features[42 + i] = if can_capture_pit(board, side, i) {
            1.0
        } else {
            0.0
        };
    }

    // [51-59] Tuzdyk potential: proximity of opponent pits to tuzdyk (=3)
    for i in 0..9 {
        let stones = opp_pits[i];
        features[51 + i] = if stones == 2 {
            1.0
        } else if stones == 1 {
            0.5
        } else {
            0.0
        };
    }

    // [60] Stones on board / 162 (game phase)
    features[60] = board.stones_on_board() as f32 / 162.0;

    // [61] Kazan difference / 82 (material advantage)
    features[61] = (my_kazan as f32 - opp_kazan as f32) / 82.0;

    // [62-63] Board stones per side / 81 (starvation risk)
    let my_board_stones: u32 = my_pits.iter().map(|&x| x as u32).sum();
    let opp_board_stones: u32 = opp_pits.iter().map(|&x| x as u32).sum();
    features[62] = my_board_stones as f32 / 81.0;
    features[63] = opp_board_stones as f32 / 81.0;

    // [64-65] Largest pit per side / 50
    features[64] = *my_pits.iter().max().unwrap() as f32 / 50.0;
    features[65] = *opp_pits.iter().max().unwrap() as f32 / 50.0;

    // [66-67] Empty pits per side / 9
    let my_empty = my_pits.iter().filter(|&&x| x == 0).count() as f32 / 9.0;
    let opp_empty = opp_pits.iter().filter(|&&x| x == 0).count() as f32 / 9.0;
    features[66] = my_empty;
    features[67] = opp_empty;

    // [68] Side to move (0.0 = White, 1.0 = Black)
    features[68] = side as f32;

    // [69] Game progress: (kazan_white + kazan_black) / 162
    features[69] =
        (board.kazan_for_side(0) as f32 + board.kazan_for_side(1) as f32) / 162.0;

    features
}

/// Check if there exists a move for `side` that would capture stones from opponent pit `target_pit`.
/// This is an approximation — we simulate each legal move and check if the target pit ends up
/// with an even number of stones after the move.
fn can_capture_pit(board: &Board, side: u8, target_pit: usize) -> bool {
    let me = side as usize;
    let opp = 1 - me;
    let opp_tuzdyk = board.tuzdyk[opp];

    for pit in 0..NUM_PITS {
        if board.pits[me][pit] == 0 {
            continue;
        }
        if opp_tuzdyk == pit as i8 {
            continue;
        }

        // Simulate the move to see where the last stone lands
        let stones = board.pits[me][pit];
        let mut current_pit = pit;
        let mut current_side = me;

        if stones == 1 {
            current_pit += 1;
            if current_pit > 8 {
                current_pit = 0;
                current_side = opp;
            }
        } else {
            // First stone back to source, then distribute remaining
            let mut remaining = stones - 1;
            while remaining > 0 {
                current_pit += 1;
                if current_pit > 8 {
                    current_pit = 0;
                    current_side = 1 - current_side;
                }
                remaining -= 1;
            }
        }

        // Check if last stone landed on the target pit on opponent's side
        if current_side == opp && current_pit == target_pit {
            // Approximate: count would be current stones + 1 (the landing stone)
            // (ignoring stones sowed along the way for simplicity)
            let final_count = board.pits[opp][target_pit] + 1;
            if final_count % 2 == 0 && final_count > 0 {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_size() {
        let board = Board::new();
        let features = board_to_features(&board);
        assert_eq!(features.len(), FEATURE_SIZE);
        assert_eq!(features.len(), 70);
    }

    #[test]
    fn test_initial_features() {
        let board = Board::new();
        let f = board_to_features(&board);

        // All pits have 9 stones -> 9/50 = 0.18
        for i in 0..18 {
            assert!((f[i] - 0.18).abs() < 0.01, "pit feature {} = {}", i, f[i]);
        }

        // Kazans are 0
        assert_eq!(f[18], 0.0);
        assert_eq!(f[19], 0.0);

        // No tuzdyks -> index 9 (=29) and index 9 (=39) are 1.0
        assert_eq!(f[29], 1.0); // my tuzdyk = none
        assert_eq!(f[39], 1.0); // opp tuzdyk = none

        // Full mobility
        assert!((f[40] - 1.0).abs() < 0.01); // my mobility = 9/9
        assert!((f[41] - 1.0).abs() < 0.01); // opp mobility = 9/9

        // Stones on board = 162/162 = 1.0
        assert!((f[60] - 1.0).abs() < 0.01);

        // Kazan difference = 0
        assert_eq!(f[61], 0.0);

        // Side to move = White = 0
        assert_eq!(f[68], 0.0);

        // Game progress = 0
        assert_eq!(f[69], 0.0);
    }

    #[test]
    fn test_features_symmetry() {
        // Features should be relative to side to move
        let mut board_w = Board::new();
        board_w.pits[0] = [5, 10, 0, 0, 0, 0, 0, 0, 0];
        board_w.pits[1] = [3, 7, 0, 0, 0, 0, 0, 0, 0];
        board_w.kazan = [60, 30];
        board_w.side_to_move = crate::board::Side::White;

        let mut board_b = board_w.clone();
        board_b.pits[0] = [3, 7, 0, 0, 0, 0, 0, 0, 0]; // swapped
        board_b.pits[1] = [5, 10, 0, 0, 0, 0, 0, 0, 0];
        board_b.kazan = [30, 60]; // swapped
        board_b.side_to_move = crate::board::Side::Black;

        let f_w = board_to_features(&board_w);
        let f_b = board_to_features(&board_b);

        // My pits should match (first 9 features)
        for i in 0..9 {
            assert!(
                (f_w[i] - f_b[i]).abs() < 0.001,
                "my pit {} mismatch: {} vs {}",
                i,
                f_w[i],
                f_b[i]
            );
        }

        // Opp pits should match
        for i in 9..18 {
            assert!(
                (f_w[i] - f_b[i]).abs() < 0.001,
                "opp pit {} mismatch: {} vs {}",
                i,
                f_w[i],
                f_b[i]
            );
        }
    }

    #[test]
    fn test_features_all_bounded() {
        let board = Board::new();
        let f = board_to_features(&board);
        for (i, &val) in f.iter().enumerate() {
            assert!(
                val >= -1.0 && val <= 2.0,
                "Feature {} out of bounds: {}",
                i,
                val
            );
        }
    }
}

/// Board state encoding for neural network input.
/// Must match Python game.py::encode_state() exactly.
///
/// 7 channels x 9 positions = 63 floats:
///   Ch 0: current player pits / 50.0
///   Ch 1: opponent pits / 50.0
///   Ch 2: current player kazan / 82.0 (broadcast)
///   Ch 3: opponent kazan / 82.0 (broadcast)
///   Ch 4: current player tuzdyk (one-hot)
///   Ch 5: opponent tuzdyk (one-hot)
///   Ch 6: player indicator (1.0 if White, 0.0 if Black)

use crate::board::{Board, Side, NUM_PITS};

pub const NUM_CHANNELS: usize = 7;
pub const ENCODED_SIZE: usize = NUM_CHANNELS * NUM_PITS; // 63

const PIT_NORM: f32 = 50.0;
const KAZAN_NORM: f32 = 82.0;

/// Encode board state into [7, 9] flattened f32 array.
/// Layout: channel-major, i.e. [ch0_pos0..ch0_pos8, ch1_pos0..ch1_pos8, ...]
pub fn encode_state(board: &Board) -> [f32; ENCODED_SIZE] {
    let mut out = [0.0f32; ENCODED_SIZE];

    let me = board.side_to_move.index();
    let opp = 1 - me;

    // Channel 0: current player pits (normalized)
    for i in 0..NUM_PITS {
        out[0 * NUM_PITS + i] = board.pits[me][i] as f32 / PIT_NORM;
    }

    // Channel 1: opponent pits (normalized)
    for i in 0..NUM_PITS {
        out[1 * NUM_PITS + i] = board.pits[opp][i] as f32 / PIT_NORM;
    }

    // Channel 2: current player kazan (broadcast)
    let my_kazan = board.kazan[me] as f32 / KAZAN_NORM;
    for i in 0..NUM_PITS {
        out[2 * NUM_PITS + i] = my_kazan;
    }

    // Channel 3: opponent kazan (broadcast)
    let opp_kazan = board.kazan[opp] as f32 / KAZAN_NORM;
    for i in 0..NUM_PITS {
        out[3 * NUM_PITS + i] = opp_kazan;
    }

    // Channel 4: current player tuzdyk (one-hot)
    if board.tuzdyk[me] >= 0 {
        out[4 * NUM_PITS + board.tuzdyk[me] as usize] = 1.0;
    }

    // Channel 5: opponent tuzdyk (one-hot)
    if board.tuzdyk[opp] >= 0 {
        out[5 * NUM_PITS + board.tuzdyk[opp] as usize] = 1.0;
    }

    // Channel 6: player indicator (1.0 if White, 0.0 if Black)
    let indicator = if board.side_to_move == Side::White { 1.0 } else { 0.0 };
    for i in 0..NUM_PITS {
        out[6 * NUM_PITS + i] = indicator;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn test_initial_encoding() {
        let board = Board::new();
        let enc = encode_state(&board);

        // Channel 0: White pits = 9/50 = 0.18
        for i in 0..9 {
            assert!((enc[i] - 0.18).abs() < 1e-6, "ch0[{}] = {}", i, enc[i]);
        }
        // Channel 1: Black pits = 9/50 = 0.18
        for i in 0..9 {
            assert!((enc[9 + i] - 0.18).abs() < 1e-6);
        }
        // Channel 2: White kazan = 0/82 = 0.0
        for i in 0..9 {
            assert!((enc[18 + i] - 0.0).abs() < 1e-6);
        }
        // Channel 4,5: tuzdyks = 0 (none)
        for i in 0..9 {
            assert!((enc[36 + i] - 0.0).abs() < 1e-6);
            assert!((enc[45 + i] - 0.0).abs() < 1e-6);
        }
        // Channel 6: White = 1.0
        for i in 0..9 {
            assert!((enc[54 + i] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_encoding_black_to_move() {
        let mut board = Board::new();
        board.make_move(0); // White plays, now Black to move

        let enc = encode_state(&board);

        // Channel 6: Black = 0.0
        for i in 0..9 {
            assert!((enc[54 + i] - 0.0).abs() < 1e-6);
        }
        // Channel 0 should be Black's pits (current player)
        // Channel 1 should be White's pits (opponent)
    }
}

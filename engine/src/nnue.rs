/// NNUE inference for Тоғызқұмалақ
///
/// Supports two architectures:
///   Legacy (40 inputs):  Input(40) → Linear(256) → CReLU → Linear(32) → CReLU → Linear(1)
///   Extended (58 inputs): Input(58) → Linear(256) → CReLU → Linear(32) → CReLU → Linear(1)
///
/// Binary format detection:
///   Old format: first u16 >= 128 → treated as hidden1 size (backward compat)
///   New format: first u16 < 128 → treated as input_size (40, 52, or 58)
///
/// New format header: [input_size: u16, hidden1: u16, hidden2: u16, hidden3: u16]
/// Old format header: [hidden1: u16, hidden2: u16]
///
/// Input features (40-input):
///   [0..9]   current player pits (raw value, scaled in first layer)
///   [9..18]  opponent pits
///   [18]     current player kazan
///   [19]     opponent kazan
///   [20..30] current player tuzdyk one-hot
///   [30..40] opponent tuzdyk one-hot
///
/// Additional features (58-input, indices 40-57):
///   [40] my total pit stones / 81
///   [41] opp total pit stones / 81
///   [42] my active pits (non-empty count) / 9
///   [43] opp active pits / 9
///   [44] my heavy pits (>=12 stones) / 9
///   [45] opp heavy pits / 9
///   [46] my weak pits (1-2 stones) / 9
///   [47] opp weak pits / 9
///   [48] my right pits (pit7+8+9 stones) / 81
///   [49] opp right pits / 81
///   [50] game phase (total board stones / 162)
///   [51] kazan difference (my_kazan - opp_kazan) / 82
///   [52] my tuzdyk threats (opp pits with exactly 2 stones) / 8
///   [53] opp tuzdyk threats (my pits with exactly 2 stones) / 8
///   [54] opp starvation pressure: max(0, 20-opp_stones)^2 / 400
///   [55] my starvation pressure: max(0, 20-my_stones)^2 / 400
///   [56] my capture targets (opp pits with even stones > 0) / 9
///   [57] opp capture targets (my pits with even stones > 0) / 9

use std::fs::File;
use std::io::Read;

use crate::board::{Board, NUM_PITS};

const SCALE: i32 = 64;
const MAX_HIDDEN1: usize = 512;
const MAX_HIDDEN2: usize = 64;
const MAX_HIDDEN3: usize = 64;

#[derive(Clone)]
pub struct NnueNetwork {
    input_size: usize,
    hidden1: usize,
    hidden2: usize,
    hidden3: usize,  // 0 = no third hidden layer (legacy)
    fc1_weight: Vec<i16>,  // [hidden1 * input_size]
    fc1_bias: Vec<i16>,    // [hidden1]
    fc2_weight: Vec<i16>,  // [hidden2 * hidden1]
    fc2_bias: Vec<i16>,    // [hidden2]
    fc3_weight: Vec<i16>,  // [hidden3 * hidden2] or [1 * hidden2] if no hidden3
    fc3_bias: Vec<i16>,
    fc4_weight: Vec<i16>,  // [1 * hidden3], empty if no hidden3
    fc4_bias: Vec<i16>,
}

impl NnueNetwork {
    /// Load from binary file — auto-detects format
    pub fn load(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).map_err(|e| format!("Failed to read: {}", e))?;

        if data.len() < 4 {
            return Err("File too small".into());
        }

        let read_i16_vec = |data: &[u8], offset: &mut usize, count: usize| -> Result<Vec<i16>, String> {
            let bytes_needed = count * 2;
            if *offset + bytes_needed > data.len() {
                return Err(format!("Unexpected EOF at offset {}", *offset));
            }
            let mut vec = Vec::with_capacity(count);
            for i in 0..count {
                let lo = data[*offset + i * 2];
                let hi = data[*offset + i * 2 + 1];
                vec.push(i16::from_le_bytes([lo, hi]));
            }
            *offset += bytes_needed;
            Ok(vec)
        };

        let first_val = u16::from_le_bytes([data[0], data[1]]) as usize;

        // New format: first u16 is input_size (40 or 52), < 128
        // Old format: first u16 is hidden1 (256 or 512), >= 128
        if first_val < 128 {
            // New format: [input_size, hidden1, hidden2, hidden3]
            if data.len() < 8 {
                return Err("New format requires 8-byte header".into());
            }
            let input_size = first_val;
            let hidden1 = u16::from_le_bytes([data[2], data[3]]) as usize;
            let hidden2 = u16::from_le_bytes([data[4], data[5]]) as usize;
            let hidden3 = u16::from_le_bytes([data[6], data[7]]) as usize;

            let mut offset = 8;
            let fc1_weight = read_i16_vec(&data, &mut offset, hidden1 * input_size)?;
            let fc1_bias = read_i16_vec(&data, &mut offset, hidden1)?;
            let fc2_weight = read_i16_vec(&data, &mut offset, hidden2 * hidden1)?;
            let fc2_bias = read_i16_vec(&data, &mut offset, hidden2)?;

            if hidden3 > 0 {
                // 4-layer: fc3 = hidden2→hidden3, fc4 = hidden3→1
                let fc3_weight = read_i16_vec(&data, &mut offset, hidden3 * hidden2)?;
                let fc3_bias = read_i16_vec(&data, &mut offset, hidden3)?;
                let fc4_weight = read_i16_vec(&data, &mut offset, hidden3)?;
                let fc4_bias = read_i16_vec(&data, &mut offset, 1)?;

                eprintln!(
                    "NNUE loaded: {} → {} → {} → {} → 1 ({} params)",
                    input_size, hidden1, hidden2, hidden3,
                    fc1_weight.len() + fc1_bias.len() + fc2_weight.len() + fc2_bias.len()
                        + fc3_weight.len() + fc3_bias.len() + fc4_weight.len() + fc4_bias.len()
                );

                Ok(NnueNetwork {
                    input_size, hidden1, hidden2, hidden3,
                    fc1_weight, fc1_bias, fc2_weight, fc2_bias,
                    fc3_weight, fc3_bias, fc4_weight, fc4_bias,
                })
            } else {
                // 3-layer with custom input_size: fc3 = hidden2→1
                let fc3_weight = read_i16_vec(&data, &mut offset, hidden2)?;
                let fc3_bias = read_i16_vec(&data, &mut offset, 1)?;

                eprintln!(
                    "NNUE loaded: {} → {} → {} → 1 ({} params)",
                    input_size, hidden1, hidden2,
                    fc1_weight.len() + fc1_bias.len() + fc2_weight.len() + fc2_bias.len()
                        + fc3_weight.len() + fc3_bias.len()
                );

                Ok(NnueNetwork {
                    input_size, hidden1, hidden2, hidden3: 0,
                    fc1_weight, fc1_bias, fc2_weight, fc2_bias,
                    fc3_weight, fc3_bias, fc4_weight: vec![], fc4_bias: vec![],
                })
            }
        } else {
            // Old format: [hidden1, hidden2] (backward compat)
            let hidden1 = first_val;
            let hidden2 = u16::from_le_bytes([data[2], data[3]]) as usize;
            let input_size = 40;

            let mut offset = 4;
            let fc1_weight = read_i16_vec(&data, &mut offset, hidden1 * input_size)?;
            let fc1_bias = read_i16_vec(&data, &mut offset, hidden1)?;
            let fc2_weight = read_i16_vec(&data, &mut offset, hidden2 * hidden1)?;
            let fc2_bias = read_i16_vec(&data, &mut offset, hidden2)?;
            let fc3_weight = read_i16_vec(&data, &mut offset, hidden2)?;
            let fc3_bias = read_i16_vec(&data, &mut offset, 1)?;

            eprintln!(
                "NNUE loaded: {} → {} → {} → 1 ({} params)",
                input_size, hidden1, hidden2,
                fc1_weight.len() + fc1_bias.len() + fc2_weight.len() + fc2_bias.len()
                    + fc3_weight.len() + fc3_bias.len()
            );

            Ok(NnueNetwork {
                input_size, hidden1, hidden2, hidden3: 0,
                fc1_weight, fc1_bias, fc2_weight, fc2_bias,
                fc3_weight, fc3_bias, fc4_weight: vec![], fc4_bias: vec![],
            })
        }
    }

    /// Build 40-input feature vector
    #[inline]
    fn build_input_40(board: &Board, input: &mut [i16]) {
        let me = board.side_to_move.index();
        let opp = 1 - me;

        for i in 0..NUM_PITS {
            input[i] = (board.pits[me][i] as i32 * SCALE / 50) as i16;
            input[9 + i] = (board.pits[opp][i] as i32 * SCALE / 50) as i16;
        }
        input[18] = (board.kazan[me] as i32 * SCALE / 82) as i16;
        input[19] = (board.kazan[opp] as i32 * SCALE / 82) as i16;

        let my_tuz = board.tuzdyk[me];
        let opp_tuz = board.tuzdyk[opp];
        if my_tuz >= 0 {
            input[20 + my_tuz as usize] = SCALE as i16;
        } else {
            input[29] = SCALE as i16;
        }
        if opp_tuz >= 0 {
            input[30 + opp_tuz as usize] = SCALE as i16;
        } else {
            input[39] = SCALE as i16;
        }
    }

    /// Build 58-input feature vector (40 base + 18 strategic features)
    #[inline]
    fn build_input_58(board: &Board, input: &mut [i16]) {
        let me = board.side_to_move.index();
        let opp = 1 - me;

        // Base 40 features
        Self::build_input_40(board, input);

        // Feature 40-41: total pit stones / 81
        let my_stones: u32 = board.pits[me].iter().map(|&x| x as u32).sum();
        let opp_stones: u32 = board.pits[opp].iter().map(|&x| x as u32).sum();
        input[40] = (my_stones as i32 * SCALE / 81) as i16;
        input[41] = (opp_stones as i32 * SCALE / 81) as i16;

        // Feature 42-43: active pits (non-empty) / 9
        let my_active = board.pits[me].iter().filter(|&&x| x > 0).count() as i32;
        let opp_active = board.pits[opp].iter().filter(|&&x| x > 0).count() as i32;
        input[42] = (my_active * SCALE / 9) as i16;
        input[43] = (opp_active * SCALE / 9) as i16;

        // Feature 44-45: heavy pits (>=12 stones) / 9
        let my_heavy = board.pits[me].iter().filter(|&&x| x >= 12).count() as i32;
        let opp_heavy = board.pits[opp].iter().filter(|&&x| x >= 12).count() as i32;
        input[44] = (my_heavy * SCALE / 9) as i16;
        input[45] = (opp_heavy * SCALE / 9) as i16;

        // Feature 46-47: weak pits (1-2 stones) / 9
        let my_weak = board.pits[me].iter().filter(|&&x| x >= 1 && x <= 2).count() as i32;
        let opp_weak = board.pits[opp].iter().filter(|&&x| x >= 1 && x <= 2).count() as i32;
        input[46] = (my_weak * SCALE / 9) as i16;
        input[47] = (opp_weak * SCALE / 9) as i16;

        // Feature 48-49: right pits (pit7+8+9, indices 6-8) / 81
        let my_right: u32 = board.pits[me][6..9].iter().map(|&x| x as u32).sum();
        let opp_right: u32 = board.pits[opp][6..9].iter().map(|&x| x as u32).sum();
        input[48] = (my_right as i32 * SCALE / 81) as i16;
        input[49] = (opp_right as i32 * SCALE / 81) as i16;

        // Feature 50: game phase (total board stones / 162)
        let total = my_stones + opp_stones;
        input[50] = (total as i32 * SCALE / 162) as i16;

        // Feature 51: kazan difference / 82
        let kaz_diff = board.kazan[me] as i32 - board.kazan[opp] as i32;
        input[51] = (kaz_diff * SCALE / 82).clamp(-SCALE, SCALE) as i16;

        // === NEW: 6 strategic features (52-57) ===

        // Feature 52-53: tuzdyk threats
        // My threats = opponent pits with exactly 2 stones (tuzdyk candidates for me)
        // Only relevant if I don't already have a tuzdyk
        let my_tuz = board.tuzdyk[me];
        let opp_tuz = board.tuzdyk[opp];
        let my_threats = if my_tuz == -1 {
            board.pits[opp].iter().enumerate()
                .filter(|&(i, &x)| x == 2 && i < 8 && opp_tuz != i as i8)
                .count() as i32
        } else { 0 };
        let opp_threats = if opp_tuz == -1 {
            board.pits[me].iter().enumerate()
                .filter(|&(i, &x)| x == 2 && i < 8 && my_tuz != i as i8)
                .count() as i32
        } else { 0 };
        input[52] = (my_threats * SCALE / 8) as i16;
        input[53] = (opp_threats * SCALE / 8) as i16;

        // Feature 54-55: starvation pressure (quadratic)
        // max(0, 20 - stones)^2 / 400, normalized to [0, SCALE]
        let opp_pressure = (20i32.saturating_sub(opp_stones as i32)).max(0);
        let my_pressure = (20i32.saturating_sub(my_stones as i32)).max(0);
        input[54] = (opp_pressure * opp_pressure * SCALE / 400) as i16;
        input[55] = (my_pressure * my_pressure * SCALE / 400) as i16;

        // Feature 56-57: capture targets (opponent pits with even stones > 0)
        let my_captures = board.pits[opp].iter()
            .filter(|&&x| x > 0 && x % 2 == 0).count() as i32;
        let opp_captures = board.pits[me].iter()
            .filter(|&&x| x > 0 && x % 2 == 0).count() as i32;
        input[56] = (my_captures * SCALE / 9) as i16;
        input[57] = (opp_captures * SCALE / 9) as i16;
    }

    /// Evaluate a position. Returns score from side-to-move perspective.
    #[inline]
    pub fn evaluate(&self, board: &Board) -> i32 {
        let mut input_buf = [0i16; 58];
        let input: &[i16] = if self.input_size >= 58 {
            Self::build_input_58(board, &mut input_buf);
            &input_buf[..58]
        } else if self.input_size >= 52 {
            Self::build_input_58(board, &mut input_buf);
            &input_buf[..52]
        } else {
            Self::build_input_40(board, &mut input_buf);
            &input_buf[..40]
        };

        let mut hidden1_buf = [0i32; MAX_HIDDEN1];
        let mut hidden2_buf = [0i32; MAX_HIDDEN2];
        let mut hidden3_buf = [0i32; MAX_HIDDEN3];
        let h1 = self.hidden1;
        let h2 = self.hidden2;
        let h3 = self.hidden3;
        let in_sz = self.input_size;

        // Layer 1: input → hidden1
        for j in 0..h1 {
            let mut acc = self.fc1_bias[j] as i32 * SCALE;
            let w = &self.fc1_weight[j * in_sz..j * in_sz + in_sz];
            for i in 0..in_sz {
                acc += w[i] as i32 * input[i] as i32;
            }
            hidden1_buf[j] = (acc / SCALE).clamp(0, SCALE);
        }

        // Layer 2: hidden1 → hidden2
        for j in 0..h2 {
            let mut acc = self.fc2_bias[j] as i32 * SCALE;
            let w = &self.fc2_weight[j * h1..j * h1 + h1];
            for i in 0..h1 {
                acc += w[i] as i32 * hidden1_buf[i];
            }
            hidden2_buf[j] = (acc / SCALE).clamp(0, SCALE);
        }

        if h3 > 0 {
            // 4-layer network: hidden2 → hidden3 → output
            for j in 0..h3 {
                let mut acc = self.fc3_bias[j] as i32 * SCALE;
                let w = &self.fc3_weight[j * h2..j * h2 + h2];
                for i in 0..h2 {
                    acc += w[i] as i32 * hidden2_buf[i];
                }
                hidden3_buf[j] = (acc / SCALE).clamp(0, SCALE);
            }
            let mut output = self.fc4_bias[0] as i32 * SCALE;
            for i in 0..h3 {
                output += self.fc4_weight[i] as i32 * hidden3_buf[i];
            }
            output / SCALE
        } else {
            // 3-layer network: hidden2 → output
            let mut output = self.fc3_bias[0] as i32 * SCALE;
            for i in 0..h2 {
                output += self.fc3_weight[i] as i32 * hidden2_buf[i];
            }
            output / SCALE
        }
    }
}

// NNUE v2: Sparse bucketed feature encoding (292 total features, 23 active)
pub const NUM_FEATURES_V2: usize = 292;
pub const NUM_BUCKETS_V2: usize = 4;
const ACTIVE_FEATURES_V2: usize = 23;

/// Stone counts enter as one-hot buckets, not as a scalar: endgames turn on exact counts
/// and parity (a pit holding exactly 2 is a tuzdyk threat), and a first layer over a
/// scaled scalar can only rescale it.
fn count_bucket(c: u8) -> usize {
    match c {
        0..=9 => c as usize,
        10..=12 => 10,
        13..=16 => 11,
        17..=24 => 12,
        _ => 13,
    }
}

fn board_stones(board: &Board) -> u32 {
    (0..NUM_PITS)
        .map(|i| board.pits[0][i] as u32 + board.pits[1][i] as u32)
        .sum()
}

pub fn build_features_v2(board: &Board) -> Vec<u16> {
    let me = board.side_to_move.index();
    let opp = 1 - me;
    let mut f = Vec::with_capacity(ACTIVE_FEATURES_V2);
    for i in 0..NUM_PITS {
        f.push((i * 14 + count_bucket(board.pits[me][i])) as u16);
    }
    for i in 0..NUM_PITS {
        f.push((126 + i * 14 + count_bucket(board.pits[opp][i])) as u16);
    }
    f.push((252 + (board.kazan[me] as usize / 10).min(8)) as u16);
    f.push((261 + (board.kazan[opp] as usize / 10).min(8)) as u16);
    let tuz = |t: i8| if t >= 0 { t as usize } else { 9 };
    f.push((270 + tuz(board.tuzdyk[me])) as u16);
    f.push((280 + tuz(board.tuzdyk[opp])) as u16);
    f.push((290 + (board_stones(board) % 2) as usize) as u16);
    debug_assert_eq!(f.len(), ACTIVE_FEATURES_V2);
    f
}

pub fn phase_bucket(board: &Board) -> usize {
    match board_stones(board) {
        121..=162 => 0,
        81..=120 => 1,
        41..=80 => 2,
        _ => 3,
    }
}

#[cfg(test)]
mod tests_v2 {
    use super::*;
    use crate::board::Board;

    #[test]
    fn start_position_features() {
        let b = Board::new();
        let f = build_features_v2(&b);
        assert_eq!(f.len(), 23, "23 active features per position");
        // every pit holds 9 stones -> bucket 9
        for i in 0..9 {
            assert!(f.contains(&((i * 14 + 9) as u16)), "me pit {i} bucket 9");
            assert!(f.contains(&((126 + i * 14 + 9) as u16)), "opp pit {i} bucket 9");
        }
        assert!(f.contains(&252), "me kazan 0 -> bucket 0");
        assert!(f.contains(&261), "opp kazan 0 -> bucket 0");
        assert!(f.contains(&(270 + 9)), "no tuzdyk for me");
        assert!(f.contains(&(280 + 9)), "no tuzdyk for opp");
        assert!(f.contains(&290), "162 stones on board -> even parity");
        assert_eq!(phase_bucket(&b), 0, "full board is phase bucket 0");
    }

    #[test]
    fn count_buckets_are_step_functions() {
        assert_eq!(count_bucket(0), 0);
        assert_eq!(count_bucket(2), 2);      // the tuzdyk-threat count must be its own bucket
        assert_eq!(count_bucket(9), 9);
        assert_eq!(count_bucket(10), 10);
        assert_eq!(count_bucket(12), 10);
        assert_eq!(count_bucket(13), 11);
        assert_eq!(count_bucket(16), 11);
        assert_eq!(count_bucket(17), 12);
        assert_eq!(count_bucket(24), 12);
        assert_eq!(count_bucket(25), 13);
        assert_eq!(count_bucket(90), 13);
    }

    #[test]
    fn tuzdyk_and_phase_are_encoded() {
        let mut b = Board::new();
        for i in 0..9 {
            b.pits[0][i] = 1;
            b.pits[1][i] = 1;
        }
        b.kazan[0] = 72;
        b.kazan[1] = 72;
        b.tuzdyk[0] = 6;
        b.tuzdyk[1] = -1;
        let f = build_features_v2(&b);
        assert!(f.contains(&(270 + 6)), "me tuzdyk on pit 6");
        assert!(f.contains(&(280 + 9)), "opp has no tuzdyk");
        assert!(f.contains(&(252 + 7)), "kazan 72 -> bucket 7");
        assert_eq!(phase_bucket(&b), 3, "18 stones on board is the last phase bucket");
    }
}

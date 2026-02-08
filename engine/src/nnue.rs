/// NNUE inference for Тоғызқұмалақ
///
/// Architecture: Input(40) → Linear(256) → ClippedReLU → Linear(32) → ClippedReLU → Linear(1)
///
/// Quantized integer arithmetic:
///   - Weights stored as i16 (scaled by SCALE=64)
///   - Accumulator is i32
///   - ClippedReLU: clamp to [0, SCALE]
///
/// Input features (40):
///   [0..9]   current player pits (raw value, scaled in first layer)
///   [9..18]  opponent pits
///   [18]     current player kazan
///   [19]     opponent kazan
///   [20..30] current player tuzdyk one-hot
///   [30..40] opponent tuzdyk one-hot

use std::fs::File;
use std::io::Read;

use crate::board::{Board, NUM_PITS};

const INPUT_SIZE: usize = 40;
const SCALE: i32 = 64;

pub struct NnueNetwork {
    hidden1: usize,
    hidden2: usize,
    // Weights stored as i16 (float * SCALE)
    fc1_weight: Vec<i16>,  // [hidden1 * INPUT_SIZE]
    fc1_bias: Vec<i16>,    // [hidden1]
    fc2_weight: Vec<i16>,  // [hidden2 * hidden1]
    fc2_bias: Vec<i16>,    // [hidden2]
    fc3_weight: Vec<i16>,  // [1 * hidden2]
    fc3_bias: Vec<i16>,    // [1]
}

impl NnueNetwork {
    /// Load from binary file
    pub fn load(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).map_err(|e| format!("Failed to read: {}", e))?;

        if data.len() < 4 {
            return Err("File too small".into());
        }

        let hidden1 = u16::from_le_bytes([data[0], data[1]]) as usize;
        let hidden2 = u16::from_le_bytes([data[2], data[3]]) as usize;

        let mut offset = 4;

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

        let fc1_weight = read_i16_vec(&data, &mut offset, hidden1 * INPUT_SIZE)?;
        let fc1_bias = read_i16_vec(&data, &mut offset, hidden1)?;
        let fc2_weight = read_i16_vec(&data, &mut offset, hidden2 * hidden1)?;
        let fc2_bias = read_i16_vec(&data, &mut offset, hidden2)?;
        let fc3_weight = read_i16_vec(&data, &mut offset, hidden2)?;
        let fc3_bias = read_i16_vec(&data, &mut offset, 1)?;

        eprintln!(
            "NNUE loaded: {} → {} → {} → 1 ({} params)",
            INPUT_SIZE,
            hidden1,
            hidden2,
            fc1_weight.len() + fc1_bias.len() + fc2_weight.len() + fc2_bias.len()
                + fc3_weight.len() + fc3_bias.len()
        );

        Ok(NnueNetwork {
            hidden1,
            hidden2,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            fc3_weight,
            fc3_bias,
        })
    }

    /// Evaluate a position using NNUE. Returns score in centipawns from side-to-move.
    pub fn evaluate(&self, board: &Board) -> i32 {
        // Build input features as fixed-point (multiply by SCALE)
        let me = board.side_to_move.index();
        let opp = 1 - me;

        let mut input = [0i16; INPUT_SIZE];

        // Pits (scaled: value * SCALE / 50, but we'll let the network handle scaling)
        // Actually, we trained with / 50.0 normalization, so we need to match.
        // Input is raw_value / 50.0, then multiplied by SCALE for fixed-point.
        // So input_fp = raw_value * SCALE / 50
        for i in 0..NUM_PITS {
            input[i] = (board.pits[me][i] as i32 * SCALE / 50) as i16;
            input[9 + i] = (board.pits[opp][i] as i32 * SCALE / 50) as i16;
        }

        // Kazan (normalized by /82)
        input[18] = (board.kazan[me] as i32 * SCALE / 82) as i16;
        input[19] = (board.kazan[opp] as i32 * SCALE / 82) as i16;

        // Tuzdyk one-hot (SCALE = 1.0 in fixed-point)
        let my_tuz = board.tuzdyk[me];
        let opp_tuz = board.tuzdyk[opp];
        if my_tuz >= 0 {
            input[20 + my_tuz as usize] = SCALE as i16;
        } else {
            input[29] = SCALE as i16; // index 9 = "no tuzdyk"
        }
        if opp_tuz >= 0 {
            input[30 + opp_tuz as usize] = SCALE as i16;
        } else {
            input[39] = SCALE as i16;
        }

        // Layer 1: input(40) → hidden1
        // acc = weight * input + bias
        // weight is already scaled by SCALE, input is scaled by SCALE
        // So acc = (w_fp * in_fp) = (w_f * SCALE) * (in_f * SCALE) = w_f * in_f * SCALE^2
        // We need to divide by SCALE to get result in SCALE units
        let mut hidden1 = vec![0i32; self.hidden1];
        for j in 0..self.hidden1 {
            let mut acc = self.fc1_bias[j] as i32 * SCALE; // bias is in SCALE units, output in SCALE^2
            let w_offset = j * INPUT_SIZE;
            for i in 0..INPUT_SIZE {
                acc += self.fc1_weight[w_offset + i] as i32 * input[i] as i32;
            }
            // Divide by SCALE to get back to SCALE units, then ClippedReLU [0, SCALE]
            let val = acc / SCALE;
            hidden1[j] = val.clamp(0, SCALE);
        }

        // Layer 2: hidden1 → hidden2
        let mut hidden2 = vec![0i32; self.hidden2];
        for j in 0..self.hidden2 {
            let mut acc = self.fc2_bias[j] as i32 * SCALE;
            let w_offset = j * self.hidden1;
            for i in 0..self.hidden1 {
                acc += self.fc2_weight[w_offset + i] as i32 * hidden1[i];
            }
            let val = acc / SCALE;
            hidden2[j] = val.clamp(0, SCALE);
        }

        // Layer 3: hidden2 → output (1)
        let mut output = self.fc3_bias[0] as i32 * SCALE;
        for i in 0..self.hidden2 {
            output += self.fc3_weight[i] as i32 * hidden2[i];
        }
        // Final output: divide by SCALE to get back to "eval" units
        output / SCALE
    }
}

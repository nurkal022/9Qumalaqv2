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

#[derive(Clone, Default)]
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

    // --- v2/v3 (NNU2) shared ---
    v2: bool,             // true for either NNU2 sub-format (f32 version 2 or i16 version 3)
    nnu2_version: u16,    // 2 = f32 (load_nnu2_f32), 3 = i16 quantised (load_nnu2_i16); 0 if !v2
    acc_size: usize,
    buckets: usize,

    // --- NNU2 version 2 (f32) ---
    v2_fc1_w: Vec<f32>,   // [num_features * acc_size]
    v2_fc1_b: Vec<f32>,   // [acc_size]
    v2_fc2_w: Vec<f32>,   // [buckets][hidden * acc_size] flattened
    v2_fc2_b: Vec<f32>,   // [buckets][hidden] flattened
    v2_fc3_w: Vec<f32>,   // [buckets][hidden] flattened
    v2_fc3_b: Vec<f32>,   // [buckets]

    // --- NNU2 version 3 (i16, quantised) ---
    // Scale factors: quantised = round(real * scale), real ~= quantised / scale.
    // fc1_w and fc1_b share one scale because they are summed directly into the same
    // i32 accumulator; each bucket's fc2/fc3 weights get their own scale because their
    // magnitude varies a lot bucket to bucket (see NnueNetwork::pick_scale).
    v3_fc1_w: Vec<i16>,   // [num_features * acc_size]
    v3_fc1_b: Vec<i16>,   // [acc_size]
    v3_fc2_w: Vec<i16>,   // [buckets][hidden * acc_size] flattened
    v3_fc2_b: Vec<i16>,   // [buckets][hidden] flattened
    v3_fc3_w: Vec<i16>,   // [buckets][hidden] flattened
    v3_fc3_b: Vec<i16>,   // [buckets]
    v3_scale_l1: f32,
    v3_scale_fc2: Vec<f32>,  // [buckets]
    v3_scale_fc3: Vec<f32>,  // [buckets]
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

        if data.len() >= 4 && u32::from_le_bytes([data[0], data[1], data[2], data[3]]) == 0x324E554E {
            return Self::load_v2(&data);
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
                    ..Default::default()
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
                    fc3_weight, fc3_bias,
                    ..Default::default()
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
                fc3_weight, fc3_bias,
                ..Default::default()
            })
        }
    }

    /// Dispatch on the NNU2 `version` field (offset 4, right after the magic) to the
    /// matching sub-format loader. The version field is the *only* thing that
    /// disambiguates the two payload layouts below it — nothing else in the header is
    /// reused with a different meaning between them.
    fn load_v2(data: &[u8]) -> Result<Self, String> {
        if data.len() < 16 {
            return Err(format!(
                "NNU2 header truncated: need 16 bytes, have {}",
                data.len()
            ));
        }
        let rd_u16 = |off: usize| u16::from_le_bytes([data[off], data[off + 1]]) as usize;
        let version = rd_u16(4);
        match version {
            2 => Self::load_nnu2_f32(data),
            3 => Self::load_nnu2_i16(data),
            v => Err(format!("unsupported NNU2 version {v}")),
        }
    }

    /// NNU2 version 2: f32 weights everywhere. The original format; unchanged since
    /// task 6 so it stays a valid comparison point for the version-3 (i16) path.
    fn load_nnu2_f32(data: &[u8]) -> Result<Self, String> {
        let rd_u16 = |off: usize| u16::from_le_bytes([data[off], data[off + 1]]) as usize;
        let num_features = rd_u16(6);
        let acc_size = rd_u16(8);
        let hidden = rd_u16(10);
        let buckets = rd_u16(12);
        if num_features != NUM_FEATURES_V2 {
            return Err(format!("expected {NUM_FEATURES_V2} features, file has {num_features}"));
        }
        let mut off = 16;
        let mut take = |n: usize| -> Result<Vec<f32>, String> {
            if off + n * 4 > data.len() {
                return Err(format!("NNU2 truncated: need {} bytes, have {}", off + n * 4, data.len()));
            }
            let v = (0..n)
                .map(|k| {
                    let p = off + k * 4;
                    f32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]])
                })
                .collect();
            off += n * 4;
            Ok(v)
        };
        let v2_fc1_w = take(num_features * acc_size)?;
        let v2_fc1_b = take(acc_size)?;
        let mut v2_fc2_w = Vec::new();
        let mut v2_fc2_b = Vec::new();
        let mut v2_fc3_w = Vec::new();
        let mut v2_fc3_b = Vec::new();
        for _ in 0..buckets {
            v2_fc2_w.extend(take(acc_size * hidden)?);
            v2_fc2_b.extend(take(hidden)?);
            v2_fc3_w.extend(take(hidden)?);
            v2_fc3_b.extend(take(1)?);
        }
        Ok(Self {
            input_size: num_features,
            hidden1: acc_size,
            hidden2: hidden,
            nnu2_version: 2,
            v2: true, acc_size, buckets,
            v2_fc1_w, v2_fc1_b, v2_fc2_w, v2_fc2_b, v2_fc3_w, v2_fc3_b,
            ..Default::default()
        })
    }

    /// NNU2 version 3: i16 weights with explicit per-tensor f32 scale factors
    /// (quantised = round(real * scale)), for an integer forward pass (see
    /// `logit_v3`). Header layout through `buckets`/pad is identical to version 2;
    /// version 3 then inserts a scale-factor section before the (now i16) weight
    /// payload. See `NnueNetwork::export_v3` for how the scales are chosen and the
    /// exact byte layout, which this must mirror field-for-field.
    fn load_nnu2_i16(data: &[u8]) -> Result<Self, String> {
        let rd_u16 = |off: usize| u16::from_le_bytes([data[off], data[off + 1]]) as usize;
        let num_features = rd_u16(6);
        let acc_size = rd_u16(8);
        let hidden = rd_u16(10);
        let buckets = rd_u16(12);
        if num_features != NUM_FEATURES_V2 {
            return Err(format!("expected {NUM_FEATURES_V2} features, file has {num_features}"));
        }
        if acc_size != ACC_SIZE_V2 || hidden != HIDDEN_V2 {
            return Err(format!(
                "NNU2 v3 expects acc_size={ACC_SIZE_V2} hidden={HIDDEN_V2}, file has {acc_size}/{hidden}"
            ));
        }

        let mut off = 16;
        let mut take_f32 = |n: usize| -> Result<Vec<f32>, String> {
            if off + n * 4 > data.len() {
                return Err(format!("NNU2 v3 truncated (scale section): need {} bytes, have {}", off + n * 4, data.len()));
            }
            let v = (0..n)
                .map(|k| {
                    let p = off + k * 4;
                    f32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]])
                })
                .collect();
            off += n * 4;
            Ok(v)
        };
        let scale_l1 = take_f32(1)?[0];
        let mut scale_fc2 = Vec::with_capacity(buckets);
        let mut scale_fc3 = Vec::with_capacity(buckets);
        for _ in 0..buckets {
            scale_fc2.push(take_f32(1)?[0]);
            scale_fc3.push(take_f32(1)?[0]);
        }

        let mut take_i16 = |n: usize| -> Result<Vec<i16>, String> {
            if off + n * 2 > data.len() {
                return Err(format!("NNU2 v3 truncated (weights): need {} bytes, have {}", off + n * 2, data.len()));
            }
            let v = (0..n)
                .map(|k| {
                    let p = off + k * 2;
                    i16::from_le_bytes([data[p], data[p + 1]])
                })
                .collect();
            off += n * 2;
            Ok(v)
        };
        let v3_fc1_w = take_i16(num_features * acc_size)?;
        let v3_fc1_b = take_i16(acc_size)?;
        let mut v3_fc2_w = Vec::new();
        let mut v3_fc2_b = Vec::new();
        let mut v3_fc3_w = Vec::new();
        let mut v3_fc3_b = Vec::new();
        for _ in 0..buckets {
            v3_fc2_w.extend(take_i16(acc_size * hidden)?);
            v3_fc2_b.extend(take_i16(hidden)?);
            v3_fc3_w.extend(take_i16(hidden)?);
            v3_fc3_b.extend(take_i16(1)?);
        }
        Ok(Self {
            input_size: num_features,
            hidden1: acc_size,
            hidden2: hidden,
            nnu2_version: 3,
            v2: true, acc_size, buckets,
            v3_fc1_w, v3_fc1_b, v3_fc2_w, v3_fc2_b, v3_fc3_w, v3_fc3_b,
            v3_scale_l1: scale_l1, v3_scale_fc2: scale_fc2, v3_scale_fc3: scale_fc3,
            ..Default::default()
        })
    }

    /// Win-probability logit for the side to move.
    ///
    /// Meaningless (returns 0.0) when this net was loaded from a legacy weight file —
    /// there is no v2 accumulator/bucket data to read.
    pub fn logit_v2(&self, board: &Board) -> f32 {
        if !self.v2 {
            return 0.0;
        }
        if self.nnu2_version == 3 {
            return self.logit_v3(board);
        }
        let acc_n = self.acc_size;
        let hidden = self.hidden2;
        let mut acc = self.v2_fc1_b.clone();
        for f in build_features_v2(board) {
            let base = f as usize * acc_n;
            for j in 0..acc_n {
                acc[j] += self.v2_fc1_w[base + j];
            }
        }
        for a in acc.iter_mut() {
            *a = a.max(0.0);
        }
        let b = phase_bucket(board).min(self.buckets - 1);
        let w2 = &self.v2_fc2_w[b * hidden * acc_n..(b + 1) * hidden * acc_n];
        let b2 = &self.v2_fc2_b[b * hidden..(b + 1) * hidden];
        let w3 = &self.v2_fc3_w[b * hidden..(b + 1) * hidden];
        let mut out = self.v2_fc3_b[b];
        for j in 0..hidden {
            let row = &w2[j * acc_n..(j + 1) * acc_n];
            let mut h = b2[j];
            for i in 0..acc_n {
                h += row[i] * acc[i];
            }
            if h > 0.0 {
                out += w3[j] * h;
            }
        }
        out
    }

    /// NNU2 version 3 forward pass: same math as `logit_v2`'s f32 path (embedding-sum
    /// first layer, ReLU, bucket-selected 2-layer head), but in integer arithmetic on
    /// the i16-quantised weights, using fixed-size stack buffers so this hot per-node
    /// call makes zero heap allocations (the f32 path above still clones a Vec and
    /// allocates a features Vec every call — left alone deliberately, see nnue.rs
    /// module docs / task-9-report.md: it must stay the unoptimised f32 baseline).
    ///
    /// Accumulation widths: the first layer (23 additions of i16 values into the
    /// per-neuron sum) fits comfortably in i32, as instructed. The two dot products
    /// after it (1024-wide and 32-wide) do NOT: a single i16*i32 product from the
    /// first dot product can already exceed i32::MAX for this net's real scale
    /// factors (verified empirically — see task-9-report.md), so those two
    /// reductions accumulate in i64. (An i16-accumulator variant was tried — capping
    /// `scale_l1` so the accumulator itself fits in i16, making the dot product
    /// i16*i16 instead of i16*i32 — on the theory that a narrower operand type
    /// vectorises better; measured effect was ~0 (within noise) both with and
    /// without AVX2, while fidelity margin got measurably worse (worst case
    /// 0.0004 -> 0.0038), so it was reverted. See task-9-report.md.) The final
    /// division back to a float logit is done in f64 before the (now small, O(1))
    /// result is narrowed to f32 — narrowing the raw i64 accumulator to f32 first
    /// would lose ~7 decimal digits of precision on values that can reach
    /// ~1e13-1e14 and silently blow the 0.02 fidelity budget.
    fn logit_v3(&self, board: &Board) -> f32 {
        let acc_n = self.acc_size;
        let hidden = self.hidden2;
        debug_assert_eq!(acc_n, ACC_SIZE_V2);
        debug_assert_eq!(hidden, HIDDEN_V2);

        let mut acc = [0i32; ACC_SIZE_V2];
        for (a, &b) in acc.iter_mut().zip(self.v3_fc1_b.iter()) {
            *a = b as i32;
        }
        for f in build_features_v2_arr(board) {
            let base = f as usize * acc_n;
            let row = &self.v3_fc1_w[base..base + acc_n];
            // Iterator form (vs. `for j in 0..acc_n { acc[j] += row[j] }`) is the more
            // idiomatic way to let LLVM elide bounds checks; measured effect on this
            // hot loop was small (~1-2%, see task-9-report.md) — kept for the (untested
            // here) codegen on other targets/compiler versions, not because it moved
            // the NPS numbers reported for this task.
            for (a, &w) in acc.iter_mut().zip(row.iter()) {
                *a += w as i32;
            }
        }
        for a in acc.iter_mut() {
            if *a < 0 {
                *a = 0;
            }
        }

        let b = phase_bucket(board).min(self.buckets - 1);
        let s1 = self.v3_scale_l1;
        let s2 = self.v3_scale_fc2[b];
        let s3 = self.v3_scale_fc3[b];
        let s1_i64 = s1 as i64;

        let w2 = &self.v3_fc2_w[b * hidden * acc_n..(b + 1) * hidden * acc_n];
        let b2 = &self.v3_fc2_b[b * hidden..(b + 1) * hidden];
        let w3 = &self.v3_fc3_w[b * hidden..(b + 1) * hidden];
        let b3 = self.v3_fc3_b[b];

        // h[j] accumulates at combined scale (s1 * s2): fc2_b is only scaled by s2, so
        // its contribution is multiplied up by s1 to match the fc2_w[i]*acc[i] terms
        // (each already at scale s1*s2, since acc[i] carries s1 and fc2_w carries s2).
        let mut h = [0i64; HIDDEN_V2];
        for j in 0..hidden {
            let row = &w2[j * acc_n..(j + 1) * acc_n];
            let mut sum: i64 = b2[j] as i64 * s1_i64;
            for (&w, &a) in row.iter().zip(acc.iter()) {
                sum += w as i64 * a as i64;
            }
            h[j] = sum.max(0);
        }

        // out accumulates at combined scale (s1 * s2 * s3); fc3_b is scaled by s3
        // only, so it is multiplied up by (s1 * s2) to match the fc3_w[j]*h[j] terms.
        let mut out: i64 = b3 as i64 * s1_i64 * s2 as i64;
        for (&w, &hv) in w3.iter().zip(h.iter()) {
            out += w as i64 * hv;
        }
        let denom = s1 as f64 * s2 as f64 * s3 as f64;
        (out as f64 / denom) as f32
    }

    /// Quantise this network's weights (must be an f32 NNU2 / version 2 net) into the
    /// version-3 (i16) byte layout that `load_nnu2_i16` reads. One scale factor per
    /// tensor group, chosen as `floor(32767 / observed_absmax * 0.999)`:
    /// - fc1_w and fc1_b share a scale (`v3_scale_l1`) because they are summed into
    ///   the same accumulator before anything else happens to them.
    /// - each bucket's fc2_w/fc2_b share a scale, and each bucket's fc3_w/fc3_b share
    ///   a (different) scale, because bucket weight magnitudes differ a lot (this
    ///   net's fc2 absmax ranges ~0.048-0.34 across buckets — sharing one scale
    ///   across buckets would waste most of int16's range on the smaller ones).
    ///
    /// The 0.999 margin exists so no weight's rounded quantised value can land
    /// exactly on the int16 boundary and overflow; `quantise_one` below still checks
    /// and returns an Err rather than silently clamping if it ever does.
    ///
    /// This is also exposed as the Rust-side reference implementation the Python
    /// exporter's `export_nnu2_v3` mirrors — see research/training/train_nnue_v2.py.
    pub fn export_v3(&self) -> Result<Vec<u8>, String> {
        if !self.v2 || self.nnu2_version != 2 {
            return Err("export_v3 requires an f32 NNU2 (version 2) network".into());
        }
        let acc_n = self.acc_size;
        let hidden = self.hidden2;
        let buckets = self.buckets;
        let num_features = self.input_size;

        let scale_l1 = pick_scale(self.v2_fc1_w.iter().chain(self.v2_fc1_b.iter()).copied());
        let mut scale_fc2 = Vec::with_capacity(buckets);
        let mut scale_fc3 = Vec::with_capacity(buckets);
        for b in 0..buckets {
            let w2 = &self.v2_fc2_w[b * hidden * acc_n..(b + 1) * hidden * acc_n];
            let bias2 = &self.v2_fc2_b[b * hidden..(b + 1) * hidden];
            scale_fc2.push(pick_scale(w2.iter().chain(bias2.iter()).copied()));
            let w3 = &self.v2_fc3_w[b * hidden..(b + 1) * hidden];
            let bias3 = &self.v2_fc3_b[b..b + 1];
            scale_fc3.push(pick_scale(w3.iter().chain(bias3.iter()).copied()));
        }

        // Multiply in f64 (matching the Python exporter's np.float64 promotion) rather
        // than f32: a handful of values per net land close enough to a rounding
        // boundary (x.4999995 vs x.5000005) that f32-vs-f64 multiply precision flips
        // the rounded integer by 1 — harmless for fidelity either way (sub-ULP), but
        // computing it the same way in both places means the two independent
        // quantisers (this one and export_nnu2_v3 in train_nnue_v2.py) produce
        // byte-identical output from the same source weights, which is worth having
        // as a cross-check.
        let quantise_one = |v: f32, s: f32| -> Result<i16, String> {
            let scaled = (v as f64 * s as f64).round();
            if scaled.abs() > 32767.0 {
                return Err(format!("quantisation overflow: {v} * {s} = {scaled}"));
            }
            Ok(scaled as i16)
        };

        let mut out = Vec::with_capacity(
            16 + 4 + buckets * 8 + (num_features * acc_n + acc_n) * 2
                + buckets * (acc_n * hidden + hidden + hidden + 1) * 2,
        );
        out.extend_from_slice(&0x324E554Eu32.to_le_bytes());
        for v in [3u16, num_features as u16, acc_n as u16, hidden as u16, buckets as u16, 0u16] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&scale_l1.to_le_bytes());
        for b in 0..buckets {
            out.extend_from_slice(&scale_fc2[b].to_le_bytes());
            out.extend_from_slice(&scale_fc3[b].to_le_bytes());
        }
        for &v in &self.v2_fc1_w {
            out.extend_from_slice(&quantise_one(v, scale_l1)?.to_le_bytes());
        }
        for &v in &self.v2_fc1_b {
            out.extend_from_slice(&quantise_one(v, scale_l1)?.to_le_bytes());
        }
        for b in 0..buckets {
            let w2 = &self.v2_fc2_w[b * hidden * acc_n..(b + 1) * hidden * acc_n];
            let bias2 = &self.v2_fc2_b[b * hidden..(b + 1) * hidden];
            let w3 = &self.v2_fc3_w[b * hidden..(b + 1) * hidden];
            let bias3 = self.v2_fc3_b[b];
            for &v in w2 {
                out.extend_from_slice(&quantise_one(v, scale_fc2[b])?.to_le_bytes());
            }
            for &v in bias2 {
                out.extend_from_slice(&quantise_one(v, scale_fc2[b])?.to_le_bytes());
            }
            for &v in w3 {
                out.extend_from_slice(&quantise_one(v, scale_fc3[b])?.to_le_bytes());
            }
            out.extend_from_slice(&quantise_one(bias3, scale_fc3[b])?.to_le_bytes());
        }
        Ok(out)
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
        if self.v2 {
            let cp = (350.0 * self.logit_v2(board)).clamp(-3000.0, 3000.0);
            return (cp * 64.0) as i32;
        }
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
// The trained architecture's dimensions (research/training/train_nnue_v2.py: ACC=1024,
// HIDDEN=32). NNU2 v3 hard-codes these as fixed-size stack buffers in `logit_v3` to
// avoid a heap allocation on every node visited during search; `load_nnu2_i16` checks
// the file's header against them and errors instead of silently truncating/panicking
// if a future net changes these dimensions.
const ACC_SIZE_V2: usize = 1024;
const HIDDEN_V2: usize = 32;

/// `floor(32767 / absmax(vals) * 0.999)`: the largest per-tensor integer scale that
/// keeps every quantised value inside i16 range with a small margin, so no value's
/// rounded quantisation can land exactly on the boundary and overflow. Falls back to
/// 1.0 for an all-zero tensor (nothing to scale, and it avoids a divide-by-zero).
///
/// This is the Rust reference the Python exporter's `_pick_scale` mirrors — see
/// research/training/train_nnue_v2.py::export_nnu2_v3. Keep both in lockstep.
fn pick_scale(vals: impl Iterator<Item = f32>) -> f32 {
    let absmax = vals.fold(0.0f32, |m, v| m.max(v.abs()));
    if absmax <= 0.0 {
        return 1.0;
    }
    (32767.0 / absmax * 0.999).floor()
}

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
    build_features_v2_arr(board).to_vec()
}

/// Same 23 active features as `build_features_v2`, in the same order, but written
/// into a fixed-size stack array instead of an allocated `Vec` — this is the one and
/// only place the feature layout is computed; `build_features_v2` just copies it out
/// into a Vec for callers that want one. Used directly by the v3 (i16) forward pass,
/// which runs per node visited in search and must not allocate.
#[inline]
fn build_features_v2_arr(board: &Board) -> [u16; ACTIVE_FEATURES_V2] {
    let me = board.side_to_move.index();
    let opp = 1 - me;
    let mut f = [0u16; ACTIVE_FEATURES_V2];
    let mut n = 0;
    for i in 0..NUM_PITS {
        f[n] = (i * 14 + count_bucket(board.pits[me][i])) as u16;
        n += 1;
    }
    for i in 0..NUM_PITS {
        f[n] = (126 + i * 14 + count_bucket(board.pits[opp][i])) as u16;
        n += 1;
    }
    f[n] = (252 + (board.kazan[me] as usize / 10).min(8)) as u16;
    n += 1;
    f[n] = (261 + (board.kazan[opp] as usize / 10).min(8)) as u16;
    n += 1;
    let tuz = |t: i8| if t >= 0 { t as usize } else { 9 };
    f[n] = (270 + tuz(board.tuzdyk[me])) as u16;
    n += 1;
    f[n] = (280 + tuz(board.tuzdyk[opp])) as u16;
    n += 1;
    f[n] = (290 + (board_stones(board) % 2) as usize) as u16;
    n += 1;
    debug_assert_eq!(n, ACTIVE_FEATURES_V2);
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
    use crate::board::{Board, parse_position};

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

    /// Build a minimal v2 file in memory: one feature contributes 1.0 to accumulator 0,
    /// bucket 0 reads accumulator 0 with weight 1.0, everything else is zero. Then the
    /// logit for the start position is exactly the number of active features that map to
    /// accumulator 0 — a value we can compute by hand.
    fn synthetic_v2(num_features: usize, acc: usize, hidden: usize, buckets: usize) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0x324E554Eu32.to_le_bytes());
        for v in [2u16, num_features as u16, acc as u16, hidden as u16, buckets as u16, 0u16] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let mut push_f32 = |out: &mut Vec<u8>, v: f32| out.extend_from_slice(&v.to_le_bytes());
        for f in 0..num_features {
            for j in 0..acc {
                // feature 9 (me pit 0 holding 9 stones) is the only one that fires
                push_f32(&mut out, if f == 9 && j == 0 { 1.0 } else { 0.0 });
            }
        }
        for _ in 0..acc {
            push_f32(&mut out, 0.0);
        }
        for b in 0..buckets {
            for j in 0..hidden {
                for i in 0..acc {
                    push_f32(&mut out, if b == 0 && j == 0 && i == 0 { 1.0 } else { 0.0 });
                }
            }
            for _ in 0..hidden {
                push_f32(&mut out, 0.0);
            }
            for j in 0..hidden {
                push_f32(&mut out, if b == 0 && j == 0 { 1.0 } else { 0.0 });
            }
            push_f32(&mut out, 0.0);
        }
        out
    }

    #[test]
    fn loads_v2_and_evaluates_by_hand() {
        let bytes = synthetic_v2(NUM_FEATURES_V2, 8, 4, NUM_BUCKETS_V2);
        let path = std::env::temp_dir().join("nnue_v2_synthetic.bin");
        std::fs::write(&path, &bytes).unwrap();
        let net = NnueNetwork::load(path.to_str().unwrap()).expect("v2 file loads");
        let b = Board::new();
        // start position fires feature 9 once -> acc0 = 1.0 -> hidden0 = 1.0 -> logit = 1.0
        let cp = net.evaluate(&b) / 64;
        assert_eq!(cp, 350, "logit 1.0 must map to 350 cp, got {cp}");
    }

    #[test]
    fn legacy_weights_still_load() {
        // cargo test runs with CWD = the crate manifest dir (engine/), not the workspace
        // root, so the path is relative to that.
        let net = NnueNetwork::load("../models/engine/nnue_weights.bin")
            .expect("the shipped legacy weights must keep loading");
        let cp = net.evaluate(&Board::new());
        assert!(cp.abs() < 100_000, "legacy eval returns a sane number, got {cp}");
    }

    #[test]
    fn truncated_v2_header_errors_instead_of_panicking() {
        // Only magic (4 bytes) + version (2 bytes), padded to 8 bytes total — well short
        // of the 16-byte header (num_features/acc_size/hidden/buckets/pad are missing). A
        // half-written export (e.g. process killed mid-write) must be reported as a load
        // error, not crash the engine process.
        let mut out = Vec::new();
        out.extend_from_slice(&0x324E554Eu32.to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        assert_eq!(out.len(), 8, "sanity: magic(4) + version(2) + 2 more bytes = 8");
        let path = std::env::temp_dir().join("nnue_v2_truncated.bin");
        std::fs::write(&path, &out).unwrap();
        let result = NnueNetwork::load(path.to_str().unwrap());
        assert!(result.is_err(), "truncated NNU2 header must return Err, not panic");
    }

    /// Same idea as `synthetic_v2`, but for version 3: real-sized (1024 acc / 32
    /// hidden / 4 buckets) since `load_nnu2_i16` rejects any other shape, all-1.0
    /// scale factors (so the hand-computed expected value is the same as
    /// `synthetic_v2`'s: an all-integer, all-zero-except-one-path network), and i16
    /// weights instead of f32. Feature 9 (me pit 0 holding 9 stones) contributes 1 to
    /// accumulator 0; bucket 0 reads accumulator 0 with weight 1; everything else is
    /// zero, so the start position's logit is exactly 1.0, same as the v2 fixture.
    fn synthetic_v3(num_features: usize, acc: usize, hidden: usize, buckets: usize) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0x324E554Eu32.to_le_bytes());
        for v in [3u16, num_features as u16, acc as u16, hidden as u16, buckets as u16, 0u16] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        // scale section: scale_l1, then (scale_fc2[b], scale_fc3[b]) per bucket — all 1.0
        out.extend_from_slice(&1.0f32.to_le_bytes());
        for _ in 0..buckets {
            out.extend_from_slice(&1.0f32.to_le_bytes());
            out.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let mut push_i16 = |out: &mut Vec<u8>, v: i16| out.extend_from_slice(&v.to_le_bytes());
        for f in 0..num_features {
            for j in 0..acc {
                push_i16(&mut out, if f == 9 && j == 0 { 1 } else { 0 });
            }
        }
        for _ in 0..acc {
            push_i16(&mut out, 0);
        }
        for b in 0..buckets {
            for j in 0..hidden {
                for i in 0..acc {
                    push_i16(&mut out, if b == 0 && j == 0 && i == 0 { 1 } else { 0 });
                }
            }
            for _ in 0..hidden {
                push_i16(&mut out, 0);
            }
            for j in 0..hidden {
                push_i16(&mut out, if b == 0 && j == 0 { 1 } else { 0 });
            }
            push_i16(&mut out, 0);
        }
        out
    }

    #[test]
    fn loads_v3_and_evaluates_by_hand() {
        let bytes = synthetic_v3(NUM_FEATURES_V2, ACC_SIZE_V2, HIDDEN_V2, NUM_BUCKETS_V2);
        let path = std::env::temp_dir().join("nnue_v3_synthetic.bin");
        std::fs::write(&path, &bytes).unwrap();
        let net = NnueNetwork::load(path.to_str().unwrap()).expect("v3 file loads");
        let b = Board::new();
        let cp = net.evaluate(&b) / 64;
        assert_eq!(cp, 350, "logit 1.0 must map to 350 cp under v3 too, got {cp}");
    }

    #[test]
    fn truncated_v3_scale_section_errors_instead_of_panicking() {
        // A well-formed 16-byte version-3 header but nothing after it: the scale
        // section (4 + buckets*8 bytes) and the entire weight payload are missing.
        // Must be a load error, not a panic or an out-of-bounds read.
        let mut out = Vec::new();
        out.extend_from_slice(&0x324E554Eu32.to_le_bytes());
        for v in [3u16, NUM_FEATURES_V2 as u16, ACC_SIZE_V2 as u16, HIDDEN_V2 as u16, NUM_BUCKETS_V2 as u16, 0u16] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(out.len(), 16);
        let path = std::env::temp_dir().join("nnue_v3_truncated.bin");
        std::fs::write(&path, &out).unwrap();
        let result = NnueNetwork::load(path.to_str().unwrap());
        assert!(result.is_err(), "truncated NNU2 v3 scale section must return Err, not panic");
    }

    #[test]
    fn v3_rejects_mismatched_dimensions() {
        // logit_v3 uses fixed-size [T; ACC_SIZE_V2]/[T; HIDDEN_V2] stack buffers, so a
        // file claiming a different acc/hidden size must be rejected at load time,
        // not silently misread (or worse, accepted and then overrun the buffers).
        let bytes = synthetic_v3(NUM_FEATURES_V2, 8, 4, NUM_BUCKETS_V2);
        let path = std::env::temp_dir().join("nnue_v3_bad_dims.bin");
        std::fs::write(&path, &bytes).unwrap();
        let result = NnueNetwork::load(path.to_str().unwrap());
        assert!(result.is_err(), "v3 file with acc=8/hidden=4 (not 1024/32) must be rejected");
    }

    /// The required fidelity gate (task 9): quantising an f32 NNU2 net to version 3
    /// must not change what it says by more than 0.02 logit on real positions.
    ///
    /// Loads the trained f32 candidate (`v2_e12.bin`, NOT committed to git — see
    /// task-9-report.md), quantises it in-process via `export_v3` (the same code
    /// path `main.rs`'s `quantize` subcommand and the Python exporter both produce
    /// independently), reloads the quantised bytes through the real `NnueNetwork::load`
    /// entry point (so this also exercises `load_nnu2_i16` end to end, not just
    /// `logit_v3` in isolation), and compares logits on 240 positions sampled from
    /// data/9qum/games/replays.jsonl.gz (see engine/src/testdata/nnue_v2_sample_positions.txt
    /// and its generation note in task-9-report.md).
    #[test]
    fn quantised_v3_matches_f32_v2_within_tolerance() {
        let f32_net = NnueNetwork::load("../models/nets/nnue_v2/v2_e12.bin").expect(
            "models/nets/nnue_v2/v2_e12.bin must exist locally (gitignored, not committed — \
             the candidate net produced by research/training/train_nnue_v2.py) for this test",
        );
        let v3_bytes = f32_net.export_v3().expect("export_v3 must succeed on a real trained net");
        let path = std::env::temp_dir().join("nnue_v2_e12_v3_fidelity.bin");
        std::fs::write(&path, &v3_bytes).unwrap();
        let v3_net = NnueNetwork::load(path.to_str().unwrap()).expect("exported v3 bytes must load");
        assert_eq!(v3_net.nnu2_version, 3, "sanity: the reloaded net took the v3 path");

        const FIXTURE: &str = include_str!("testdata/nnue_v2_sample_positions.txt");
        let positions: Vec<&str> = FIXTURE
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        assert!(
            positions.len() >= 200,
            "fixture must supply at least 200 positions, has {}",
            positions.len()
        );

        let mut worst = 0.0f32;
        let mut worst_pos = "";
        let mut sum_abs = 0.0f64;
        for &pos in &positions {
            let b = parse_position(pos).unwrap_or_else(|e| panic!("bad fixture position {pos}: {e}"));
            let want = f32_net.logit_v2(&b);
            let got = v3_net.logit_v2(&b);
            let diff = (want - got).abs();
            sum_abs += diff as f64;
            if diff > worst {
                worst = diff;
                worst_pos = pos;
            }
        }
        let mean = sum_abs / positions.len() as f64;
        println!(
            "quantised_v3_matches_f32_v2_within_tolerance: n={} worst={:.6} (at {}) mean={:.6}",
            positions.len(), worst, worst_pos, mean
        );
        assert!(
            worst <= 0.02,
            "v3 logit diverges from f32 v2 by {worst:.6} at {worst_pos} (n={}), exceeds the 0.02 budget",
            positions.len()
        );
    }
}

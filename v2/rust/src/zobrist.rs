/// Zobrist hashing for transposition table and position deduplication
///
/// Hash components:
/// - pits[2][9] with stone counts (quantized to buckets)
/// - kazan[2] values
/// - tuzdyk[2] positions
/// - side to move

use crate::board::{Board, Side, NUM_PITS};

const MAX_STONES_BUCKET: usize = 32;

pub struct ZobristKeys {
    pit_keys: [[[u64; MAX_STONES_BUCKET + 1]; NUM_PITS]; 2],
    kazan_keys: [[u64; 163]; 2],
    tuzdyk_keys: [[u64; NUM_PITS + 1]; 2],
    side_key: u64,
}

/// Simple xorshift64 PRNG with fixed seed for deterministic hashing
struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        DeterministicRng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

impl ZobristKeys {
    pub fn new() -> Self {
        let mut rng = DeterministicRng::new(0x12345678_DEADBEEF);

        let mut keys = ZobristKeys {
            pit_keys: [[[0u64; MAX_STONES_BUCKET + 1]; NUM_PITS]; 2],
            kazan_keys: [[0u64; 163]; 2],
            tuzdyk_keys: [[0u64; NUM_PITS + 1]; 2],
            side_key: rng.next_u64(),
        };

        for side in 0..2 {
            for pit in 0..NUM_PITS {
                for stones in 0..=MAX_STONES_BUCKET {
                    keys.pit_keys[side][pit][stones] = rng.next_u64();
                }
            }
            for k in 0..163 {
                keys.kazan_keys[side][k] = rng.next_u64();
            }
            for t in 0..=NUM_PITS {
                keys.tuzdyk_keys[side][t] = rng.next_u64();
            }
        }

        keys
    }

    /// Compute full hash from scratch
    pub fn hash(&self, board: &Board) -> u64 {
        let mut h: u64 = 0;

        for side in 0..2 {
            for pit in 0..NUM_PITS {
                let stones = board.pits[side][pit] as usize;
                let bucket = stones.min(MAX_STONES_BUCKET);
                h ^= self.pit_keys[side][pit][bucket];
            }
            h ^= self.kazan_keys[side][board.kazan[side] as usize];

            let tuzdyk_idx = if board.tuzdyk[side] >= 0 {
                board.tuzdyk[side] as usize + 1
            } else {
                0
            };
            h ^= self.tuzdyk_keys[side][tuzdyk_idx];
        }

        if board.side_to_move == Side::Black {
            h ^= self.side_key;
        }

        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_different_positions_different_hashes() {
        let keys = ZobristKeys::new();
        let b1 = Board::new();
        let mut b2 = Board::new();
        b2.pits[0][0] = 10;
        assert_ne!(keys.hash(&b1), keys.hash(&b2));
    }

    #[test]
    fn test_same_position_same_hash() {
        let keys = ZobristKeys::new();
        let b1 = Board::new();
        let b2 = Board::new();
        assert_eq!(keys.hash(&b1), keys.hash(&b2));
    }

    #[test]
    fn test_side_to_move_affects_hash() {
        let keys = ZobristKeys::new();
        let mut b1 = Board::new();
        let mut b2 = Board::new();
        b2.side_to_move = Side::Black;
        assert_ne!(keys.hash(&b1), keys.hash(&b2));
    }
}

/// Endgame Tablebase (EGTB) for Togyz Kumalak
///
/// Generates and probes perfect endgame solutions for positions
/// with N or fewer total stones on the board.
///
/// Key insight: total board stones never increase after a move
/// (captures/tuzdyk drain stones to kazan), enabling bottom-up solving.

use crate::board::{Board, GameResult, Side, NUM_PITS};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

/// WDL result from side-to-move perspective
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EgtbResult {
    Win = 1,
    Draw = 2,
    Loss = 3,
}

/// Packed EGTB entry: bits[7:6] = result, bits[5:0] = DTM (0-63)
#[derive(Clone, Copy, Debug)]
pub struct EgtbEntry {
    pub result: EgtbResult,
    pub dtm: u8,
}

impl EgtbEntry {
    fn pack(&self) -> u8 {
        ((self.result as u8) << 6) | (self.dtm & 0x3F)
    }

    fn unpack(v: u8) -> Option<Self> {
        let result = match v >> 6 {
            1 => EgtbResult::Win,
            2 => EgtbResult::Draw,
            3 => EgtbResult::Loss,
            _ => return None,
        };
        Some(EgtbEntry {
            result,
            dtm: v & 0x3F,
        })
    }
}

/// Encode board position into a unique u128 key
fn encode_position(board: &Board) -> u128 {
    let mut key: u128 = 0;
    let mut shift = 0;

    // 18 pits, 4 bits each (supports 0-15 stones per pit)
    for side in 0..2 {
        for pit in 0..NUM_PITS {
            key |= (board.pits[side][pit] as u128) << shift;
            shift += 4;
        }
    }
    // kazan[0], kazan[1]: 8 bits each
    key |= (board.kazan[0] as u128) << shift;
    shift += 8;
    key |= (board.kazan[1] as u128) << shift;
    shift += 8;
    // tuzdyk: encode as 0 (none) or pit+1 (1-9), 4 bits each
    let t0 = if board.tuzdyk[0] >= 0 {
        (board.tuzdyk[0] + 1) as u128
    } else {
        0
    };
    let t1 = if board.tuzdyk[1] >= 0 {
        (board.tuzdyk[1] + 1) as u128
    } else {
        0
    };
    key |= t0 << shift;
    shift += 4;
    key |= t1 << shift;
    shift += 4;
    // side_to_move: 1 bit
    key |= (board.side_to_move.index() as u128) << shift;
    key
}

/// Runtime endgame tablebase for probing during search
pub struct EndgameTablebase {
    pub max_stones: u32,
    keys: Vec<u128>,
    values: Vec<u8>,
}

const EGTB_MAGIC: &[u8; 8] = b"TKEGTB01";

impl EndgameTablebase {
    /// Load EGTB from binary file
    pub fn load(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .map_err(|e| format!("Read magic: {}", e))?;
        if &magic != EGTB_MAGIC {
            return Err("Invalid EGTB magic".to_string());
        }

        let mut buf4 = [0u8; 4];
        reader
            .read_exact(&mut buf4)
            .map_err(|e| format!("Read max_stones: {}", e))?;
        let max_stones = u32::from_le_bytes(buf4);

        reader
            .read_exact(&mut buf4)
            .map_err(|e| format!("Read num_entries: {}", e))?;
        let num_entries = u32::from_le_bytes(buf4) as usize;

        let mut keys = Vec::with_capacity(num_entries);
        let mut values = Vec::with_capacity(num_entries);

        let mut buf16 = [0u8; 16];
        let mut buf1 = [0u8; 1];
        for _ in 0..num_entries {
            reader
                .read_exact(&mut buf16)
                .map_err(|e| format!("Read key: {}", e))?;
            reader
                .read_exact(&mut buf1)
                .map_err(|e| format!("Read value: {}", e))?;
            keys.push(u128::from_le_bytes(buf16));
            values.push(buf1[0]);
        }

        Ok(EndgameTablebase {
            max_stones,
            keys,
            values,
        })
    }

    /// Probe the tablebase for a position
    #[inline]
    pub fn probe(&self, board: &Board) -> Option<EgtbEntry> {
        let total = board.total_board_stones();
        if total as u32 > self.max_stones {
            return None;
        }
        let key = encode_position(board);
        match self.keys.binary_search(&key) {
            Ok(idx) => EgtbEntry::unpack(self.values[idx]),
            Err(_) => None,
        }
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.keys.len()
    }
}

/// Save EGTB to binary file
fn save_egtb(
    path: &str,
    max_stones: u32,
    table: &HashMap<u128, EgtbEntry>,
) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("Cannot create {}: {}", path, e))?;
    let mut writer = BufWriter::new(file);

    // Only save non-terminal positions (terminal ones are handled by game_result() in search)
    let mut entries: Vec<(u128, u8)> = table
        .iter()
        .filter(|(_, v)| v.dtm > 0) // dtm=0 means terminal position, skip it
        .map(|(&k, &v)| (k, v.pack()))
        .collect();
    entries.sort_by_key(|&(k, _)| k);

    writer
        .write_all(EGTB_MAGIC)
        .map_err(|e| format!("Write: {}", e))?;
    writer
        .write_all(&max_stones.to_le_bytes())
        .map_err(|e| format!("Write: {}", e))?;
    writer
        .write_all(&(entries.len() as u32).to_le_bytes())
        .map_err(|e| format!("Write: {}", e))?;

    for &(key, value) in &entries {
        writer
            .write_all(&key.to_le_bytes())
            .map_err(|e| format!("Write: {}", e))?;
        writer
            .write_all(&[value])
            .map_err(|e| format!("Write: {}", e))?;
    }

    writer.flush().map_err(|e| format!("Flush: {}", e))?;
    Ok(())
}

/// Generate all valid positions with exactly `n` stones distributed across 18 pits
fn enumerate_pit_distributions(n: u8) -> Vec<[[u8; NUM_PITS]; 2]> {
    let mut results = Vec::new();
    let mut pits = [[0u8; NUM_PITS]; 2];
    distribute_stones(&mut results, &mut pits, n, 0, 0);
    results
}

fn distribute_stones(
    results: &mut Vec<[[u8; NUM_PITS]; 2]>,
    pits: &mut [[u8; NUM_PITS]; 2],
    remaining: u8,
    side: usize,
    pit: usize,
) {
    if side == 2 {
        // All 18 pits assigned
        if remaining == 0 {
            results.push(*pits);
        }
        return;
    }

    let next_side = if pit + 1 >= NUM_PITS { side + 1 } else { side };
    let next_pit = if pit + 1 >= NUM_PITS { 0 } else { pit + 1 };

    // Remaining pits after this one
    let pits_left = if side == 0 {
        (NUM_PITS - pit - 1) + NUM_PITS // remaining on this side + all of other side
    } else {
        NUM_PITS - pit - 1
    };

    if pits_left == 0 {
        // Last pit: must take all remaining
        pits[side][pit] = remaining;
        results.push(*pits);
        pits[side][pit] = 0;
        return;
    }

    let max_here = remaining; // can put all remaining in this pit
    for s in 0..=max_here {
        pits[side][pit] = s;
        distribute_stones(results, pits, remaining - s, next_side, next_pit);
    }
    pits[side][pit] = 0;
}

/// Generate endgame tablebase for all positions with ≤ max_stones on the board
pub fn generate_egtb(max_stones: u32, output_path: &str) {
    println!("Generating EGTB for positions with <= {} board stones", max_stones);

    let mut table: HashMap<u128, EgtbEntry> = HashMap::new();
    let mut total_positions = 0u64;
    let mut total_terminal = 0u64;

    // Solve bottom-up: N=0, 1, ..., max_stones
    for n in 0..=max_stones as u8 {
        let distributions = enumerate_pit_distributions(n);
        let total_kazan = 162u16 - n as u16;

        // Valid kazan range: both < 82 (if either >= 82, game is terminal by kazan rule)
        // k0 + k1 = total_kazan, 0 <= k0 <= min(81, total_kazan), k1 = total_kazan - k0
        // k1 < 82 means k0 > total_kazan - 82
        let k0_min = if total_kazan > 81 {
            (total_kazan - 81) as u8
        } else {
            0
        };
        let k0_max = std::cmp::min(81, total_kazan as u8);

        // Also include terminal kazan values (k0 >= 82 or k1 >= 82) for completeness
        // Actually we DO need to include k0=82+ positions so the retrograde analysis
        // can correctly label them as terminal. Let's include all valid kazan splits.
        let real_k0_min = if total_kazan > 162 { 0 } else { 0 };
        let real_k0_max = std::cmp::min(total_kazan as u8, 162);

        let mut n_positions = 0u64;
        let mut n_terminal = 0u64;
        let mut n_new_solved = 0u64;

        for dist in &distributions {
            // For kazan: iterate valid range
            // Non-terminal range: both < 82
            // Terminal range: either >= 82
            // We need terminal positions too, so iterate full range
            let k_max = std::cmp::min(real_k0_max, total_kazan as u8);
            for k0 in real_k0_min..=k_max {
                let k1 = (total_kazan as u8).wrapping_sub(k0);
                if (k0 as u16) + (k1 as u16) != total_kazan {
                    continue;
                }

                // Tuzdyk combinations
                for t0 in -1i8..8 {
                    for t1 in -1i8..8 {
                        // Constraint: can't both be same index (when both >= 0)
                        if t0 >= 0 && t1 >= 0 && t0 == t1 {
                            continue;
                        }

                        for side in [Side::White, Side::Black] {
                            let board = Board::from_parts(*dist, [k0, k1], [t0, t1], side);

                            // Validate total stones = 162
                            let actual_total = board.total_board_stones()
                                + board.kazan[0] as u16
                                + board.kazan[1] as u16;
                            if actual_total != 162 {
                                continue;
                            }

                            let key = encode_position(&board);
                            if table.contains_key(&key) {
                                continue;
                            }

                            n_positions += 1;

                            // Check if terminal
                            if let Some(result) = board.game_result() {
                                let entry = match result {
                                    GameResult::Win(winner) => {
                                        if winner == side {
                                            EgtbEntry {
                                                result: EgtbResult::Win,
                                                dtm: 0,
                                            }
                                        } else {
                                            EgtbEntry {
                                                result: EgtbResult::Loss,
                                                dtm: 0,
                                            }
                                        }
                                    }
                                    GameResult::Draw => EgtbEntry {
                                        result: EgtbResult::Draw,
                                        dtm: 0,
                                    },
                                };
                                table.insert(key, entry);
                                n_terminal += 1;
                                continue;
                            }

                            // Try to solve using already-solved successors
                            if let Some(entry) = try_solve(&board, &table) {
                                table.insert(key, entry);
                                n_new_solved += 1;
                            }
                        }
                    }
                }
            }
        }

        total_positions += n_positions;
        total_terminal += n_terminal;

        // Iterative propagation for positions where successors have same stone count
        let mut changed = true;
        let mut iterations = 0;
        while changed {
            changed = false;
            iterations += 1;

            // Collect unsolved positions with N stones
            let unsolved: Vec<(u128, Board)> = {
                let mut v = Vec::new();
                for dist in &distributions {
                    for k0 in k0_min..=k0_max {
                        let k1 = (total_kazan as u8) - k0;
                        for t0 in -1i8..8 {
                            for t1 in -1i8..8 {
                                if t0 >= 0 && t1 >= 0 && t0 == t1 {
                                    continue;
                                }
                                for side in [Side::White, Side::Black] {
                                    let board =
                                        Board::from_parts(*dist, [k0, k1], [t0, t1], side);
                                    let actual_total = board.total_board_stones()
                                        + board.kazan[0] as u16
                                        + board.kazan[1] as u16;
                                    if actual_total != 162 {
                                        continue;
                                    }
                                    if board.is_terminal() {
                                        continue;
                                    }
                                    let key = encode_position(&board);
                                    if !table.contains_key(&key) {
                                        v.push((key, board));
                                    }
                                }
                            }
                        }
                    }
                }
                v
            };

            if unsolved.is_empty() {
                break;
            }

            for (key, board) in &unsolved {
                if let Some(entry) = try_solve(board, &table) {
                    table.insert(*key, entry);
                    changed = true;
                    n_new_solved += 1;
                }
            }

            if iterations > 200 {
                // Remaining positions are draws (cycles)
                for (key, _board) in &unsolved {
                    if !table.contains_key(key) {
                        table.insert(
                            *key,
                            EgtbEntry {
                                result: EgtbResult::Draw,
                                dtm: 63,
                            },
                        );
                    }
                }
                break;
            }
        }

        let solved = n_terminal + n_new_solved;
        println!(
            "  N={}: {} positions, {} terminal, {} solved, {} iterations",
            n, n_positions, n_terminal, solved, iterations
        );
    }

    println!(
        "\nTotal: {} positions solved, {} terminal",
        table.len(),
        total_terminal
    );

    // Verify
    let mut wins = 0u64;
    let mut draws = 0u64;
    let mut losses = 0u64;
    for entry in table.values() {
        match entry.result {
            EgtbResult::Win => wins += 1,
            EgtbResult::Draw => draws += 1,
            EgtbResult::Loss => losses += 1,
        }
    }
    println!(
        "Results: {} wins, {} draws, {} losses",
        wins, draws, losses
    );

    // Save
    println!("Saving to {}...", output_path);
    save_egtb(output_path, max_stones as u32, &table).expect("Failed to save EGTB");

    let file_size = std::fs::metadata(output_path)
        .map(|m| m.len())
        .unwrap_or(0);
    println!("Done! File size: {} bytes ({:.1} MB)", file_size, file_size as f64 / 1_000_000.0);
}

/// Try to solve a position using already-solved successor positions
fn try_solve(board: &Board, table: &HashMap<u128, EgtbEntry>) -> Option<EgtbEntry> {
    let mut moves = [0usize; NUM_PITS];
    let count = board.valid_moves_array(&mut moves);

    if count == 0 {
        // No valid moves — remaining stones go to opponent (handled by game_result)
        // This shouldn't happen for non-terminal positions, but handle it
        return Some(EgtbEntry {
            result: EgtbResult::Loss,
            dtm: 0,
        });
    }

    let mut all_resolved = true;
    let mut all_win_for_opponent = true; // all successors are WIN for opponent (= LOSS for us)
    let mut best_win_dtm = u8::MAX;
    let mut worst_loss_dtm = 0u8;
    let mut has_draw = false;

    for i in 0..count {
        let mut child = *board;
        child.make_move(moves[i]);

        // Check if child is terminal
        if let Some(result) = child.game_result() {
            match result {
                GameResult::Win(winner) => {
                    if winner == board.side_to_move {
                        // We win immediately
                        best_win_dtm = std::cmp::min(best_win_dtm, 1);
                        all_win_for_opponent = false;
                    } else {
                        // Opponent wins = loss for us in child = actually this means
                        // opponent just won. From child's perspective (opponent's turn),
                        // it's a loss for them? No — game_result returns absolute winner.
                        // If winner != our side, this move leads to our loss.
                        worst_loss_dtm = std::cmp::max(worst_loss_dtm, 1);
                    }
                }
                GameResult::Draw => {
                    has_draw = true;
                    all_win_for_opponent = false;
                }
            }
            continue;
        }

        let child_key = encode_position(&child);
        if let Some(child_entry) = table.get(&child_key) {
            // Child is from opponent's perspective (sides are swapped after make_move)
            match child_entry.result {
                EgtbResult::Loss => {
                    // Opponent loses = we win!
                    let dtm = 1 + child_entry.dtm;
                    best_win_dtm = std::cmp::min(best_win_dtm, dtm);
                    all_win_for_opponent = false;
                }
                EgtbResult::Win => {
                    // Opponent wins = we lose via this move
                    let dtm = 1 + child_entry.dtm;
                    worst_loss_dtm = std::cmp::max(worst_loss_dtm, dtm);
                }
                EgtbResult::Draw => {
                    has_draw = true;
                    all_win_for_opponent = false;
                }
            }
        } else {
            // Child not yet in table
            all_resolved = false;
            all_win_for_opponent = false;
        }
    }

    // If we found a winning move, this position is a WIN
    if best_win_dtm < u8::MAX {
        return Some(EgtbEntry {
            result: EgtbResult::Win,
            dtm: std::cmp::min(best_win_dtm, 63),
        });
    }

    // If all successors resolved and all are WIN for opponent → we LOSE
    if all_resolved && all_win_for_opponent {
        return Some(EgtbEntry {
            result: EgtbResult::Loss,
            dtm: std::cmp::min(worst_loss_dtm, 63),
        });
    }

    // If all resolved and some are draws but no wins → DRAW
    if all_resolved && has_draw {
        return Some(EgtbEntry {
            result: EgtbResult::Draw,
            dtm: 63,
        });
    }

    // Not yet solvable
    None
}

/// Verify EGTB by checking consistency: each entry's result must be consistent with its successors
pub fn verify_egtb(egtb: &EndgameTablebase, num_tests: u32) {
    println!("Verifying EGTB consistency with {} entries...", num_tests);
    let mut rng = SimpleRng::new(42);
    let mut correct = 0u32;
    let mut tested = 0u32;
    let mut errors = 0u32;

    for _i in 0..std::cmp::min(num_tests, egtb.len() as u32) {
        let idx = rng.next() as usize % egtb.len();
        let key = egtb.keys[idx];
        let entry = match EgtbEntry::unpack(egtb.values[idx]) {
            Some(e) => e,
            None => continue,
        };
        let board = match decode_position(key) {
            Some(b) => b,
            None => continue,
        };
        if board.is_terminal() {
            continue;
        }

        tested += 1;
        let mut moves = [0usize; NUM_PITS];
        let count = board.valid_moves_array(&mut moves);

        let mut has_winning_move = false;
        let mut all_losing = true; // all successor are Win for opponent (= Loss for us)

        for j in 0..count {
            let mut child = board;
            child.make_move(moves[j]);

            let child_result = if let Some(result) = child.game_result() {
                match result {
                    GameResult::Win(winner) => {
                        if winner == board.side_to_move {
                            EgtbResult::Loss // from child's perspective, opponent (=us) wins
                        } else {
                            EgtbResult::Win // from child's perspective, side to move wins
                        }
                    }
                    GameResult::Draw => EgtbResult::Draw,
                }
            } else if let Some(child_entry) = egtb.probe(&child) {
                child_entry.result
            } else {
                all_losing = false;
                continue; // unknown successor
            };

            match child_result {
                EgtbResult::Loss => has_winning_move = true, // opponent loses = we can win
                EgtbResult::Win => {} // opponent wins = bad for us
                EgtbResult::Draw => all_losing = false,
            }
        }

        let consistent = match entry.result {
            EgtbResult::Win => has_winning_move,
            EgtbResult::Loss => all_losing && !has_winning_move,
            EgtbResult::Draw => !has_winning_move, // no winning move available
        };

        if consistent {
            correct += 1;
        } else {
            errors += 1;
            if errors <= 5 {
                println!(
                    "  Inconsistency #{}: EGTB says {:?} but has_win={} all_lose={}",
                    errors, entry.result, has_winning_move, all_losing
                );
            }
        }
    }

    println!(
        "Consistency: {}/{} correct ({:.1}%), {} errors",
        correct,
        tested,
        correct as f64 / tested.max(1) as f64 * 100.0,
        errors
    );
}

/// Decode a u128 key back to a Board
fn decode_position(key: u128) -> Option<Board> {
    let mut pits = [[0u8; NUM_PITS]; 2];
    let mut shift = 0;

    for side in 0..2 {
        for pit in 0..NUM_PITS {
            pits[side][pit] = ((key >> shift) & 0xF) as u8;
            shift += 4;
        }
    }

    let k0 = ((key >> shift) & 0xFF) as u8;
    shift += 8;
    let k1 = ((key >> shift) & 0xFF) as u8;
    shift += 8;

    let t0_enc = ((key >> shift) & 0xF) as i8;
    shift += 4;
    let t1_enc = ((key >> shift) & 0xF) as i8;
    shift += 4;

    let side_idx = ((key >> shift) & 1) as usize;
    let side = if side_idx == 0 {
        Side::White
    } else {
        Side::Black
    };

    let t0 = if t0_enc > 0 { t0_enc - 1 } else { -1 };
    let t1 = if t1_enc > 0 { t1_enc - 1 } else { -1 };

    Some(Board::from_parts(pits, [k0, k1], [t0, t1], side))
}

/// Play out a position using EGTB-guided moves
fn playout_with_egtb(
    start: &Board,
    egtb: &EndgameTablebase,
    max_moves: u32,
) -> EgtbResult {
    let mut board = *start;
    let starting_side = board.side_to_move;

    for _ in 0..max_moves {
        if let Some(result) = board.game_result() {
            return match result {
                GameResult::Win(winner) => {
                    if winner == starting_side {
                        EgtbResult::Win
                    } else {
                        EgtbResult::Loss
                    }
                }
                GameResult::Draw => EgtbResult::Draw,
            };
        }

        let mut moves = [0usize; NUM_PITS];
        let count = board.valid_moves_array(&mut moves);
        if count == 0 {
            return EgtbResult::Draw;
        }

        // Find best EGTB move
        let mut best_move = moves[0];
        let mut best_score = -200i32;

        for i in 0..count {
            let mut child = board;
            child.make_move(moves[i]);

            // First check if child is terminal (not stored in EGTB)
            let score = if let Some(result) = child.game_result() {
                match result {
                    GameResult::Win(winner) => {
                        if winner == board.side_to_move {
                            150 // We win immediately = best possible
                        } else {
                            -150 // Opponent wins = worst
                        }
                    }
                    GameResult::Draw => 0,
                }
            } else if let Some(entry) = egtb.probe(&child) {
                match entry.result {
                    EgtbResult::Loss => 100 - entry.dtm as i32, // Opponent loses = best for us
                    EgtbResult::Draw => 0,
                    EgtbResult::Win => -100 + entry.dtm as i32, // Opponent wins = worst for us
                }
            } else {
                -1 // Unknown
            };

            if score > best_score {
                best_score = score;
                best_move = moves[i];
            }
        }

        board.make_move(best_move);
    }

    // Max moves reached without terminal — call it a draw
    EgtbResult::Draw
}

/// Simple RNG for verification
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }
    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
}

/// Тоғызқұмалақ board representation and game logic (V2 - Gumbel AlphaZero)
///
/// Board layout (from White's perspective):
///   Black: pit[1][8] pit[1][7] ... pit[1][0]   <- Black's pits (opponent)
///   White: pit[0][0] pit[0][1] ... pit[0][8]   <- White's pits
///
/// Pits are numbered 1-9 for each player (internally 0-8).
/// Each pit starts with 9 stones. Total: 162 stones.
/// Win condition: first to capture 82+ stones in kazan.
///
/// Adapted from engine/src/board.rs (V1) with additional API for AlphaZero.

use std::fmt;

pub const NUM_PITS: usize = 9;
pub const INITIAL_STONES: u8 = 9;
pub const WIN_THRESHOLD: u8 = 82;
pub const TOTAL_STONES: u32 = 162;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    White = 0,
    Black = 1,
}

impl Side {
    #[inline]
    pub fn opposite(self) -> Side {
        match self {
            Side::White => Side::Black,
            Side::Black => Side::White,
        }
    }

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameResult {
    Win(Side),
    Draw,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Board {
    pub pits: [[u8; NUM_PITS]; 2],
    pub kazan: [u8; 2],
    pub tuzdyk: [i8; 2],       // -1 = none, 0-7 = pit index
    pub side_to_move: Side,
    pub move_count: u16,
}

impl Board {
    /// Create initial position: 9 stones in each pit
    pub fn new() -> Self {
        Board {
            pits: [[INITIAL_STONES; NUM_PITS]; 2],
            kazan: [0; 2],
            tuzdyk: [-1; 2],
            side_to_move: Side::White,
            move_count: 0,
        }
    }

    /// List of legal moves (pit indices 0-8) for the current side
    pub fn legal_moves(&self) -> Vec<usize> {
        let me = self.side_to_move.index();
        let opp = self.side_to_move.opposite().index();
        let opp_tuzdyk = self.tuzdyk[opp];
        let mut moves = Vec::with_capacity(NUM_PITS);
        for i in 0..NUM_PITS {
            if self.pits[me][i] > 0 && opp_tuzdyk != i as i8 {
                moves.push(i);
            }
        }
        moves
    }

    /// Get valid moves as a bitmask and count
    #[inline]
    pub fn valid_moves_mask(&self) -> (u16, usize) {
        let me = self.side_to_move.index();
        let opp = self.side_to_move.opposite().index();
        let opp_tuzdyk = self.tuzdyk[opp];
        let mut mask: u16 = 0;
        let mut count = 0;
        for i in 0..NUM_PITS {
            if self.pits[me][i] > 0 && opp_tuzdyk != i as i8 {
                mask |= 1 << i;
                count += 1;
            }
        }
        (mask, count)
    }

    /// Fill array with valid moves, return count
    pub fn valid_moves_array(&self, moves: &mut [usize; NUM_PITS]) -> usize {
        let me = self.side_to_move.index();
        let opp = self.side_to_move.opposite().index();
        let opp_tuzdyk = self.tuzdyk[opp];
        let mut count = 0;
        for i in 0..NUM_PITS {
            if self.pits[me][i] > 0 && opp_tuzdyk != i as i8 {
                moves[count] = i;
                count += 1;
            }
        }
        count
    }

    /// Check if a move is valid
    #[inline]
    pub fn is_valid_move(&self, pit: usize) -> bool {
        if pit >= NUM_PITS {
            return false;
        }
        let me = self.side_to_move.index();
        let opp = self.side_to_move.opposite().index();
        self.pits[me][pit] > 0 && self.tuzdyk[opp] != pit as i8
    }

    /// Make a move (pit_index 0-8). Modifies board in place.
    pub fn make_move(&mut self, pit_index: usize) {
        let me = self.side_to_move.index();
        let opp = self.side_to_move.opposite().index();

        let stones = self.pits[me][pit_index];
        debug_assert!(stones > 0, "Cannot play from empty pit");

        self.pits[me][pit_index] = 0;

        let mut current_pit = pit_index;
        let mut current_side = me;

        if stones == 1 {
            // Special case: single stone moves to next pit
            current_pit += 1;
            if current_pit > 8 {
                current_pit = 0;
                current_side = opp;
            }

            if current_side == opp && self.tuzdyk[me] == current_pit as i8 {
                // Our tuzdyk on opponent's side — collect
                self.kazan[me] += 1;
            } else if current_side == me && self.tuzdyk[opp] == current_pit as i8 {
                // Opponent's tuzdyk on our side — they collect
                self.kazan[opp] += 1;
            } else {
                self.pits[current_side][current_pit] += 1;
            }
        } else {
            // Normal: first stone back to source pit
            self.pits[current_side][current_pit] += 1;
            let mut remaining = stones - 1;

            while remaining > 0 {
                current_pit += 1;
                if current_pit > 8 {
                    current_pit = 0;
                    current_side = 1 - current_side;
                }

                if current_side == opp && self.tuzdyk[me] == current_pit as i8 {
                    self.kazan[me] += 1;
                } else if current_side == me && self.tuzdyk[opp] == current_pit as i8 {
                    self.kazan[opp] += 1;
                } else {
                    self.pits[current_side][current_pit] += 1;
                }

                remaining -= 1;
            }
        }

        // Check capture and tuzdyk (only if landed on opponent's side, not on a tuzdyk)
        let is_tuzdyk_pit = (current_side == opp && self.tuzdyk[me] == current_pit as i8)
            || (current_side == me && self.tuzdyk[opp] == current_pit as i8);

        if current_side == opp && !is_tuzdyk_pit {
            let count = self.pits[opp][current_pit];

            if count == 3 && self.can_create_tuzdyk(me, current_pit) {
                // Tuzdyk creation
                self.tuzdyk[me] = current_pit as i8;
                self.kazan[me] += count;
                self.pits[opp][current_pit] = 0;
            } else if count % 2 == 0 && count > 0 {
                // Capture even
                self.kazan[me] += count;
                self.pits[opp][current_pit] = 0;
            }
        }

        // Switch side
        self.side_to_move = self.side_to_move.opposite();
        self.move_count += 1;
    }

    /// Check if player can create tuzdyk at given opponent pit
    #[inline]
    fn can_create_tuzdyk(&self, player: usize, pit_index: usize) -> bool {
        // Already has tuzdyk
        if self.tuzdyk[player] != -1 {
            return false;
        }
        // Can't at pit 9 (index 8)
        if pit_index == 8 {
            return false;
        }
        // Can't at same position as opponent's tuzdyk
        let opponent = 1 - player;
        if self.tuzdyk[opponent] == pit_index as i8 {
            return false;
        }
        true
    }

    /// Check game result
    pub fn game_result(&self) -> Option<GameResult> {
        if self.kazan[0] >= WIN_THRESHOLD {
            return Some(GameResult::Win(Side::White));
        }
        if self.kazan[1] >= WIN_THRESHOLD {
            return Some(GameResult::Win(Side::Black));
        }

        // Check if either side is empty
        let white_empty = self.pits[0].iter().all(|&x| x == 0);
        let black_empty = self.pits[1].iter().all(|&x| x == 0);

        if white_empty || black_empty {
            if self.kazan[0] > self.kazan[1] {
                return Some(GameResult::Win(Side::White));
            } else if self.kazan[1] > self.kazan[0] {
                return Some(GameResult::Win(Side::Black));
            } else {
                return Some(GameResult::Draw);
            }
        }

        None
    }

    /// Is the game over?
    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.game_result().is_some()
    }

    /// Game outcome from White's perspective: +1 White wins, -1 Black wins, 0 draw
    pub fn outcome(&self) -> f32 {
        match self.game_result() {
            Some(GameResult::Win(Side::White)) => 1.0,
            Some(GameResult::Win(Side::Black)) => -1.0,
            Some(GameResult::Draw) => 0.0,
            None => 0.0,
        }
    }

    /// Game outcome from the perspective of the given side
    pub fn outcome_for_side(&self, side: u8) -> f32 {
        let o = self.outcome();
        if side == 0 { o } else { -o }
    }

    /// Current side to move as u8: 0 = White, 1 = Black
    #[inline]
    pub fn side_to_move_u8(&self) -> u8 {
        self.side_to_move.index() as u8
    }

    /// Stones in pits for a given side (0=White, 1=Black)
    #[inline]
    pub fn pits_for_side(&self, side: u8) -> &[u8; NUM_PITS] {
        &self.pits[side as usize]
    }

    /// Kazan (captured stones) for a given side (0=White, 1=Black)
    #[inline]
    pub fn kazan_for_side(&self, side: u8) -> u8 {
        self.kazan[side as usize]
    }

    /// Tuzdyk position for a given side (None if no tuzdyk)
    #[inline]
    pub fn tuzdyk_for_side(&self, side: u8) -> Option<u8> {
        let t = self.tuzdyk[side as usize];
        if t >= 0 { Some(t as u8) } else { None }
    }

    /// Total stones currently on the board (not in kazans)
    pub fn stones_on_board(&self) -> u32 {
        let white_board: u32 = self.pits[0].iter().map(|&x| x as u32).sum();
        let black_board: u32 = self.pits[1].iter().map(|&x| x as u32).sum();
        white_board + black_board
    }

    /// Total stones on one side (pits only, not kazan)
    #[inline]
    pub fn stones_on_side(&self, side: Side) -> u16 {
        self.pits[side.index()].iter().map(|&x| x as u16).sum()
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "==================================================")?;
        write!(f, "  Black Kazan: {}  |  Tuzdyk: ", self.kazan[1])?;
        if self.tuzdyk[1] >= 0 {
            writeln!(f, "{}", self.tuzdyk[1] + 1)?;
        } else {
            writeln!(f, "-")?;
        }
        write!(f, "  ")?;
        for i in (0..NUM_PITS).rev() {
            write!(f, "{:3}", self.pits[1][i])?;
        }
        writeln!(f)?;
        write!(f, "  ")?;
        for i in (0..NUM_PITS).rev() {
            write!(f, "{:3}", 9 - i)?;
        }
        writeln!(f)?;
        writeln!(f, "--------------------------------------------------")?;
        write!(f, "  ")?;
        for i in 0..NUM_PITS {
            write!(f, "{:3}", i + 1)?;
        }
        writeln!(f)?;
        write!(f, "  ")?;
        for i in 0..NUM_PITS {
            write!(f, "{:3}", self.pits[0][i])?;
        }
        writeln!(f)?;
        write!(f, "  White Kazan: {}  |  Tuzdyk: ", self.kazan[0])?;
        if self.tuzdyk[0] >= 0 {
            writeln!(f, "{}", self.tuzdyk[0] + 1)?;
        } else {
            writeln!(f, "-")?;
        }
        let side_str = match self.side_to_move {
            Side::White => "White",
            Side::Black => "Black",
        };
        writeln!(f, "  Move: {} ({})", self.move_count + 1, side_str)?;
        write!(f, "==================================================")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_position() {
        let board = Board::new();
        assert_eq!(board.legal_moves().len(), 9);
        assert_eq!(board.stones_on_board(), 162);
        assert_eq!(board.kazan_for_side(0), 0);
        assert_eq!(board.kazan_for_side(1), 0);
        assert_eq!(board.side_to_move, Side::White);
        assert_eq!(board.tuzdyk_for_side(0), None);
        assert_eq!(board.tuzdyk_for_side(1), None);
        assert!(!board.is_terminal());
    }

    #[test]
    fn test_initial_board_symmetry() {
        let b = Board::new();
        assert_eq!(b.pits[0], [9; 9]);
        assert_eq!(b.pits[1], [9; 9]);
    }

    #[test]
    fn test_valid_moves_initial() {
        let b = Board::new();
        let moves = b.legal_moves();
        assert_eq!(moves.len(), 9);
        for i in 0..9 {
            assert_eq!(moves[i], i);
        }
    }

    #[test]
    fn test_sowing_single_stone() {
        // If pit has 1 stone, it moves to the next pit
        let mut b = Board::new();
        b.pits[0] = [1, 0, 0, 0, 0, 0, 0, 0, 0];
        b.pits[1] = [9, 9, 9, 9, 9, 9, 9, 9, 9];
        b.kazan = [71, 0];

        b.make_move(0);
        assert_eq!(b.pits[0][0], 0);
        assert_eq!(b.pits[0][1], 1); // stone moved to next pit
    }

    #[test]
    fn test_sowing_single_stone_wraps() {
        // Single stone from pit 9 (index 8) wraps to opponent pit 1 (index 0)
        let mut b = Board::new();
        b.pits[0] = [0, 0, 0, 0, 0, 0, 0, 0, 1];
        b.pits[1] = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        b.kazan = [80, 81];

        b.make_move(8);
        assert_eq!(b.pits[0][8], 0);
        assert_eq!(b.pits[1][0], 1); // wrapped to opponent side
    }

    #[test]
    fn test_normal_sowing() {
        // White plays pit 1 (index 0): 9 stones
        // First stone back to pit 1, then 8 stones to pits 2-9
        let mut b = Board::new();
        b.make_move(0);

        assert_eq!(b.pits[0][0], 1); // first stone back to source
        for i in 1..NUM_PITS {
            assert_eq!(b.pits[0][i], 10); // one extra stone each
        }
    }

    #[test]
    fn test_capture_even() {
        // White plays pit 7 (index 6): 9 stones
        // First stone back to pit 7, then 8 stones to pit 8, 9, opp 1-6
        // Last stone lands on opponent pit 6 (index 5) which had 9+1=10 (even) -> capture
        let mut b = Board::new();
        b.make_move(6);

        assert_eq!(b.kazan[0], 10); // White captured 10 stones
        assert_eq!(b.pits[1][5], 0); // opponent pit 6 emptied
        assert_eq!(b.pits[0][6], 1); // source pit has 1 (first stone back)
    }

    #[test]
    fn test_no_capture_odd() {
        // Set up a position where landing creates odd count (no capture)
        let mut b = Board::new();
        b.pits[0] = [0, 0, 0, 0, 0, 0, 0, 0, 2];
        b.pits[1] = [4, 0, 0, 0, 0, 0, 0, 0, 0]; // opponent pit 1 has 4

        // White plays pit 9: 2 stones, first back to pit 9, second to opp pit 1
        // Opp pit 1: 4+1=5 (odd) -> no capture
        b.kazan = [75, 75];
        b.make_move(8);

        assert_eq!(b.pits[1][0], 5); // no capture, stones remain
        assert_eq!(b.kazan[0], 75);  // no change
    }

    #[test]
    fn test_tuzdyk_creation() {
        // Set up: landing on opponent pit with exactly 3 stones creates tuzdyk
        let mut b = Board::new();
        b.pits[0] = [0, 0, 0, 0, 0, 0, 0, 0, 2];
        b.pits[1] = [2, 0, 0, 0, 0, 0, 0, 0, 0];
        b.kazan = [75, 75];

        // White plays pit 9 (index 8): 2 stones
        // First stone back to pit 9, second wraps to opponent pit 1 (index 0)
        // Opponent pit 1 now has 2+1=3 -> tuzdyk!
        b.make_move(8);

        assert_eq!(b.tuzdyk[0], 0); // White's tuzdyk at opponent pit 1
        assert_eq!(b.pits[1][0], 0); // pit emptied
        assert_eq!(b.kazan[0], 78); // captured 3 stones
    }

    #[test]
    fn test_tuzdyk_restrictions_no_pit9() {
        // Cannot create tuzdyk at pit 9 (index 8)
        let mut b = Board::new();
        b.pits[0] = [0, 0, 0, 0, 0, 0, 0, 0, 10];
        b.pits[1] = [0, 0, 0, 0, 0, 0, 0, 0, 2];
        b.kazan = [75, 75];

        // White plays pit 9 (index 8): 10 stones
        // After sowing, opponent pit 9 (index 8) gets 2+1=3
        // But tuzdyk at pit 9 is forbidden!
        b.make_move(8);

        assert_eq!(b.tuzdyk[0], -1); // no tuzdyk created
        assert_eq!(b.pits[1][8], 3); // stones remain
    }

    #[test]
    fn test_tuzdyk_restrictions_already_has_one() {
        // Cannot create a second tuzdyk
        let mut b = Board::new();
        b.tuzdyk[0] = 3; // White already has tuzdyk at opponent pit 4
        b.pits[0] = [0, 0, 0, 0, 0, 0, 0, 0, 2];
        b.pits[1] = [2, 0, 0, 0, 0, 0, 0, 0, 0];
        b.kazan = [75, 75];

        b.make_move(8);

        assert_eq!(b.tuzdyk[0], 3); // still only the old tuzdyk
        // Since 3 is odd and not tuzdyk-eligible, it stays
        assert_eq!(b.pits[1][0], 3);
    }

    #[test]
    fn test_tuzdyk_restrictions_same_number() {
        // Cannot create tuzdyk at same index as opponent's tuzdyk
        let mut b = Board::new();
        b.tuzdyk[1] = 0; // Black has tuzdyk at White's pit 1 (index 0)
        b.pits[0] = [0, 0, 0, 0, 0, 0, 0, 0, 2];
        b.pits[1] = [2, 0, 0, 0, 0, 0, 0, 0, 0];
        b.kazan = [75, 75];

        // White tries to create tuzdyk at opponent pit 1 (index 0)
        // But Black already has tuzdyk at index 0 -> forbidden
        b.make_move(8);

        assert_eq!(b.tuzdyk[0], -1); // not created
        assert_eq!(b.pits[1][0], 3); // stones remain (3 is odd, no capture)
    }

    #[test]
    fn test_tuzdyk_collects_stones() {
        // Stones landing on a tuzdyk go to the tuzdyk owner's kazan
        let mut b = Board::new();
        b.tuzdyk[0] = 2; // White's tuzdyk at opponent pit 3 (index 2)
        b.pits[0] = [9, 0, 0, 0, 0, 0, 0, 0, 0];
        b.pits[1] = [0, 0, 5, 0, 0, 0, 0, 0, 0]; // pit 3 has 5 stones (on tuzdyk)
        b.kazan = [0, 0];

        // We need a move that sows onto the tuzdyk pit
        // White plays pit 1 (index 0): 9 stones
        // First stone back to pit 1, stones 2-8 go to pits 2-8
        // Stone 9 goes to pit 9 (index 8)
        // ... hmm, this doesn't reach opponent side
        // Let me set up differently
        b.pits[0] = [0, 0, 0, 0, 0, 0, 0, 0, 4];
        // White plays pit 9 (index 8): 4 stones
        // First stone back to pit 9, then 3 wrap to opponent pit 1, 2, 3
        // Opponent pit 3 (index 2) is White's tuzdyk -> stone goes to White's kazan
        b.make_move(8);

        assert_eq!(b.kazan[0], 1); // one stone collected via tuzdyk
        assert_eq!(b.pits[1][2], 5); // tuzdyk pit stones don't change via normal sowing
        // Actually, let me reconsider - when a stone lands on a tuzdyk pit,
        // the stone goes to the kazan, not to the pit
    }

    #[test]
    fn test_game_end_82_threshold() {
        let mut b = Board::new();
        b.kazan[0] = 82;
        assert!(b.is_terminal());
        assert_eq!(b.game_result(), Some(GameResult::Win(Side::White)));
        assert_eq!(b.outcome(), 1.0);
    }

    #[test]
    fn test_game_end_black_wins() {
        let mut b = Board::new();
        b.kazan[1] = 82;
        assert!(b.is_terminal());
        assert_eq!(b.game_result(), Some(GameResult::Win(Side::Black)));
        assert_eq!(b.outcome(), -1.0);
    }

    #[test]
    fn test_game_end_empty_side() {
        let mut b = Board::new();
        b.pits[0] = [0; 9];
        b.kazan = [50, 40];
        // White side empty, remaining 162-50-40=72 stones on black side
        // White has 50, Black has 40 -> White wins
        assert!(b.is_terminal());
        assert_eq!(b.game_result(), Some(GameResult::Win(Side::White)));
    }

    #[test]
    fn test_game_end_draw() {
        let mut b = Board::new();
        b.pits[0] = [0; 9];
        b.pits[1] = [0; 9];
        b.kazan = [81, 81];
        assert!(b.is_terminal());
        assert_eq!(b.game_result(), Some(GameResult::Draw));
        assert_eq!(b.outcome(), 0.0);
    }

    #[test]
    fn test_outcome_for_side() {
        let mut b = Board::new();
        b.kazan[0] = 82; // White wins
        assert_eq!(b.outcome_for_side(0), 1.0);  // White's perspective
        assert_eq!(b.outcome_for_side(1), -1.0); // Black's perspective
    }

    #[test]
    fn test_stones_on_board() {
        let b = Board::new();
        assert_eq!(b.stones_on_board(), 162);

        let mut b2 = Board::new();
        b2.kazan = [10, 20];
        // Reduce some pit stones to maintain consistency
        b2.pits[0][0] = 0;
        b2.pits[1][0] = 0;
        let board_stones = b2.stones_on_board();
        // 162 - 9 - 9 = 144 stones on board
        assert_eq!(board_stones, 144);
    }

    #[test]
    fn test_side_to_move_alternates() {
        let mut b = Board::new();
        assert_eq!(b.side_to_move, Side::White);
        b.make_move(0);
        assert_eq!(b.side_to_move, Side::Black);
        b.make_move(0);
        assert_eq!(b.side_to_move, Side::White);
    }

    #[test]
    fn test_legal_moves_exclude_empty_pits() {
        let mut b = Board::new();
        b.pits[0] = [0, 5, 0, 3, 0, 0, 0, 0, 7];
        let moves = b.legal_moves();
        assert_eq!(moves, vec![1, 3, 8]);
    }

    #[test]
    fn test_stone_conservation() {
        // After any move, total stones (board + kazans) should be 162
        let mut b = Board::new();
        for pit in 0..9 {
            let mut test_board = b.clone();
            test_board.make_move(pit);
            let total = test_board.stones_on_board()
                + test_board.kazan[0] as u32
                + test_board.kazan[1] as u32;
            assert_eq!(total, 162, "Stone conservation violated after move {}", pit);
        }
    }

    #[test]
    fn test_full_game_simulation() {
        // Play a short sequence and verify no panics
        let mut b = Board::new();
        let moves_seq = [4, 3, 2, 7, 0, 8, 6, 1, 5, 4, 3, 2];
        for &m in &moves_seq {
            if b.is_terminal() {
                break;
            }
            let legal = b.legal_moves();
            if legal.contains(&m) {
                b.make_move(m);
            } else if !legal.is_empty() {
                b.make_move(legal[0]);
            }
        }
        // Verify stone conservation
        let total = b.stones_on_board() + b.kazan[0] as u32 + b.kazan[1] as u32;
        assert_eq!(total, 162);
    }
}

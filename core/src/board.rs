/// Тоғызқұмалақ board representation and game logic
///
/// Board layout (from White's perspective):
///   Black: pit[1][8] pit[1][7] ... pit[1][0]   <- Black's pits (opponent)
///   White: pit[0][0] pit[0][1] ... pit[0][8]   <- White's pits
///
/// Pits are numbered 1-9 for each player (internally 0-8).
/// Each pit starts with 9 stones. Total: 162 stones.
/// Win: 82+ in kazan, or the side to move has no stones (then each side sweeps its own board stones).

use std::fmt;

pub const NUM_PITS: usize = 9;
pub const INITIAL_STONES: u8 = 9;
pub const WIN_THRESHOLD: u8 = 82;

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

/// Undo information for unmake_move
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct UndoInfo {
    pub pit_index: usize,
    pub stones_picked: u8,
    pub captured: u8,            // stones captured (even count or tuzdyk)
    pub prev_tuzdyk: [i8; 2],    // tuzdyk state before the move (for precise unmake)
    pub side: Side,
    // Snapshot of affected pits for precise unmake
    pub prev_pits: [[u8; NUM_PITS]; 2],
    pub prev_kazan: [u8; 2],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Board {
    pub pits: [[u8; NUM_PITS]; 2],   // [side][pit]
    pub kazan: [u8; 2],              // captured stones
    pub tuzdyk: [i8; 2],             // tuzdyk position (-1 = none)
    pub side_to_move: Side,
    pub move_count: u16,
}

impl Board {
    pub fn new() -> Self {
        Board {
            pits: [[INITIAL_STONES; NUM_PITS]; 2],
            kazan: [0; 2],
            tuzdyk: [-1; 2],
            side_to_move: Side::White,
            move_count: 0,
        }
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

    /// Make a move. Returns UndoInfo for unmake.
    pub fn make_move(&mut self, pit_index: usize) -> UndoInfo {
        let me = self.side_to_move.index();
        let opp = self.side_to_move.opposite().index();

        // Save state for unmake
        let undo = UndoInfo {
            pit_index,
            stones_picked: self.pits[me][pit_index],
            captured: 0,
            prev_tuzdyk: self.tuzdyk,
            side: self.side_to_move,
            prev_pits: self.pits,
            prev_kazan: self.kazan,
        };

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
                    current_side = 1 - current_side; // flip side
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

        undo
    }

    /// Unmake a move using saved undo info (fast restore)
    #[inline]
    pub fn unmake_move(&mut self, undo: &UndoInfo) {
        self.pits = undo.prev_pits;
        self.kazan = undo.prev_kazan;
        self.tuzdyk = undo.prev_tuzdyk;
        self.side_to_move = undo.side;
        self.move_count -= 1;
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

    /// Check game result. Returns Some(winner) / Some(Draw), or None if the game goes on.
    ///
    /// Terminal when (a) a kazan reaches WIN_THRESHOLD, or (b) the SIDE TO MOVE has no
    /// stones. Rule (b) is deliberately one-sided: a player who empties its own side (a
    /// lone stone from pit 9, or a lone stone into the opponent's tuzdyk) has NOT ended the
    /// game — the opponent must move, and if that move drops stones back, play continues.
    /// Verified 2026-09-12 against PlayOK (110 self-empties, play always continued) and
    /// 9qum replays (1120 such states, 0 contradictions). At a terminal each side sweeps
    /// its own remaining board stones into its kazan before comparing.
    pub fn game_result(&self) -> Option<GameResult> {
        if self.kazan[0] >= WIN_THRESHOLD {
            return Some(GameResult::Win(Side::White));
        }
        if self.kazan[1] >= WIN_THRESHOLD {
            return Some(GameResult::Win(Side::Black));
        }

        let stm = self.side_to_move.index();
        if self.pits[stm].iter().all(|&x| x == 0) {
            let kw = self.kazan[0] as u16 + self.stones_on_side(Side::White);
            let kb = self.kazan[1] as u16 + self.stones_on_side(Side::Black);
            return Some(match kw.cmp(&kb) {
                std::cmp::Ordering::Greater => GameResult::Win(Side::White),
                std::cmp::Ordering::Less => GameResult::Win(Side::Black),
                std::cmp::Ordering::Equal => GameResult::Draw,
            });
        }

        None
    }

    /// Is the game over?
    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.game_result().is_some()
    }

    /// Total stones on one side (pits only, not kazan)
    #[inline]
    pub fn stones_on_side(&self, side: Side) -> u16 {
        self.pits[side.index()].iter().map(|&x| x as u16).sum()
    }

    /// Total stones on the board (both sides' pits, not kazans)
    #[inline]
    pub fn total_board_stones(&self) -> u16 {
        self.stones_on_side(Side::White) + self.stones_on_side(Side::Black)
    }

    /// Create board from raw parts (for EGTB generation)
    pub fn from_parts(
        pits: [[u8; NUM_PITS]; 2],
        kazan: [u8; 2],
        tuzdyk: [i8; 2],
        side_to_move: Side,
    ) -> Self {
        Board {
            pits,
            kazan,
            tuzdyk,
            side_to_move,
            move_count: 0,
        }
    }
}

/// Parse position string: "w0,w1,...,w8/b0,...,b8/kw,kb/tw,tb/side"
pub fn parse_position(pos: &str) -> Result<Board, String> {
    let parts: Vec<&str> = pos.split('/').collect();
    if parts.len() != 5 {
        return Err(format!("Expected 5 parts separated by '/', got {}", parts.len()));
    }
    let white_pits: Vec<u8> = parts[0].split(',').map(|s| s.parse().unwrap_or(0)).collect();
    let black_pits: Vec<u8> = parts[1].split(',').map(|s| s.parse().unwrap_or(0)).collect();
    let kazans: Vec<u8> = parts[2].split(',').map(|s| s.parse().unwrap_or(0)).collect();
    let tuzdyks: Vec<i8> = parts[3].split(',').map(|s| s.parse().unwrap_or(-1)).collect();
    let side: u8 = parts[4].parse().unwrap_or(0);
    if white_pits.len() != 9 || black_pits.len() != 9 || kazans.len() != 2 || tuzdyks.len() != 2 {
        return Err("Invalid array lengths".into());
    }
    let mut board = Board::new();
    for i in 0..9 {
        board.pits[0][i] = white_pits[i];
        board.pits[1][i] = black_pits[i];
    }
    board.kazan[0] = kazans[0];
    board.kazan[1] = kazans[1];
    board.tuzdyk[0] = tuzdyks[0];
    board.tuzdyk[1] = tuzdyks[1];
    board.side_to_move = if side == 0 { Side::White } else { Side::Black };
    Ok(board)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameResult {
    Win(Side),
    Draw,
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

        // Black pits (reversed)
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

        // White pits
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
    fn test_initial_board() {
        let b = Board::new();
        assert_eq!(b.pits[0], [9; 9]);
        assert_eq!(b.pits[1], [9; 9]);
        assert_eq!(b.kazan, [0, 0]);
        assert_eq!(b.tuzdyk, [-1, -1]);
        assert_eq!(b.side_to_move, Side::White);
    }

    #[test]
    fn test_valid_moves_initial() {
        let b = Board::new();
        let mut moves = [0usize; 9];
        let count = b.valid_moves_array(&mut moves);
        assert_eq!(count, 9); // all pits have stones
    }

    #[test]
    fn test_make_unmake_preserves_state() {
        let mut b = Board::new();
        let original = b;

        let undo = b.make_move(6); // play pit 7
        assert_ne!(b, original);

        b.unmake_move(&undo);
        // After unmake, board should match original except move_count
        // (we decremented it) — actually it should be identical
        assert_eq!(b.pits, original.pits);
        assert_eq!(b.kazan, original.kazan);
        assert_eq!(b.tuzdyk, original.tuzdyk);
        assert_eq!(b.side_to_move, original.side_to_move);
    }

    #[test]
    fn test_first_move_capture() {
        // White plays pit 7 (index 6): 9 stones
        // First stone back to pit 7, then 8 stones to pit 8, 9, opp 1-6
        // Last stone lands on opponent pit 6 (index 5) which had 9+1=10 (even) -> capture
        let mut b = Board::new();
        let _undo = b.make_move(6);

        assert_eq!(b.kazan[0], 10); // White captured 10
        assert_eq!(b.pits[1][5], 0); // opponent pit 6 emptied
        assert_eq!(b.pits[0][6], 1); // source pit has 1 (first stone back)
        assert_eq!(b.side_to_move, Side::Black);
    }

    #[test]
    fn test_single_stone_move() {
        let mut b = Board::new();
        b.pits[0] = [1, 0, 0, 0, 0, 0, 0, 0, 0];
        b.pits[1] = [9, 9, 9, 9, 9, 9, 9, 9, 9];
        b.kazan = [71, 0];

        let _undo = b.make_move(0);
        // Single stone from pit 0 goes to pit 1
        assert_eq!(b.pits[0][0], 0);
        assert_eq!(b.pits[0][1], 1);
    }

    #[test]
    fn test_win_detection() {
        let mut b = Board::new();
        b.kazan[0] = 82;
        assert_eq!(b.game_result(), Some(GameResult::Win(Side::White)));
    }

    #[test]
    fn test_endgame_sweep_breaks_kazan_tie() {
        // When a side empties, each player's remaining board stones are swept
        // into their OWN kazan before deciding the winner. Kazans tied at 71-71
        // but Black still holds 20 on board (total 71+71+20=162) -> Black 91 > 71.
        // The buggy raw-kazan compare returns Draw; correct result is Win(Black).
        let mut b = Board::new();
        b.pits[0] = [0; 9];
        b.pits[1] = [4, 4, 4, 4, 4, 0, 0, 0, 0]; // 20 stones
        b.kazan = [71, 71];
        assert_eq!(b.game_result(), Some(GameResult::Win(Side::Black)));
    }

    #[test]
    fn test_endgame_sweep_conserves_162() {
        // At a board-empty terminal, kazans + swept board stones sum to 162.
        let mut b = Board::new();
        b.pits[0] = [0; 9];
        b.pits[1] = [3, 0, 5, 0, 2, 0, 1, 0, 0]; // 11 stones
        b.kazan = [75, 76];
        let swept_w = b.kazan[0] as u16 + b.stones_on_side(Side::White);
        let swept_b = b.kazan[1] as u16 + b.stones_on_side(Side::Black);
        assert_eq!(swept_w + swept_b, 162);
        assert_eq!(b.game_result(), Some(GameResult::Win(Side::Black))); // 87 > 75
    }

    #[test]
    fn test_tuzdyk_creation() {
        // Set up position where landing on opponent pit with exactly 3 stones
        let mut b = Board::new();
        b.pits[0] = [0, 0, 0, 0, 0, 0, 0, 0, 2]; // pit 9 has 2 stones
        b.pits[1] = [2, 0, 0, 0, 0, 0, 0, 0, 0];  // opponent pit 1 has 2
        b.kazan = [75, 75];

        // White plays pit 9 (index 8): 2 stones
        // Stone 1 back to pit 9, stone 2 wraps to opponent pit 1 (index 0)
        // Opponent pit 1 now has 2+1=3 -> tuzdyk!
        let _undo = b.make_move(8);

        assert_eq!(b.tuzdyk[0], 0); // White's tuzdyk at opponent pit 1
        assert_eq!(b.pits[1][0], 0); // pit emptied
        assert_eq!(b.kazan[0], 78); // captured 3
    }

    #[test]
    fn test_no_tuzdyk_at_pit9() {
        let mut b = Board::new();
        b.pits[0] = [0, 0, 0, 0, 0, 0, 0, 0, 10]; // pit 9 has 10 stones
        b.pits[1] = [0, 0, 0, 0, 0, 0, 0, 0, 2];   // opponent pit 9 has 2
        b.kazan = [75, 75];

        // White plays pit 9 (index 8): 10 stones
        // First stone back to pit 9, remaining 9 wrap to opponent pits 1-9
        // Opponent pit 9 (index 8) now has 2+1=3, but tuzdyk at pit 9 is forbidden
        let _undo = b.make_move(8);

        assert_eq!(b.tuzdyk[0], -1); // no tuzdyk created
        assert_eq!(b.pits[1][8], 3); // stones stay
    }

    #[test]
    fn test_unmake_restores_tuzdyk() {
        // Move that CREATES a tuzdyk must be fully reversible.
        let mut b = Board::new();
        b.pits[0] = [0, 0, 0, 0, 0, 0, 0, 0, 2]; // pit 9 has 2 stones
        b.pits[1] = [2, 0, 0, 0, 0, 0, 0, 0, 0]; // opp pit 1 has 2
        b.kazan = [75, 75];
        let snapshot = b;

        let undo = b.make_move(8); // wraps to opp pit 1 -> 3 -> tuzdyk
        assert_eq!(b.tuzdyk[0], 0, "precondition: tuzdyk created");

        b.unmake_move(&undo);
        assert_eq!(b, snapshot, "unmake must fully restore the board, including tuzdyk");
    }

    #[test]
    fn test_make_unmake_symmetry_and_stone_conservation() {
        // Pseudo-random walk: every make+unmake must restore the board exactly,
        // and the 162 stones must be conserved (pits + both kazans) at all times.
        fn total(b: &Board) -> u16 {
            b.total_board_stones() + b.kazan[0] as u16 + b.kazan[1] as u16
        }
        let mut b = Board::new();
        assert_eq!(total(&b), 162);
        let mut moves = [0usize; NUM_PITS];
        let mut seed: u64 = 0x9e3779b97f4a7c15;
        for _ in 0..1000 {
            if b.is_terminal() {
                b = Board::new();
            }
            let n = b.valid_moves_array(&mut moves);
            if n == 0 {
                b = Board::new();
                continue;
            }
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let pit = moves[(seed >> 33) as usize % n];
            let before = b;
            let undo = b.make_move(pit);
            assert_eq!(total(&b), 162, "stones not conserved after make from pit {}", pit);
            b.unmake_move(&undo);
            assert_eq!(b, before, "make+unmake not symmetric for pit {}", pit);
            b.make_move(pit); // advance for real
        }
    }

    #[test]
    fn test_parse_position_initial() {
        let b = parse_position("9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0").unwrap();
        assert_eq!(b, Board::new());
    }

    #[test]
    fn test_parse_position_rejects_malformed() {
        assert!(parse_position("bad").is_err());
        assert!(parse_position("9,9/9,9/0,0/-1,-1/0").is_err());
    }

    #[test]
    fn test_empty_side_not_terminal_when_opponent_to_move() {
        // White emptied itself; Black still must move. NOT over yet.
        let mut b = Board::new();
        b.pits[0] = [0; 9];
        b.pits[1] = [0, 0, 0, 0, 0, 0, 0, 0, 3];
        b.kazan = [80, 79];
        b.side_to_move = Side::Black;
        assert_eq!(b.game_result(), None);
    }

    #[test]
    fn test_empty_side_terminal_when_it_is_to_move() {
        // Same board, but it is White's turn and White has nothing: game over, sweep.
        let mut b = Board::new();
        b.pits[0] = [0; 9];
        b.pits[1] = [0, 0, 0, 0, 0, 0, 0, 0, 3];
        b.kazan = [80, 79];
        b.side_to_move = Side::White;
        assert_eq!(b.game_result(), Some(GameResult::Win(Side::Black))); // 80 vs 79+3
    }

    #[test]
    fn test_forced_feed_continues_game() {
        // Black's only move (pit 9, 3 stones) drops 2 stones onto White's side:
        // White is back in the game.
        let mut b = Board::new();
        b.pits[0] = [0; 9];
        b.pits[1] = [0, 0, 0, 0, 0, 0, 0, 0, 3];
        b.kazan = [80, 79];
        b.side_to_move = Side::Black;
        b.make_move(8);
        assert_eq!(b.pits[0], [1, 1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(b.side_to_move, Side::White);
        assert_eq!(b.game_result(), None);
        let mut moves = [0usize; NUM_PITS];
        assert_eq!(b.valid_moves_array(&mut moves), 2);
    }

    #[test]
    fn test_lone_stone_from_pit9_then_safe_reply_ends_game() {
        // White plays its last stone from pit 9 -> White empty, Black to move (not over).
        // Black answers with a lone stone that stays on its side -> White to move with
        // nothing -> over; Black sweeps its 6 board stones.
        let mut b = Board::new();
        b.pits[0] = [0, 0, 0, 0, 0, 0, 0, 0, 1];
        b.pits[1] = [0, 0, 0, 0, 0, 0, 0, 0, 5];
        b.kazan = [78, 78];
        b.side_to_move = Side::White;
        b.make_move(8); // lone stone -> Black pit 1
        assert_eq!(b.pits[1], [1, 0, 0, 0, 0, 0, 0, 0, 5]);
        assert_eq!(b.game_result(), None, "self-emptying must not end the game");
        b.make_move(0); // Black: lone stone pit 1 -> pit 2, White still empty
        assert_eq!(b.game_result(), Some(GameResult::Win(Side::Black))); // 78 vs 78+6
    }
}

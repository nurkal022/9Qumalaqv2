/// Opening book from human expert games
///
/// Loads book positions from a text file and provides move lookup.
/// Used in the first ~16 plies to play expert-proven opening moves.

use crate::board::Board;

pub struct BookEntry {
    pits: [[u8; 9]; 2],
    kazan: [u8; 2],
    tuzdyk: [i8; 2],
    side_to_move: u8,
    best_move: u8,
    game_count: u16,
}

pub struct OpeningBook {
    entries: Vec<BookEntry>,
}

impl OpeningBook {
    /// Load opening book from text file
    pub fn load(path: &str) -> Option<Self> {
        let content = std::fs::read_to_string(path).ok()?;
        let mut entries = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() < 7 {
                continue;
            }

            let pits0: Vec<u8> = parts[0].split(',').filter_map(|s| s.parse().ok()).collect();
            let pits1: Vec<u8> = parts[1].split(',').filter_map(|s| s.parse().ok()).collect();
            let kazan: Vec<u8> = parts[2].split(',').filter_map(|s| s.parse().ok()).collect();
            let tuzdyk: Vec<i8> = parts[3].split(',').filter_map(|s| s.parse().ok()).collect();
            let stm: u8 = parts[4].parse().unwrap_or(0);
            let best_move: u8 = parts[5].parse().unwrap_or(0);
            let game_count: u16 = parts[6].parse().unwrap_or(0);

            if pits0.len() != 9 || pits1.len() != 9 || kazan.len() != 2 || tuzdyk.len() != 2 {
                continue;
            }

            let mut p0 = [0u8; 9];
            let mut p1 = [0u8; 9];
            p0.copy_from_slice(&pits0);
            p1.copy_from_slice(&pits1);

            entries.push(BookEntry {
                pits: [p0, p1],
                kazan: [kazan[0], kazan[1]],
                tuzdyk: [tuzdyk[0], tuzdyk[1]],
                side_to_move: stm,
                best_move,
                game_count,
            });
        }

        if entries.is_empty() {
            return None;
        }

        eprintln!("Opening book loaded: {} positions", entries.len());
        Some(OpeningBook { entries })
    }

    /// Look up a book move for the given position
    /// Returns Some(pit_index) if found, None otherwise
    ///
    /// Only uses book for OPENING positions (< 20 stones played).
    /// Beyond that, the search must decide — book moves from mid-game
    /// are based on frequency statistics, not optimal play.
    pub fn lookup(&self, board: &Board) -> Option<usize> {
        // Opening book only: limit to early game
        let stones_played = (board.kazan[0] as u32) + (board.kazan[1] as u32);
        if stones_played >= 20 {
            return None;
        }

        let stm = board.side_to_move.index() as u8;

        for entry in &self.entries {
            if entry.side_to_move != stm {
                continue;
            }
            if entry.pits[0] != board.pits[0] || entry.pits[1] != board.pits[1] {
                continue;
            }
            if entry.kazan[0] != board.kazan[0] || entry.kazan[1] != board.kazan[1] {
                continue;
            }
            if entry.tuzdyk[0] != board.tuzdyk[0] || entry.tuzdyk[1] != board.tuzdyk[1] {
                continue;
            }
            return Some(entry.best_move as usize);
        }
        None
    }
}

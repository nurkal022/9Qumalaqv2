/// Transposition Table for Alpha-Beta search
///
/// Fixed-size hash table storing previously searched positions.
/// Uses "always replace" scheme for simplicity.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TTFlag {
    Exact,      // exact score
    LowerBound, // score >= beta (beta cutoff)
    UpperBound, // score <= alpha (all-node)
}

#[derive(Clone, Copy, Debug)]
pub struct TTEntry {
    pub hash: u64,
    pub depth: i32,
    pub score: i32,
    pub flag: TTFlag,
    pub best_move: i8, // pit index 0-8, -1 if none
}

impl TTEntry {
    pub const EMPTY: TTEntry = TTEntry {
        hash: 0,
        depth: -1,
        score: 0,
        flag: TTFlag::Exact,
        best_move: -1,
    };
}

pub struct TranspositionTable {
    entries: Vec<TTEntry>,
    size: usize,
    mask: usize,
    hits: u64,
    misses: u64,
}

impl TranspositionTable {
    /// Create TT with given size in MB
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<TTEntry>();
        let num_entries = (size_mb * 1024 * 1024) / entry_size;
        // Round down to power of 2
        let size = num_entries.next_power_of_two() / 2;
        let mask = size - 1;

        TranspositionTable {
            entries: vec![TTEntry::EMPTY; size],
            size,
            mask,
            hits: 0,
            misses: 0,
        }
    }

    /// Probe the TT for a position
    #[inline]
    pub fn probe(&mut self, hash: u64) -> Option<&TTEntry> {
        let index = (hash as usize) & self.mask;
        let entry = &self.entries[index];
        if entry.hash == hash && entry.depth >= 0 {
            self.hits += 1;
            Some(entry)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Store a position in the TT
    #[inline]
    pub fn store(&mut self, hash: u64, depth: i32, score: i32, flag: TTFlag, best_move: i8) {
        let index = (hash as usize) & self.mask;
        let entry = &mut self.entries[index];

        // Always replace (simple but effective)
        // Could add depth-preferred replacement later
        entry.hash = hash;
        entry.depth = depth;
        entry.score = score;
        entry.flag = flag;
        entry.best_move = best_move;
    }

    /// Clear the table
    pub fn clear(&mut self) {
        self.entries.fill(TTEntry::EMPTY);
        self.hits = 0;
        self.misses = 0;
    }

    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }
}

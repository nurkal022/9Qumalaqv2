/// Transposition Table for Alpha-Beta search
///
/// Thread-safe for Lazy SMP: uses UnsafeCell for lockless access.
/// Race conditions are benign — hash verification catches corrupted reads.
///
/// Depth-preferred replacement with aging.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

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
    pub generation: u8, // age counter for replacement
}

impl TTEntry {
    pub const EMPTY: TTEntry = TTEntry {
        hash: 0,
        depth: -1,
        score: 0,
        flag: TTFlag::Exact,
        best_move: -1,
        generation: 0,
    };
}

/// Wrapper for UnsafeCell<TTEntry> to allow Send + Sync
struct TTSlot(UnsafeCell<TTEntry>);

// SAFETY: Races are benign — hash check catches corrupted reads.
// This is the standard approach used by Stockfish and other game engines.
unsafe impl Send for TTSlot {}
unsafe impl Sync for TTSlot {}

impl TTSlot {
    fn new(entry: TTEntry) -> Self {
        TTSlot(UnsafeCell::new(entry))
    }
}

pub struct TranspositionTable {
    entries: Vec<TTSlot>,
    mask: usize,
    hits: AtomicU64,
    generation: AtomicU8,
}

// SAFETY: TTSlot is Send + Sync, atomics are Send + Sync
unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}

impl TranspositionTable {
    /// Create TT with given size in MB
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<TTEntry>();
        let num_entries = (size_mb * 1024 * 1024) / entry_size;
        // Round down to power of 2
        let size = num_entries.next_power_of_two() / 2;
        let mask = size - 1;

        let mut entries = Vec::with_capacity(size);
        for _ in 0..size {
            entries.push(TTSlot::new(TTEntry::EMPTY));
        }

        TranspositionTable {
            entries,
            mask,
            hits: AtomicU64::new(0),
            generation: AtomicU8::new(0),
        }
    }

    /// Increment generation (call at start of each new search from root)
    #[inline]
    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Probe the TT for a position. Returns a COPY of the entry.
    #[inline]
    pub fn probe(&self, hash: u64) -> Option<TTEntry> {
        let index = (hash as usize) & self.mask;
        // SAFETY: benign race — hash check catches corrupted reads
        let entry = unsafe { &*self.entries[index].0.get() };
        if entry.hash == hash && entry.depth >= 0 {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(*entry)
        } else {
            None
        }
    }

    /// Store a position in the TT with depth-preferred replacement + aging
    #[inline]
    pub fn store(&self, hash: u64, depth: i32, score: i32, flag: TTFlag, best_move: i8) {
        let index = (hash as usize) & self.mask;
        let gen = self.generation.load(Ordering::Relaxed);
        // SAFETY: benign race — worst case is a corrupted entry caught by hash check
        let entry = unsafe { &mut *self.entries[index].0.get() };

        let should_replace = entry.depth < 0
            || entry.hash == hash
            || entry.generation != gen
            || depth >= entry.depth;

        if should_replace {
            entry.hash = hash;
            entry.depth = depth;
            entry.score = score;
            entry.flag = flag;
            entry.best_move = best_move;
            entry.generation = gen;
        }
    }

    /// Clear the table
    pub fn clear(&self) {
        for slot in &self.entries {
            // SAFETY: only called when no searches are active
            let entry = unsafe { &mut *slot.0.get() };
            *entry = TTEntry::EMPTY;
        }
        self.hits.store(0, Ordering::Relaxed);
        self.generation.store(0, Ordering::Relaxed);
    }

    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }
}

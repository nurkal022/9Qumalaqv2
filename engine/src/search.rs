/// Alpha-Beta search with iterative deepening and Lazy SMP
///
/// Features:
/// - Negamax alpha-beta with fail-soft
/// - Iterative deepening with Aspiration Windows
/// - Lazy SMP (multi-threaded search with shared TT)
/// - Smart time management (early exit on stable/forced moves)
/// - Transposition table (depth-preferred + aging)
/// - Move ordering: TT move, captures, killers, countermove, history
/// - Null move pruning (adaptive R)
/// - Late Move Reductions (logarithmic, history-adjusted)
/// - Late Move Pruning (LMP)
/// - Reverse Futility Pruning (Static NMP)
/// - Razoring
/// - Futility Pruning
/// - Internal Iterative Reductions (IIR, replaces IID)
/// - Tuzdyk Extensions (like check extensions)
/// - Killer moves + Countermove heuristic
/// - History Heuristic with gravity (bonus + malus + aging)
/// - Continuation history (1-ply context for move ordering)
/// - Improving heuristic (scale pruning by eval trend)

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use crate::board::{Board, GameResult, NUM_PITS};
use crate::book::OpeningBook;
use crate::egtb::{EndgameTablebase, EgtbResult};
use crate::eval::{evaluate, EVAL_INF, EVAL_MATE};
use crate::nnue::NnueNetwork;
use crate::tt::{TranspositionTable, TTFlag};
use crate::zobrist::ZobristKeys;

const MAX_DEPTH: i32 = 64;
const NULL_MOVE_R: i32 = 2;
const LMR_THRESHOLD: usize = 2; // reduce moves after this many (0-indexed)

/// Aspiration window initial delta (calibrated for NNUE/64 scale)
const ASP_DELTA: i32 = 35;

/// Reverse Futility Pruning margins per depth (calibrated for NNUE/64 scale)
const RFP_MARGIN: i32 = 70;

// Singular Extension disabled — causes node explosion in togyz kumalak
// (branching factor 5-8 means SE searches 4-7 extra subtrees per TT move)

/// Late Move Pruning: max moves to try at shallow depths
/// lmp_table[depth] = max quiet moves before pruning
const LMP_TABLE: [usize; 7] = [0, 5, 8, 12, 16, 20, 24];

pub struct SearchResult {
    pub best_move: usize,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
    pub tt_hits: u64,
    pub time_ms: u64,
}

pub struct Searcher {
    pub tt: Arc<TranspositionTable>,
    pub zobrist: ZobristKeys,
    pub nodes: u64,
    pub max_time_ms: u64,
    pub silent: bool,
    start_time: Instant,
    stopped: bool,
    /// Shared stop flag for SMP — when set, all threads should stop
    abort: Arc<AtomicBool>,
    killer_moves: [[i8; 2]; MAX_DEPTH as usize], // [depth][slot]
    history: [[i32; NUM_PITS]; 2], // [side][pit] history heuristic
    countermove: [[i8; NUM_PITS]; 2], // [side][prev_move] → refutation move
    /// Continuation history: [side][prev_move][current_move] → score
    /// Tracks which moves are good responses to a given previous move
    cont_history: [[[i32; NUM_PITS]; NUM_PITS]; 2],
    nnue: Option<NnueNetwork>,
    egtb: Option<Arc<EndgameTablebase>>,
    opening_book: Option<OpeningBook>,
    /// Positions played in the actual game (for repetition detection)
    pub game_history: Vec<u64>,
    /// Search path hashes (indexed by ply, for in-tree repetition detection)
    search_path: [u64; MAX_DEPTH as usize],
    /// Previous move at each ply (for countermove heuristic)
    prev_move: [i8; MAX_DEPTH as usize],
    /// Static eval at each ply (for "improving" heuristic)
    static_evals: [i32; MAX_DEPTH as usize],
}

impl Searcher {
    pub fn new(tt_size_mb: usize) -> Self {
        Searcher {
            tt: Arc::new(TranspositionTable::new(tt_size_mb)),
            zobrist: ZobristKeys::new(),
            nodes: 0,
            max_time_ms: 5000,
            silent: false,
            start_time: Instant::now(),
            stopped: false,
            abort: Arc::new(AtomicBool::new(false)),
            killer_moves: [[-1; 2]; MAX_DEPTH as usize],
            history: [[0; NUM_PITS]; 2],
            countermove: [[-1; NUM_PITS]; 2],
            cont_history: [[[0; NUM_PITS]; NUM_PITS]; 2],
            nnue: None,
            egtb: None,
            opening_book: None,
            game_history: Vec::new(),
            search_path: [0; MAX_DEPTH as usize],
            prev_move: [-1; MAX_DEPTH as usize],
            static_evals: [0; MAX_DEPTH as usize],
        }
    }

    /// Create a worker searcher sharing the TT and abort flag with another searcher
    fn new_worker(tt: Arc<TranspositionTable>, abort: Arc<AtomicBool>, nnue: Option<NnueNetwork>, egtb: Option<Arc<EndgameTablebase>>) -> Self {
        Searcher {
            tt,
            zobrist: ZobristKeys::new(),
            nodes: 0,
            max_time_ms: u64::MAX, // workers rely on abort flag, not time
            silent: true,
            start_time: Instant::now(),
            stopped: false,
            abort,
            killer_moves: [[-1; 2]; MAX_DEPTH as usize],
            history: [[0; NUM_PITS]; 2],
            countermove: [[-1; NUM_PITS]; 2],
            cont_history: [[[0; NUM_PITS]; NUM_PITS]; 2],
            nnue,
            egtb,
            opening_book: None,
            game_history: Vec::new(),
            search_path: [0; MAX_DEPTH as usize],
            prev_move: [-1; MAX_DEPTH as usize],
            static_evals: [0; MAX_DEPTH as usize],
        }
    }

    pub fn set_nnue(&mut self, nnue: NnueNetwork) {
        self.nnue = Some(nnue);
    }

    pub fn set_egtb(&mut self, egtb: Arc<EndgameTablebase>) {
        self.egtb = Some(egtb);
    }

    pub fn set_book(&mut self, book: OpeningBook) {
        self.opening_book = Some(book);
    }

    /// Push a position hash to game history (call after each actual game move)
    pub fn push_game_position(&mut self, hash: u64) {
        self.game_history.push(hash);
    }

    /// Compute hash for a board position
    pub fn compute_hash(&self, board: &Board) -> u64 {
        self.zobrist.hash(board)
    }

    /// Evaluate using NNUE if available, otherwise handcrafted.
    /// NNUE output is divided by NNUE SCALE (64) to convert from quantized
    /// fixed-point back to the model's native scale (~centipawn-like).
    /// In endgame, adds mobility/stone-preservation bonus to NNUE eval.
    #[inline]
    fn eval(&self, board: &Board) -> i32 {
        if let Some(ref nnue) = self.nnue {
            let base = nnue.evaluate(board) / 64;

            // Endgame correction: reward having more active pits (mobility)
            // and penalize positions where stones are too concentrated
            let me = board.side_to_move.index();
            let opp = 1 - me;
            let my_stones: u16 = board.pits[me].iter().map(|&x| x as u16).sum();
            let opp_stones: u16 = board.pits[opp].iter().map(|&x| x as u16).sum();
            let total = my_stones + opp_stones;

            if total <= 60 {
                let my_active = board.pits[me].iter().filter(|&&x| x > 0).count() as i32;
                let opp_active = board.pits[opp].iter().filter(|&&x| x > 0).count() as i32;

                // Scale corrections smoothly: total=60→1, total=30→2, total=15→3, total=5→4
                let scale = ((65 - total as i32).max(1)) / 15;
                let scale = scale.clamp(1, 4);

                // Mobility: critical endgame factor (PlayOK: mobility weight = 124 in HCE)
                let mobility_bonus = (my_active - opp_active) * 3 * scale;

                // Starvation: quadratic pressure when opponent running low
                let starvation = if opp_stones <= 20 {
                    let pressure = 21 - opp_stones as i32;
                    (pressure * pressure * scale) / 8
                } else { 0 };

                // Finishing: huge bonus to close out won games
                let my_kazan = board.kazan[me] as i32;
                let opp_kazan = board.kazan[opp] as i32;
                let finish_bonus = if my_kazan > opp_kazan + 5 && opp_stones <= 8 {
                    (9 - opp_stones as i32) * 8 * scale
                } else { 0 };

                // Kazan proximity: accelerate when close to 82
                let kazan_bonus = if my_kazan >= 65 {
                    (my_kazan - 65) * 2 * scale
                } else { 0 };

                base + mobility_bonus + starvation + finish_bonus + kazan_bonus
            } else {
                base
            }
        } else {
            evaluate(board)
        }
    }

    /// Check if a move creates a tuzdyk
    #[inline]
    fn move_creates_tuzdyk(&self, board: &Board, pit: usize) -> bool {
        let me = board.side_to_move.index();
        let stones = board.pits[me][pit];

        if stones == 0 || board.tuzdyk[me] >= 0 {
            return false;
        }

        let (landing_side, landing_pit) = self.predict_landing(board, pit);
        let opp = board.side_to_move.opposite().index();

        if landing_side == opp && landing_pit < 8 {
            let target = board.pits[opp][landing_pit] + 1;
            if target == 3 && board.tuzdyk[opp] != landing_pit as i8 {
                return true;
            }
        }

        false
    }

    /// Predict where last stone lands (side, pit)
    #[inline]
    fn predict_landing(&self, board: &Board, pit: usize) -> (usize, usize) {
        let me = board.side_to_move.index();
        let opp = board.side_to_move.opposite().index();
        let stones = board.pits[me][pit];

        if stones == 1 {
            let next = pit + 1;
            if next > 8 {
                (opp, 0)
            } else {
                (me, next)
            }
        } else {
            let remaining = stones as usize - 1;
            let mut pos = pit + remaining;
            let mut side = me;
            while pos > 8 {
                pos -= 9;
                side = 1 - side;
            }
            (side, pos)
        }
    }

    /// Lazy SMP search: spawn helper threads that share the TT
    pub fn search_smp(&mut self, board: &Board, max_depth: i32, time_ms: u64, num_threads: usize) -> SearchResult {
        if num_threads <= 1 {
            return self.search(board, max_depth, time_ms);
        }

        // Reset abort flag
        self.abort.store(false, Ordering::SeqCst);

        // Spawn helper threads
        let mut handles = Vec::new();
        for tid in 1..num_threads {
            let tt = self.tt.clone();
            let abort = self.abort.clone();
            let nnue = self.nnue.clone();
            let egtb = self.egtb.clone();
            let board_copy = *board;
            let game_hist = self.game_history.clone();

            handles.push(std::thread::spawn(move || {
                let mut worker = Searcher::new_worker(tt, abort, nnue, egtb);
                worker.game_history = game_hist;
                worker.start_time = Instant::now();
                worker.max_time_ms = time_ms + 1000; // workers get extra time, rely on abort

                // Depth offset: helpers search at different starting depths for diversity
                // Thread 1: depths 1,2,3,...  Thread 2: depths 2,3,4,...  etc.
                let start_depth = 1 + (tid as i32 % 3);

                for depth in start_depth..=max_depth {
                    if worker.abort.load(Ordering::Relaxed) {
                        break;
                    }
                    worker.alpha_beta(&board_copy, depth, -EVAL_INF, EVAL_INF, 0);
                    if worker.stopped {
                        break;
                    }
                }
                worker.nodes
            }));
        }

        // Main thread does normal search
        let result = self.search(board, max_depth, time_ms);

        // Signal all helpers to stop
        self.abort.store(true, Ordering::SeqCst);

        // Collect helper thread nodes for reporting
        let mut total_nodes = result.nodes;
        for handle in handles {
            if let Ok(worker_nodes) = handle.join() {
                total_nodes += worker_nodes;
            }
        }

        SearchResult {
            nodes: total_nodes,
            ..result
        }
    }

    /// Search with iterative deepening + aspiration windows
    pub fn search(&mut self, board: &Board, max_depth: i32, time_ms: u64) -> SearchResult {
        self.nodes = 0;
        self.start_time = Instant::now();
        self.stopped = false;
        self.killer_moves = [[-1; 2]; MAX_DEPTH as usize];

        // === ENDGAME TIME MANAGEMENT ===
        // Allocate more time in endgame where precision matters most
        let total_board_stones: u16 = board.pits[0].iter().map(|&x| x as u16).sum::<u16>()
            + board.pits[1].iter().map(|&x| x as u16).sum::<u16>();
        self.max_time_ms = if total_board_stones <= 30 {
            time_ms * 2       // Deep endgame: 2x time
        } else if total_board_stones <= 60 {
            time_ms * 3 / 2   // Endgame: 1.5x time
        } else {
            time_ms
        };

        // Opening book probe — instant move if in book
        if let Some(ref book) = self.opening_book {
            if let Some(book_move) = book.lookup(board) {
                if !self.silent {
                    eprintln!("info depth 0 score 0 nodes 0 nps 0 time 0 pv pit{} (book)", book_move + 1);
                }
                return SearchResult {
                    best_move: book_move,
                    score: 0,
                    depth: 0,
                    nodes: 0,
                    tt_hits: 0,
                    time_ms: 0,
                };
            }
        }

        // Age history values: halve to prevent stale values from dominating
        for side in 0..2 {
            for pit in 0..NUM_PITS {
                self.history[side][pit] /= 2;
            }
        }

        // New search generation for TT aging
        self.tt.new_search();

        let mut best_move = 0usize;
        let mut best_score = -EVAL_INF;
        let mut completed_depth = 0i32;

        // Get any valid move as fallback
        let mut moves = [0usize; NUM_PITS];
        let num_moves = board.valid_moves_array(&mut moves);
        if num_moves == 0 {
            return SearchResult {
                best_move: 0,
                score: self.eval(board),
                depth: 0,
                nodes: 0,
                tt_hits: 0,
                time_ms: 0,
            };
        }
        best_move = moves[0];

        // Only one legal move: return immediately
        if num_moves == 1 {
            return SearchResult {
                best_move: moves[0],
                score: self.eval(board),
                depth: 0,
                nodes: 1,
                tt_hits: 0,
                time_ms: 0,
            };
        }

        let mut prev_score = 0i32;
        let mut best_move_stable_count = 0u32;
        let mut prev_best_move = usize::MAX;

        // Iterative deepening with aspiration windows
        for depth in 1..=max_depth {
            // Check abort flag (for SMP workers)
            if self.abort.load(Ordering::Relaxed) {
                self.stopped = true;
                break;
            }

            let score = if depth >= 4 && prev_score.abs() < EVAL_MATE - 100 {
                // Aspiration windows: search with narrow window around previous score
                let mut delta = ASP_DELTA;
                let mut asp_alpha = (prev_score - delta).max(-EVAL_INF);
                let mut asp_beta = (prev_score + delta).min(EVAL_INF);
                let mut result = 0i32;

                loop {
                    result = self.alpha_beta(board, depth, asp_alpha, asp_beta, 0);
                    if self.stopped { break; }

                    if result <= asp_alpha {
                        // Fail low: widen alpha
                        delta *= 2;
                        asp_alpha = (prev_score - delta).max(-EVAL_INF);
                    } else if result >= asp_beta {
                        // Fail high: widen beta
                        delta *= 2;
                        asp_beta = (prev_score + delta).min(EVAL_INF);
                    } else {
                        break;
                    }

                    if delta > 500 {
                        // Window too wide, fall back to full search
                        result = self.alpha_beta(board, depth, -EVAL_INF, EVAL_INF, 0);
                        break;
                    }
                }
                result
            } else {
                self.alpha_beta(board, depth, -EVAL_INF, EVAL_INF, 0)
            };

            if self.stopped {
                break;
            }

            best_score = score;
            prev_score = score;
            completed_depth = depth;

            // Get best move from TT
            let hash = self.zobrist.hash(board);
            if let Some(entry) = self.tt.probe(hash) {
                if entry.best_move >= 0 {
                    best_move = entry.best_move as usize;
                }
            }

            let elapsed = self.start_time.elapsed().as_millis() as u64;
            let nps = if elapsed > 0 {
                self.nodes * 1000 / elapsed
            } else {
                self.nodes
            };

            if !self.silent {
                eprintln!(
                    "info depth {} score {} nodes {} nps {} time {} pv pit{}",
                    depth,
                    best_score,
                    self.nodes,
                    nps,
                    elapsed,
                    best_move + 1,
                );
            }

            // If we found a mate, no need to search deeper
            if best_score.abs() > EVAL_MATE - 100 {
                break;
            }

            // Smart time management: stop early if position is "easy"
            // or if next depth is unlikely to complete in time
            // In endgame: be more reluctant to exit early (every ply matters)
            if depth >= 6 && elapsed > 0 {
                // Track best move stability
                if best_move == prev_best_move {
                    best_move_stable_count += 1;
                } else {
                    best_move_stable_count = 0;
                    prev_best_move = best_move;
                }

                // Stability exit: require MORE stability in endgame
                let stability_threshold = if total_board_stones <= 60 { 6u32 } else { 4u32 };
                let time_fraction = if total_board_stones <= 60 { 2u64 } else { 3u64 };
                if best_move_stable_count >= stability_threshold && elapsed > self.max_time_ms / time_fraction {
                    break;
                }

                // Estimate: next depth will take ~3x current depth
                // If 3 * elapsed > time_limit, we won't complete next depth
                if elapsed * 3 > self.max_time_ms {
                    break;
                }
            }
        }

        let elapsed = self.start_time.elapsed().as_millis() as u64;

        SearchResult {
            best_move,
            score: best_score,
            depth: completed_depth,
            nodes: self.nodes,
            tt_hits: self.tt.hits(),
            time_ms: elapsed,
        }
    }

    /// Negamax alpha-beta with fail-soft
    fn alpha_beta(
        &mut self,
        board: &Board,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: i32,
    ) -> i32 {
        // Time check periodically
        if self.nodes & 4095 == 0 {
            if self.abort.load(Ordering::Relaxed) {
                self.stopped = true;
                return 0;
            }
            let elapsed = self.start_time.elapsed().as_millis() as u64;
            if elapsed >= self.max_time_ms {
                self.stopped = true;
                return 0;
            }
        }

        self.nodes += 1;

        // Terminal check
        if let Some(result) = board.game_result() {
            return match result {
                GameResult::Win(winner) => {
                    if winner == board.side_to_move {
                        EVAL_MATE - ply
                    } else {
                        -EVAL_MATE + ply
                    }
                }
                GameResult::Draw => 0,
            };
        }

        // EGTB probe: perfect endgame evaluation
        if let Some(ref egtb) = self.egtb {
            if let Some(entry) = egtb.probe(board) {
                return match entry.result {
                    EgtbResult::Win => EVAL_MATE - ply - entry.dtm as i32,
                    EgtbResult::Loss => -EVAL_MATE + ply + entry.dtm as i32,
                    EgtbResult::Draw => 0,
                };
            }
        }

        // Leaf node
        if depth <= 0 {
            return self.quiescence(board, alpha, beta, ply);
        }

        // Hard ply limit to prevent stack overflow from extensions
        if ply >= MAX_DEPTH - 4 {
            return self.eval(board);
        }

        // PV node detection
        let is_pv = beta - alpha > 1;

        // TT probe
        let hash = self.zobrist.hash(board);

        // === REPETITION DETECTION ===
        // Store current hash in search path
        self.search_path[ply as usize] = hash;

        if ply > 0 {
            // Check game history (positions already played)
            for &h in &self.game_history {
                if h == hash {
                    return 0; // repetition = draw
                }
            }
            // Check search path (in-tree repetition)
            for i in 0..(ply as usize) {
                if self.search_path[i] == hash {
                    return 0; // cycle in search tree
                }
            }
        }

        let mut tt_move: i8 = -1;

        if let Some(entry) = self.tt.probe(hash) {
            tt_move = entry.best_move;
            if entry.depth >= depth && !is_pv {
                match entry.flag {
                    TTFlag::Exact => return entry.score,
                    TTFlag::LowerBound => {
                        if entry.score >= beta {
                            return entry.score;
                        }
                        if entry.score > alpha {
                            alpha = entry.score;
                        }
                    }
                    TTFlag::UpperBound => {
                        if entry.score <= alpha {
                            return entry.score;
                        }
                    }
                }
            }
        }

        // === ENDGAME DETECTION ===
        // Two separate concepts:
        // - is_endgame/is_deep_endgame: controls search pruning (keep tight, pruning = depth)
        // - Eval corrections in eval(): apply from 60 stones (where games are decided)
        let total_board_stones: u16 = board.pits[0].iter().map(|&x| x as u16).sum::<u16>()
            + board.pits[1].iter().map(|&x| x as u16).sum::<u16>();
        let is_endgame = total_board_stones <= 30;
        let is_deep_endgame = total_board_stones <= 15;

        // Static eval (for pruning decisions)
        let static_eval = self.eval(board);
        self.static_evals[ply as usize] = static_eval;

        // === IMPROVING HEURISTIC ===
        // Position is "improving" if our eval is better than 2 plies ago
        // When improving: prune less aggressively (we're on a good track)
        // When not improving: prune more aggressively
        let improving = ply >= 2 && static_eval > self.static_evals[(ply - 2) as usize];

        // === RAZORING ===
        if !is_pv && depth <= 2 && !is_endgame && static_eval + 150 * depth < alpha {
            let razor_score = self.quiescence(board, alpha, beta, ply);
            if razor_score <= alpha {
                return razor_score;
            }
        }

        // === REVERSE FUTILITY PRUNING (Static NMP) ===
        // When improving, use tighter margin (less pruning — we need accuracy)
        let rfp_margin = RFP_MARGIN * depth - if improving { 30 } else { 0 };
        if !is_pv && depth <= 3 && !is_endgame && static_eval - rfp_margin >= beta {
            return static_eval;
        }

        // === INTERNAL ITERATIVE DEEPENING (IID) ===
        // If no TT move at PV node, do a shallow search to find a good move for ordering
        if is_pv && tt_move < 0 && depth >= 4 {
            self.alpha_beta(board, depth - 2, alpha, beta, ply);
            if let Some(entry) = self.tt.probe(hash) {
                tt_move = entry.best_move;
            }
        }

        // === NULL MOVE PRUNING (aggressive) ===
        let my_stones: u16 = board.pits[board.side_to_move.index()]
            .iter()
            .map(|&x| x as u16)
            .sum();

        if depth >= 3 && my_stones > 10 && !is_endgame && ply > 0 && !is_pv {
            let mut null_board = *board;
            null_board.side_to_move = null_board.side_to_move.opposite();
            null_board.move_count += 1;

            let nmp_r = NULL_MOVE_R + depth / 6;
            let null_score = -self.alpha_beta(&null_board, (depth - 1 - nmp_r).max(0), -beta, -beta + 1, ply + 1);

            if self.stopped {
                return 0;
            }

            if null_score >= beta {
                return null_score;
            }
        }

        // ProbCut disabled — causes recursive explosion in this game tree

        // Generate and order moves
        let mut moves = [0usize; NUM_PITS];
        let num_moves = board.valid_moves_array(&mut moves);

        if num_moves == 0 {
            return self.eval(board);
        }

        // Order moves
        let mut move_scores = [0i32; NUM_PITS];
        let prev_move_for_cm = if ply > 0 { self.prev_move[ply as usize - 1] } else { -1 };
        self.score_moves(board, &moves, num_moves, tt_move, ply, prev_move_for_cm, &mut move_scores);

        // Sort by score (selection sort — fast for 4-9 elements)
        for i in 0..num_moves {
            let mut best_idx = i;
            for j in (i + 1)..num_moves {
                if move_scores[j] > move_scores[best_idx] {
                    best_idx = j;
                }
            }
            if best_idx != i {
                moves.swap(i, best_idx);
                move_scores.swap(i, best_idx);
            }
        }

        // === ENDGAME DEPTH EXTENSION ===
        // Deep endgame (≤30 stones): extend at ply ≤ 6
        // Regular endgame (≤60 stones): extend at ply ≤ 2
        let endgame_ext: i32 = if is_deep_endgame && ply <= 6 { 1 }
            else if is_endgame && ply <= 2 { 1 }
            else { 0 };

        // Singular Extension removed — node explosion in togyz kumalak

        let mut best_score = -EVAL_INF;
        let mut best_move: i8 = moves[0] as i8;
        let mut flag = TTFlag::UpperBound;

        // Futility pruning flag
        let futility_pruning = !is_pv && depth <= 2 && !is_endgame
            && static_eval + 80 * depth < alpha;

        // Track quiet moves tried (for history malus)
        let mut quiet_moves_tried: [usize; NUM_PITS] = [0; NUM_PITS];
        let mut num_quiet_tried: usize = 0;

        for i in 0..num_moves {
            let m = moves[i];
            let is_capture = self.is_capture_move(board, m);
            let creates_tuzdyk = self.move_creates_tuzdyk(board, m);
            let is_tt_move = tt_move >= 0 && m == tt_move as usize;

            // LMP: skip late quiet moves at shallow depths
            if !is_pv && depth <= 3 && !is_endgame && !is_capture && !creates_tuzdyk && !is_tt_move {
                if depth < LMP_TABLE.len() as i32 {
                    if i >= LMP_TABLE[depth as usize] {
                        continue;
                    }
                }
            }

            // Futility pruning
            if futility_pruning && !is_capture && !creates_tuzdyk && !is_tt_move && i > 0 {
                continue;
            }

            // History-based pruning disabled for now

            let mut new_board = *board;
            let _undo = new_board.make_move(m);

            // Track move for countermove heuristic at child nodes
            self.prev_move[ply as usize] = m as i8;

            // === EXTENSIONS ===
            let mut extension = 0i32;
            if creates_tuzdyk {
                extension += 1;
            }

            // Singular Extension removed (see comment above)

            let effective_depth = depth - 1 + endgame_ext + extension;

            let score;

            // LMR: reduce depth for late quiet moves (disabled in endgame)
            if i >= LMR_THRESHOLD && depth >= 3 && !is_endgame && !is_capture && !creates_tuzdyk {
                let ln_d = (depth as f32).ln();
                let ln_i = ((i + 1) as f32).ln();
                let mut reduction = (ln_d * ln_i / 2.5) as i32;
                if is_pv { reduction -= 1; }
                if improving { reduction -= 1; }
                let side = board.side_to_move.index();
                let hist = self.history[side][m];
                if hist > 3000 { reduction -= 1; }
                else if hist < -3000 { reduction += 1; }
                reduction = reduction.clamp(1, depth - 1);

                let reduced = -self.alpha_beta(&new_board, (effective_depth - reduction).max(1), -alpha - 1, -alpha, ply + 1);
                if reduced > alpha {
                    score = -self.alpha_beta(&new_board, effective_depth, -beta, -alpha, ply + 1);
                } else {
                    score = reduced;
                }
            } else if i > 0 {
                // PVS: search with null window first
                let pv_score = -self.alpha_beta(&new_board, effective_depth, -alpha - 1, -alpha, ply + 1);
                if pv_score > alpha && pv_score < beta {
                    score = -self.alpha_beta(&new_board, effective_depth, -beta, -alpha, ply + 1);
                } else {
                    score = pv_score;
                }
            } else {
                score = -self.alpha_beta(&new_board, effective_depth, -beta, -alpha, ply + 1);
            }

            if self.stopped {
                return 0;
            }

            // Track quiet moves for history malus
            if !is_capture {
                quiet_moves_tried[num_quiet_tried] = m;
                num_quiet_tried += 1;
            }

            if score > best_score {
                best_score = score;
                best_move = m as i8;

                if score > alpha {
                    alpha = score;
                    flag = TTFlag::Exact;

                    if score >= beta {
                        flag = TTFlag::LowerBound;

                        // Update killer moves
                        if (ply as usize) < MAX_DEPTH as usize && !is_capture {
                            let ply_idx = ply as usize;
                            if self.killer_moves[ply_idx][0] != m as i8 {
                                self.killer_moves[ply_idx][1] = self.killer_moves[ply_idx][0];
                                self.killer_moves[ply_idx][0] = m as i8;
                            }
                        }

                        // Update countermove
                        if ply > 0 && !is_capture {
                            let prev = self.prev_move[ply as usize - 1];
                            if prev >= 0 {
                                let opp = 1 - board.side_to_move.index();
                                self.countermove[opp][prev as usize] = m as i8;
                            }
                        }

                        // === HISTORY GRAVITY (bonus + malus) ===
                        let side = board.side_to_move.index();
                        let bonus = depth * depth;
                        self.history[side][m] = (self.history[side][m] + bonus).min(16384);

                        // Update continuation history
                        if ply > 0 {
                            let prev = self.prev_move[ply as usize - 1];
                            if prev >= 0 {
                                let opp = 1 - side;
                                self.cont_history[opp][prev as usize][m] =
                                    (self.cont_history[opp][prev as usize][m] + bonus).min(16384);
                            }
                        }

                        for j in 0..num_quiet_tried {
                            let tried = quiet_moves_tried[j];
                            if tried != m {
                                self.history[side][tried] = (self.history[side][tried] - bonus).max(-16384);
                                // Malus for continuation history too
                                if ply > 0 {
                                    let prev = self.prev_move[ply as usize - 1];
                                    if prev >= 0 {
                                        let opp = 1 - side;
                                        self.cont_history[opp][prev as usize][tried] =
                                            (self.cont_history[opp][prev as usize][tried] - bonus).max(-16384);
                                    }
                                }
                            }
                        }

                        break;
                    }
                }
            }
        }

        // Store in TT
        self.tt.store(hash, depth, best_score, flag, best_move);

        best_score
    }

    /// Quiescence search: only search captures and tuzdyk-creating moves
    /// In deep endgame, searches all moves for better tactical vision
    fn quiescence(&mut self, board: &Board, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        self.quiescence_inner(board, alpha, beta, ply, 0)
    }

    fn quiescence_inner(&mut self, board: &Board, mut alpha: i32, beta: i32, ply: i32, qs_depth: i32) -> i32 {
        if self.nodes & 4095 == 0 {
            if self.abort.load(Ordering::Relaxed) {
                self.stopped = true;
                return 0;
            }
            let elapsed = self.start_time.elapsed().as_millis() as u64;
            if elapsed >= self.max_time_ms {
                self.stopped = true;
                return 0;
            }
        }

        self.nodes += 1;

        if let Some(result) = board.game_result() {
            return match result {
                GameResult::Win(winner) => {
                    if winner == board.side_to_move {
                        EVAL_MATE - ply
                    } else {
                        -EVAL_MATE + ply
                    }
                }
                GameResult::Draw => 0,
            };
        }

        // EGTB probe in quiescence: perfect endgame evaluation
        if let Some(ref egtb) = self.egtb {
            if let Some(entry) = egtb.probe(board) {
                return match entry.result {
                    EgtbResult::Win => EVAL_MATE - ply - entry.dtm as i32,
                    EgtbResult::Loss => -EVAL_MATE + ply + entry.dtm as i32,
                    EgtbResult::Draw => 0,
                };
            }
        }

        // === TT PROBE IN QUIESCENCE ===
        let hash = self.zobrist.hash(board);
        if let Some(entry) = self.tt.probe(hash) {
            // In qsearch, depth=0 entries are always valid
            if entry.depth >= 0 {
                match entry.flag {
                    TTFlag::Exact => return entry.score,
                    TTFlag::LowerBound => {
                        if entry.score >= beta { return entry.score; }
                    }
                    TTFlag::UpperBound => {
                        if entry.score <= alpha { return entry.score; }
                    }
                }
            }
        }

        let stand_pat = self.eval(board);

        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        if ply >= MAX_DEPTH - 2 {
            return stand_pat;
        }

        // Endgame qsearch: search ALL moves (not just captures) for better tactical vision
        // ≤30 stones: first 2 ply all-moves (light), ≤15 stones: first 4 ply (deep)
        let total_board_stones: u16 = board.pits[0].iter().map(|&x| x as u16).sum::<u16>()
            + board.pits[1].iter().map(|&x| x as u16).sum::<u16>();
        let qsearch_all_moves = (total_board_stones <= 15 && qs_depth < 4)
            || (total_board_stones <= 30 && qs_depth < 2);

        let mut moves = [0usize; NUM_PITS];
        let num_moves = board.valid_moves_array(&mut moves);

        // === QSEARCH MOVE ORDERING ===
        // Order captures by expected capture value (MVV-like) for better beta cutoffs
        let mut move_scores = [0i32; NUM_PITS];
        for i in 0..num_moves {
            let m = moves[i];
            let mut score = 0i32;
            if self.move_creates_tuzdyk(board, m) {
                score += 50_000;
            }
            if self.is_capture_move(board, m) {
                let (landing_side, landing_pit) = self.predict_landing(board, m);
                let opp = board.side_to_move.opposite().index();
                if landing_side == opp {
                    score += 10_000 + board.pits[opp][landing_pit] as i32 * 100;
                }
            }
            score += board.pits[board.side_to_move.index()][m] as i32;
            move_scores[i] = score;
        }
        // Selection sort (fast for ≤9 elements)
        for i in 0..num_moves {
            let mut best_idx = i;
            for j in (i + 1)..num_moves {
                if move_scores[j] > move_scores[best_idx] {
                    best_idx = j;
                }
            }
            if best_idx != i {
                moves.swap(i, best_idx);
                move_scores.swap(i, best_idx);
            }
        }

        let mut best_score = stand_pat;

        for i in 0..num_moves {
            let m = moves[i];

            if !qsearch_all_moves {
                let is_capture = self.is_capture_move(board, m);
                let creates_tuzdyk = self.move_creates_tuzdyk(board, m);

                if !is_capture && !creates_tuzdyk {
                    continue;
                }
            }

            let mut new_board = *board;
            let _undo = new_board.make_move(m);

            let score = -self.quiescence_inner(&new_board, -beta, -alpha, ply + 1, qs_depth + 1);

            if self.stopped {
                return 0;
            }

            if score > best_score {
                best_score = score;
            }

            if score >= beta {
                // Store in TT as lower bound
                self.tt.store(hash, 0, score, TTFlag::LowerBound, m as i8);
                return score;
            }
            if score > alpha {
                alpha = score;
            }
        }

        // Store best score in TT (fail-hard: return alpha, not best_score)
        let flag = if best_score > stand_pat { TTFlag::Exact } else { TTFlag::UpperBound };
        self.tt.store(hash, 0, best_score, flag, -1);

        alpha
    }

    /// Check if a move would result in a capture (approximate)
    #[inline]
    fn is_capture_move(&self, board: &Board, pit: usize) -> bool {
        let me = board.side_to_move.index();
        let opp = board.side_to_move.opposite().index();
        let stones = board.pits[me][pit];

        if stones == 0 {
            return false;
        }

        let (landing_side, landing_pit) = self.predict_landing(board, pit);

        if landing_side == opp {
            let target = board.pits[opp][landing_pit] + 1;
            if target % 2 == 0 || target == 3 {
                return true;
            }
        }

        false
    }

    /// Score moves for ordering
    fn score_moves(
        &self,
        board: &Board,
        moves: &[usize; NUM_PITS],
        num_moves: usize,
        tt_move: i8,
        ply: i32,
        prev_move_for_cm: i8,
        scores: &mut [i32; NUM_PITS],
    ) {
        let side = board.side_to_move.index();

        for i in 0..num_moves {
            let m = moves[i];
            let mut score = 0i32;

            if tt_move >= 0 && m == tt_move as usize {
                score += 100_000;
            }

            if self.move_creates_tuzdyk(board, m) {
                score += 50_000;
            }

            if self.is_capture_move(board, m) {
                score += 10_000;
                let (landing_side, landing_pit) = self.predict_landing(board, m);
                let opp = board.side_to_move.opposite().index();
                if landing_side == opp {
                    score += board.pits[opp][landing_pit] as i32 * 100;
                }
            }

            if (ply as usize) < MAX_DEPTH as usize {
                let ply_idx = ply as usize;
                if self.killer_moves[ply_idx][0] == m as i8 {
                    score += 9_000;
                } else if self.killer_moves[ply_idx][1] == m as i8 {
                    score += 8_000;
                }
            }

            if prev_move_for_cm >= 0 {
                let opp = 1 - side;
                if self.countermove[opp][prev_move_for_cm as usize] == m as i8 {
                    score += 7_000;
                }
            }

            score += self.history[side][m];

            // Continuation history: how good is this move as a response to prev_move?
            if prev_move_for_cm >= 0 {
                let opp = 1 - side;
                score += self.cont_history[opp][prev_move_for_cm as usize][m] / 2;
            }

            score += board.pits[side][m] as i32;

            scores[i] = score;
        }
    }

    /// Clear state between games
    pub fn clear(&mut self) {
        self.tt.clear();
        self.killer_moves = [[-1; 2]; MAX_DEPTH as usize];
        self.history = [[0; NUM_PITS]; 2];
        self.countermove = [[-1; NUM_PITS]; 2];
        self.cont_history = [[[0; NUM_PITS]; NUM_PITS]; 2];
        self.game_history.clear();
    }
}

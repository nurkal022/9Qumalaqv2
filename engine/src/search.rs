/// Alpha-Beta search with iterative deepening
///
/// Features:
/// - Negamax alpha-beta with fail-soft
/// - Iterative deepening
/// - Transposition table
/// - Move ordering: TT move first, then by eval heuristic
/// - Null move pruning (R=2)
/// - Late Move Reductions (LMR)
/// - Killer moves

use std::time::Instant;
use crate::board::{Board, GameResult, NUM_PITS};
use crate::eval::{evaluate, EVAL_INF, EVAL_MATE};
use crate::nnue::NnueNetwork;
use crate::tt::{TranspositionTable, TTFlag};
use crate::zobrist::ZobristKeys;

const MAX_DEPTH: i32 = 64;
const NULL_MOVE_R: i32 = 2;
const LMR_THRESHOLD: usize = 3; // reduce moves after this many

pub struct SearchResult {
    pub best_move: usize,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
    pub tt_hits: u64,
    pub time_ms: u64,
}

pub struct Searcher {
    pub tt: TranspositionTable,
    pub zobrist: ZobristKeys,
    pub nodes: u64,
    pub max_time_ms: u64,
    pub silent: bool,
    start_time: Instant,
    stopped: bool,
    killer_moves: [[i8; 2]; MAX_DEPTH as usize], // [depth][slot]
    history: [[i32; NUM_PITS]; 2], // [side][pit] history heuristic
    nnue: Option<NnueNetwork>,
}

impl Searcher {
    pub fn new(tt_size_mb: usize) -> Self {
        Searcher {
            tt: TranspositionTable::new(tt_size_mb),
            zobrist: ZobristKeys::new(),
            nodes: 0,
            max_time_ms: 5000,
            silent: false,
            start_time: Instant::now(),
            stopped: false,
            killer_moves: [[-1; 2]; MAX_DEPTH as usize],
            history: [[0; NUM_PITS]; 2],
            nnue: None,
        }
    }

    pub fn set_nnue(&mut self, nnue: NnueNetwork) {
        self.nnue = Some(nnue);
    }

    /// Evaluate using NNUE if available, otherwise handcrafted
    fn eval(&self, board: &Board) -> i32 {
        if let Some(ref nnue) = self.nnue {
            nnue.evaluate(board)
        } else {
            evaluate(board)
        }
    }

    /// Search with iterative deepening
    pub fn search(&mut self, board: &Board, max_depth: i32, time_ms: u64) -> SearchResult {
        self.nodes = 0;
        self.max_time_ms = time_ms;
        self.start_time = Instant::now();
        self.stopped = false;
        self.killer_moves = [[-1; 2]; MAX_DEPTH as usize];

        #[allow(unused_assignments)]
        let mut best_move = 0usize;
        let mut best_score = -EVAL_INF;

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

        // Iterative deepening
        for depth in 1..=max_depth {
            let score = self.alpha_beta(board, depth, -EVAL_INF, EVAL_INF, 0);

            if self.stopped {
                break;
            }

            best_score = score;

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
        }

        let elapsed = self.start_time.elapsed().as_millis() as u64;

        SearchResult {
            best_move,
            score: best_score,
            depth: max_depth,
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

        // Leaf node
        if depth <= 0 {
            return self.quiescence(board, alpha, beta, ply);
        }

        // TT probe
        let hash = self.zobrist.hash(board);
        let mut tt_move: i8 = -1;

        if let Some(entry) = self.tt.probe(hash) {
            tt_move = entry.best_move;
            if entry.depth >= depth {
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

        // Null move pruning (skip if in endgame with few pieces)
        let total_stones: u16 = board.pits[board.side_to_move.index()]
            .iter()
            .map(|&x| x as u16)
            .sum();

        if depth >= 3 && total_stones > 10 && ply > 0 {
            let mut null_board = *board;
            null_board.side_to_move = null_board.side_to_move.opposite();
            null_board.move_count += 1;

            let null_score = -self.alpha_beta(&null_board, depth - 1 - NULL_MOVE_R, -beta, -beta + 1, ply + 1);

            if self.stopped {
                return 0;
            }

            if null_score >= beta {
                return null_score;
            }
        }

        // Generate and order moves
        let mut moves = [0usize; NUM_PITS];
        let num_moves = board.valid_moves_array(&mut moves);

        if num_moves == 0 {
            // No valid moves — evaluate
            return self.eval(board);
        }

        // Order moves
        let mut move_scores = [0i32; NUM_PITS];
        self.score_moves(board, &moves, num_moves, tt_move, ply, &mut move_scores);

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

        let mut best_score = -EVAL_INF;
        let mut best_move: i8 = moves[0] as i8;
        let mut flag = TTFlag::UpperBound;

        for i in 0..num_moves {
            let m = moves[i];
            let mut new_board = *board;
            let _undo = new_board.make_move(m);

            let score;

            // LMR: reduce depth for late quiet moves
            if i >= LMR_THRESHOLD && depth >= 3 {
                // Reduced search
                let reduced = -self.alpha_beta(&new_board, depth - 2, -alpha - 1, -alpha, ply + 1);
                if reduced > alpha {
                    // Re-search at full depth
                    score = -self.alpha_beta(&new_board, depth - 1, -beta, -alpha, ply + 1);
                } else {
                    score = reduced;
                }
            } else if i > 0 {
                // PVS: search with null window first
                let pv_score = -self.alpha_beta(&new_board, depth - 1, -alpha - 1, -alpha, ply + 1);
                if pv_score > alpha && pv_score < beta {
                    score = -self.alpha_beta(&new_board, depth - 1, -beta, -alpha, ply + 1);
                } else {
                    score = pv_score;
                }
            } else {
                score = -self.alpha_beta(&new_board, depth - 1, -beta, -alpha, ply + 1);
            }

            if self.stopped {
                return 0;
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
                        if (ply as usize) < MAX_DEPTH as usize {
                            let ply_idx = ply as usize;
                            if self.killer_moves[ply_idx][0] != m as i8 {
                                self.killer_moves[ply_idx][1] = self.killer_moves[ply_idx][0];
                                self.killer_moves[ply_idx][0] = m as i8;
                            }
                        }

                        // Update history
                        let side = board.side_to_move.index();
                        self.history[side][m] += depth * depth;

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
    fn quiescence(&mut self, board: &Board, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        // Time check
        if self.nodes & 4095 == 0 {
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

        let stand_pat = self.eval(board);

        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        // In quiescence, only search moves that capture
        let mut moves = [0usize; NUM_PITS];
        let num_moves = board.valid_moves_array(&mut moves);

        for i in 0..num_moves {
            let m = moves[i];

            // Check if this move would result in a capture
            if !self.is_capture_move(board, m) {
                continue;
            }

            let mut new_board = *board;
            let _undo = new_board.make_move(m);

            let score = -self.quiescence(&new_board, -beta, -alpha, ply + 1);

            if self.stopped {
                return 0;
            }

            if score >= beta {
                return score;
            }
            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }

    /// Check if a move would result in a capture (approximate)
    fn is_capture_move(&self, board: &Board, pit: usize) -> bool {
        let me = board.side_to_move.index();
        let opp = board.side_to_move.opposite().index();
        let stones = board.pits[me][pit];

        if stones == 0 {
            return false;
        }

        // Predict landing position
        let (landing_side, landing_pit) = if stones == 1 {
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
        };

        // Check if landing on opponent's side would capture
        if landing_side == opp {
            let target = board.pits[opp][landing_pit] + 1; // +1 for the stone we'll add
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
        scores: &mut [i32; NUM_PITS],
    ) {
        let side = board.side_to_move.index();

        for i in 0..num_moves {
            let m = moves[i];
            let mut score = 0i32;

            // TT move gets highest priority
            if tt_move >= 0 && m == tt_move as usize {
                score += 100_000;
            }

            // Killer moves
            if (ply as usize) < MAX_DEPTH as usize {
                let ply_idx = ply as usize;
                if self.killer_moves[ply_idx][0] == m as i8 {
                    score += 9_000;
                } else if self.killer_moves[ply_idx][1] == m as i8 {
                    score += 8_000;
                }
            }

            // Capture heuristic
            if self.is_capture_move(board, m) {
                score += 10_000;
            }

            // History heuristic
            score += self.history[side][m];

            // Prefer moves with more stones (create longer distributions)
            score += board.pits[side][m] as i32;

            scores[i] = score;
        }
    }

    /// Clear state between games
    pub fn clear(&mut self) {
        self.tt.clear();
        self.killer_moves = [[-1; 2]; MAX_DEPTH as usize];
        self.history = [[0; NUM_PITS]; 2];
    }
}

/// Evaluate model vs engine using NN policy + value-guided move selection.
/// For each position: evaluate with NN, pick best move by combined policy+value score.

use crate::board::{Board, Side, NUM_PITS};
use crate::encoding::{encode_state, ENCODED_SIZE};
use crate::evaluator::{EvalRequest, EvalResponse};
use crate::mcts;
use crossbeam_channel::{Sender, Receiver};
use std::io::Write;

pub fn board_to_position_string(board: &Board) -> String {
    let white: Vec<String> = (0..NUM_PITS).map(|i| board.pits[0][i].to_string()).collect();
    let black: Vec<String> = (0..NUM_PITS).map(|i| board.pits[1][i].to_string()).collect();
    format!(
        "{}/{}/{},{}/{},{}/{}",
        white.join(","),
        black.join(","),
        board.kazan[0], board.kazan[1],
        board.tuzdyk[0], board.tuzdyk[1],
        board.side_to_move.index(),
    )
}

#[derive(Debug, Clone, Copy)]
pub enum GameOutcome {
    MctsWin,
    MctsLoss,
    Draw,
}

/// Eval context with reusable response channel
pub struct EvalContext {
    pub eval_tx: Sender<EvalRequest>,
    resp_tx: Sender<EvalResponse>,
    resp_rx: Receiver<EvalResponse>,
}

impl EvalContext {
    pub fn new(eval_tx: Sender<EvalRequest>) -> Self {
        let (resp_tx, resp_rx) = crossbeam_channel::bounded(256);
        EvalContext { eval_tx, resp_tx, resp_rx }
    }

    fn evaluate_batch(&self, states: &[[f32; ENCODED_SIZE]]) -> Vec<EvalResponse> {
        let n = states.len();
        for i in 0..n {
            let req = EvalRequest {
                encoded_state: states[i],
                response_tx: self.resp_tx.clone(),
            };
            self.eval_tx.send(req).expect("Evaluator closed");
        }
        let mut responses = Vec::with_capacity(n);
        for _ in 0..n {
            responses.push(self.resp_rx.recv().expect("Response closed"));
        }
        responses
    }

    /// Pick best move: policy-guided with 1-ply value refinement.
    /// Policy head is primary signal. Value used only to reject clearly bad moves.
    /// Deep MCTS (PUCT tree search) — uses num_sims simulations
    pub fn best_move_deep(&self, board: &Board, num_sims: u32) -> usize {
        let config = mcts::MctsConfig {
            num_simulations: num_sims,
            c_puct: 2.5,
            dirichlet_alpha: 0.0,
            dirichlet_epsilon: 0.0,
            temperature_threshold: 0,
            virtual_batch: std::cmp::min(num_sims as usize, 64),
        };
        let mcts_ctx = mcts::EvalContext::new(self.eval_tx.clone());
        let (policy, _value) = mcts::search(board, &config, &mcts_ctx, false);

        // Greedy: pick highest visit count
        let mut moves = [0usize; NUM_PITS];
        let n = board.valid_moves_array(&mut moves);
        if n == 0 { return 0; }
        *moves[..n].iter().max_by(|a, b| {
            policy[**a].partial_cmp(&policy[**b]).unwrap()
        }).unwrap()
    }

    /// 1-ply lookahead: policy-guided with value refinement
    pub fn best_move(&self, board: &Board) -> usize {
        let mut moves = [0usize; NUM_PITS];
        let n = board.valid_moves_array(&mut moves);
        if n == 0 { return 0; }
        if n == 1 { return moves[0]; }

        // Get root policy + value
        let root_enc = encode_state(board);
        let root_resp = self.evaluate_batch(&[root_enc]);
        let root_policy = root_resp[0].policy;

        // Batch evaluate ALL children for value check
        let mut child_states: Vec<[f32; ENCODED_SIZE]> = Vec::with_capacity(n);
        let mut terminal_values: Vec<Option<f32>> = Vec::with_capacity(n);

        for i in 0..n {
            let mut child = *board;
            child.make_move(moves[i]);

            if let Some(result) = child.game_result() {
                let val = match result {
                    crate::board::GameResult::Win(side) => {
                        if side == board.side_to_move { 1.0 } else { -1.0 }
                    }
                    crate::board::GameResult::Draw => 0.0,
                };
                terminal_values.push(Some(val));
                child_states.push(encode_state(&child));
            } else {
                terminal_values.push(None);
                child_states.push(encode_state(&child));
            }
        }

        let child_responses = self.evaluate_batch(&child_states);

        // Score: policy (dominant) + small value bonus
        // Policy is the trained signal (p_loss=1.09), value is secondary
        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;

        for i in 0..n {
            // Check for instant win
            if let Some(tv) = terminal_values[i] {
                if tv > 0.99 { return moves[i]; }
            }

            let parent_val = if let Some(tv) = terminal_values[i] {
                tv as f64
            } else {
                -(child_responses[i].value as f64)
            };

            let prior = root_policy[moves[i]] as f64;

            // Policy-dominant scoring: policy * 3 + value * 1
            // This follows the trained policy while using value to avoid blunders
            let score = prior * 3.0 + parent_val.max(-0.5);

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        moves[best_idx]
    }
}

/// Play one game: NN 1-ply lookahead vs engine
pub fn play_eval_game(
    ctx: &EvalContext,
    _num_sims: u32,  // unused for now, kept for API compat
    engine_stdin: &mut dyn Write,
    engine_stdout: &mut dyn std::io::BufRead,
    mcts_is_white: bool,
    time_ms: u64,
) -> GameOutcome {
    let mut board = Board::new();
    let mut move_count = 0u32;

    writeln!(engine_stdin, "newgame").unwrap();
    engine_stdin.flush().unwrap();
    let _ = read_line(engine_stdout);

    while board.game_result().is_none() && move_count < 200 {
        let is_nn_turn = (board.side_to_move == Side::White) == mcts_is_white;

        if is_nn_turn {
            let action = if _num_sims > 1 {
                ctx.best_move_deep(&board, _num_sims)
            } else {
                ctx.best_move(&board)
            };
            board.make_move(action);
        } else {
            let pos_str = board_to_position_string(&board);
            writeln!(engine_stdin, "go pos {} time {}", pos_str, time_ms).unwrap();
            engine_stdin.flush().unwrap();

            let response = read_line(engine_stdout);
            let action = parse_engine_response(&response);

            if action < 0 {
                return GameOutcome::MctsWin;
            }
            let action = action as usize;
            if board.is_valid_move(action) {
                board.make_move(action);
            } else {
                return GameOutcome::MctsWin;
            }
        }
        move_count += 1;
    }

    let nn_side = if mcts_is_white { Side::White } else { Side::Black };
    match board.game_result() {
        Some(crate::board::GameResult::Win(side)) => {
            if side == nn_side { GameOutcome::MctsWin } else { GameOutcome::MctsLoss }
        }
        Some(crate::board::GameResult::Draw) | None => GameOutcome::Draw,
    }
}

fn read_line(reader: &mut dyn std::io::BufRead) -> String {
    let mut line = String::new();
    let _ = reader.read_line(&mut line);
    line.trim().to_string()
}

fn parse_engine_response(response: &str) -> i32 {
    if response.starts_with("bestmove") {
        let parts: Vec<&str> = response.split_whitespace().collect();
        if parts.len() >= 2 {
            return parts[1].parse().unwrap_or(-1);
        }
    }
    -1
}

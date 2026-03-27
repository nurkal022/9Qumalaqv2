/// Evaluate model vs engine using Gumbel MCTS (Sequential Halving).
/// Matches Python ConfigurableMCTS: 1-ply lookahead with multiple iterations.

use crate::board::{Board, Side, NUM_PITS};
use crate::encoding::{encode_state, ENCODED_SIZE};
use crate::evaluator::{EvalRequest, EvalResponse};
use crossbeam_channel::{Sender, Receiver};
use rand::Rng;
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

    /// Gumbel MCTS: Sequential Halving with 1-ply lookahead.
    /// N simulations distributed across valid moves with halving.
    fn gumbel_best_move(&self, board: &Board, num_sims: u32) -> usize {
        let mut moves = [0usize; NUM_PITS];
        let n = board.valid_moves_array(&mut moves);
        if n == 0 { return 0; }
        if n == 1 { return moves[0]; }

        let valid_moves = &moves[..n];

        // Get root evaluation for log-priors
        let root_enc = encode_state(board);
        let root_resp = self.evaluate_batch(&[root_enc]);
        let root_policy = root_resp[0].policy;
        let root_value = root_resp[0].value;

        // Debug first few calls
        static CALL_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let cc = CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if cc < 4 {
            eprintln!("  [gumbel] root_value={:.3} policy={:?} valid={:?}",
                root_value,
                root_policy.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>(),
                valid_moves);
            // Evaluate each child
            for &m in valid_moves {
                let mut child = *board;
                child.make_move(m);
                let child_enc = encode_state(&child);
                let child_resp = self.evaluate_batch(&[child_enc]);
                eprintln!("    move {}: child_value={:.3} → parent_value={:.3}",
                    m, child_resp[0].value, -child_resp[0].value);
            }
        }

        // Log-priors (masked)
        let mut logits = [f32::NEG_INFINITY; 9];
        for &m in valid_moves {
            logits[m] = root_policy[m].max(1e-8).ln();
        }

        // Sample Gumbel noise
        let mut rng = rand::thread_rng();
        let mut gumbel = [0.0f32; 9];
        for &m in valid_moves {
            // Gumbel(0,1) = -ln(-ln(U)), U ~ Uniform(0,1)
            let u: f32 = rng.gen::<f32>().max(1e-10).min(1.0 - 1e-10);
            gumbel[m] = -((-u.ln()).ln());
        }

        // Initialize Q-values with root value
        let mut q_values = [root_value; 9];
        let mut q_sum = [0.0f64; 9];
        let mut visit_counts = [0u32; 9];

        // Sequential Halving
        let mut remaining: Vec<usize> = valid_moves.to_vec();
        let mut remaining_sims = num_sims;
        let num_phases = (remaining.len() as f64).log2().ceil().max(1.0) as u32;

        for phase in 0..num_phases {
            if remaining.len() <= 1 || remaining_sims == 0 { break; }

            let phases_left = num_phases.saturating_sub(phase).max(1);
            let sims_per_action = (remaining_sims / (remaining.len() as u32 * phases_left)).max(1);

            for &action in &remaining {
                for _ in 0..sims_per_action {
                    if remaining_sims == 0 { break; }

                    let mut child = *board;
                    child.make_move(action);

                    let child_value = if let Some(result) = child.game_result() {
                        match result {
                            crate::board::GameResult::Win(side) => {
                                if side == board.side_to_move { 1.0 } else { -1.0 }
                            }
                            crate::board::GameResult::Draw => 0.0,
                        }
                    } else {
                        let child_enc = encode_state(&child);
                        let resp = self.evaluate_batch(&[child_enc]);
                        // Negate: child value is from child's (opponent's) perspective
                        -resp[0].value
                    };

                    visit_counts[action] += 1;
                    q_sum[action] += child_value as f64;
                    q_values[action] = (q_sum[action] / visit_counts[action] as f64) as f32;
                    remaining_sims = remaining_sims.saturating_sub(1);
                }
            }

            // Discard bottom half by score = gumbel + logit + sigma(q)
            if remaining.len() > 1 {
                let c_visit = 50.0f32; // sigma scaling (from Python)
                let max_visit = *visit_counts.iter().max().unwrap_or(&1) as f32;
                remaining.sort_by(|&a, &b| {
                    let sa = gumbel[a] + logits[a] + c_visit * q_values[a];
                    let sb = gumbel[b] + logits[b] + c_visit * q_values[b];
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
                let half = (remaining.len() / 2).max(1);
                remaining.truncate(half);
            }
        }

        // Return best remaining action
        if remaining.len() == 1 {
            return remaining[0];
        }

        // Final: pick by improved logit + sigma(q)
        let c_visit = 50.0f32;
        let mut best = remaining[0];
        let mut best_score = f32::NEG_INFINITY;
        for &a in &remaining {
            let score = logits[a] + c_visit * q_values[a];
            if score > best_score {
                best_score = score;
                best = a;
            }
        }
        best
    }
}

/// Play one game: Gumbel MCTS vs engine
pub fn play_eval_game(
    ctx: &EvalContext,
    num_sims: u32,
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
            let action = ctx.gumbel_best_move(&board, num_sims);
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
    eprintln!("  Game done: {} moves, kazan={},{}, nn_side={:?}, result={:?}",
        move_count, board.kazan[0], board.kazan[1], nn_side, board.game_result());
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

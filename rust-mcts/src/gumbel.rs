/// Gumbel MCTS for Togyz Kumalak.
///
/// Instead of deep tree search (PUCT + virtual loss), uses 1-ply lookahead
/// with Gumbel noise for policy improvement. Much simpler and matches
/// the Python ConfigurableMCTS / TrueBatchMCTS.
///
/// For each position:
/// 1. Evaluate root → policy priors + root value
/// 2. Evaluate all children (1-ply) → child values
/// 3. Improved policy = softmax(log_prior + c_visit * (-child_value))
/// 4. Action = argmax(improved_policy + gumbel_noise)  [selfplay]
///          or argmax(improved_policy)                   [eval/greedy]

use crate::board::{Board, GameResult, Side, NUM_PITS};
use crate::encoding::{encode_state, ENCODED_SIZE};
use crate::evaluator::{EvalRequest, EvalResponse};
use crossbeam_channel::{Sender, Receiver};
use rand::Rng;

const C_VISIT: f32 = 50.0; // sigma scaling (matches Python)

/// Reusable eval context for a worker thread
pub struct GumbelContext {
    pub eval_tx: Sender<EvalRequest>,
    resp_tx: Sender<EvalResponse>,
    resp_rx: Receiver<EvalResponse>,
}

impl GumbelContext {
    pub fn new(eval_tx: Sender<EvalRequest>) -> Self {
        let (resp_tx, resp_rx) = crossbeam_channel::bounded(256);
        GumbelContext { eval_tx, resp_tx, resp_rx }
    }

    /// Evaluate multiple positions in one batch
    fn evaluate_batch(&self, states: &[[f32; ENCODED_SIZE]]) -> Vec<EvalResponse> {
        let n = states.len();
        if n == 0 { return Vec::new(); }
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
}

/// Result of Gumbel search for one position
pub struct GumbelResult {
    pub improved_policy: [f32; 9],  // training target
    pub root_value: f32,
    pub action: usize,              // selected move
}

/// Full Gumbel search: evaluate root + all children, compute improved policy.
/// Used for 25% of moves (playout cap: full search).
pub fn gumbel_search(
    board: &Board,
    ctx: &GumbelContext,
    add_noise: bool,
    temperature: f32,
) -> GumbelResult {
    let mut moves = [0usize; NUM_PITS];
    let n = board.valid_moves_array(&mut moves);
    let valid_moves = &moves[..n];

    // Terminal or single move
    if n == 0 {
        return GumbelResult {
            improved_policy: [0.0; 9],
            root_value: 0.0,
            action: 0,
        };
    }
    if n == 1 {
        let mut policy = [0.0f32; 9];
        policy[valid_moves[0]] = 1.0;
        // Evaluate root for value
        let root_enc = encode_state(board);
        let root_resp = ctx.evaluate_batch(&[root_enc]);
        return GumbelResult {
            improved_policy: policy,
            root_value: root_resp[0].value,
            action: valid_moves[0],
        };
    }

    // Step 1: Evaluate root
    let root_enc = encode_state(board);
    let root_resp = ctx.evaluate_batch(&[root_enc]);
    let root_policy = root_resp[0].policy;
    let root_value = root_resp[0].value;

    // Step 2: Evaluate ALL children in one batch
    let mut child_states: Vec<[f32; ENCODED_SIZE]> = Vec::with_capacity(n);
    let mut terminal_values: Vec<Option<f32>> = Vec::with_capacity(n);

    for i in 0..n {
        let mut child = *board;
        child.make_move(valid_moves[i]);

        if let Some(result) = child.game_result() {
            let val = match result {
                GameResult::Win(side) => {
                    if side == board.side_to_move { 1.0 } else { -1.0 }
                }
                GameResult::Draw => 0.0,
            };
            terminal_values.push(Some(val));
            // Placeholder encoding (won't be used, but needed for batch alignment)
            child_states.push(encode_state(&child));
        } else {
            terminal_values.push(None);
            child_states.push(encode_state(&child));
        }
    }

    let child_responses = ctx.evaluate_batch(&child_states);

    // Step 3: Compute parent-perspective values for each move
    let mut parent_values = [0.0f32; 9];
    for i in 0..n {
        let m = valid_moves[i];
        parent_values[m] = if let Some(tv) = terminal_values[i] {
            tv
        } else {
            -child_responses[i].value // negate: child perspective → parent perspective
        };
    }

    // Step 4: Compute improved policy
    // improved_logit(a) = log_prior(a) + c_visit * parent_value(a)
    let mut improved_logits = [f32::NEG_INFINITY; 9];
    for i in 0..n {
        let m = valid_moves[i];
        let log_prior = root_policy[m].max(1e-8).ln();
        improved_logits[m] = log_prior + C_VISIT * parent_values[m];
    }

    // Softmax to get improved policy
    let max_logit = improved_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut improved_policy = [0.0f32; 9];
    let mut sum = 0.0f32;
    for i in 0..9 {
        if improved_logits[i] > f32::NEG_INFINITY + 1.0 {
            improved_policy[i] = (improved_logits[i] - max_logit).exp();
            sum += improved_policy[i];
        }
    }
    if sum > 0.0 {
        for p in improved_policy.iter_mut() { *p /= sum; }
    }

    // Step 5: Select action
    let action = if temperature < 0.01 {
        // Greedy: argmax of improved policy (no noise)
        let mut best = valid_moves[0];
        let mut best_p = improved_policy[valid_moves[0]];
        for &m in &valid_moves[1..] {
            if improved_policy[m] > best_p {
                best_p = improved_policy[m];
                best = m;
            }
        }
        best
    } else if add_noise {
        // Gumbel noise for exploration (selfplay)
        let mut rng = rand::thread_rng();
        let mut best = valid_moves[0];
        let mut best_score = f32::NEG_INFINITY;
        for &m in valid_moves {
            let u: f32 = rng.gen::<f32>().max(1e-10).min(1.0 - 1e-10);
            let gumbel = -((-u.ln()).ln());
            let score = improved_logits[m] + gumbel;
            if score > best_score {
                best_score = score;
                best = m;
            }
        }
        best
    } else {
        // Temperature-based sampling (no Gumbel)
        sample_with_temperature(&improved_policy, valid_moves, temperature)
    };

    GumbelResult {
        improved_policy,
        root_value,
        action,
    }
}

/// Fast move: just use raw NN policy (no children evaluation).
/// Used for 75% of moves (playout cap: fast).
pub fn fast_move(
    board: &Board,
    ctx: &GumbelContext,
    temperature: f32,
) -> GumbelResult {
    let mut moves = [0usize; NUM_PITS];
    let n = board.valid_moves_array(&mut moves);
    let valid_moves = &moves[..n];

    if n == 0 {
        return GumbelResult {
            improved_policy: [0.0; 9],
            root_value: 0.0,
            action: 0,
        };
    }

    // Just evaluate root
    let root_enc = encode_state(board);
    let root_resp = ctx.evaluate_batch(&[root_enc]);
    let root_policy = root_resp[0].policy;
    let root_value = root_resp[0].value;

    let action = if temperature < 0.01 {
        *valid_moves.iter().max_by(|a, b| root_policy[**a].partial_cmp(&root_policy[**b]).unwrap()).unwrap()
    } else {
        sample_with_temperature(&root_policy, valid_moves, temperature)
    };

    GumbelResult {
        improved_policy: [0.0; 9], // zero = marker for value-only training
        root_value,
        action,
    }
}

fn sample_with_temperature(policy: &[f32; 9], valid_moves: &[usize], temperature: f32) -> usize {
    let mut rng = rand::thread_rng();
    if valid_moves.is_empty() { return 0; }

    let inv_t = 1.0 / temperature;
    let mut probs = [0.0f64; 9];
    let mut sum = 0.0f64;
    for &m in valid_moves {
        let p = (policy[m] as f64).powf(inv_t as f64);
        probs[m] = p;
        sum += p;
    }
    if sum < 1e-10 {
        return valid_moves[rng.gen_range(0..valid_moves.len())];
    }
    for p in probs.iter_mut() { *p /= sum; }

    let r: f64 = rng.gen();
    let mut c = 0.0;
    for &m in valid_moves {
        c += probs[m];
        if r < c { return m; }
    }
    *valid_moves.last().unwrap()
}

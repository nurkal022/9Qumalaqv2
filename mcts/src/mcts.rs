/// MCTS with PUCT selection for Togyz Kumalak.
///
/// Batch-parallel: collects N leaves per GPU call using virtual loss.
/// Instead of 800 sequential eval calls, does ~7 batched calls of 128.

use crate::board::{Board, GameResult, Side, NUM_PITS};
use crate::encoding::{encode_state, ENCODED_SIZE};
use crate::evaluator::{EvalRequest, EvalResponse};
use crossbeam_channel::{Sender, Receiver};
use rand::Rng;
use rand_distr::{Dirichlet, Distribution};

#[derive(Clone)]
pub struct MctsConfig {
    pub num_simulations: u32,
    pub c_puct: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    pub temperature_threshold: u32,
    pub virtual_batch: usize, // leaves per GPU call (e.g., 128)
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            num_simulations: 800,
            c_puct: 1.5,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            temperature_threshold: 15,
            virtual_batch: 128,
        }
    }
}

/// Worker-local eval context with reusable response channel
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

    /// Send multiple positions, wait for all responses (in order)
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
}

// ── Tree structures ─────────────────────────────────────

pub struct MctsEdge {
    pub action: u8,
    pub prior: f32,
    pub child: Option<Box<MctsNode>>,
    pub visit_count: u32,
    pub value_sum: f64,
}

impl MctsEdge {
    #[inline]
    fn q_value(&self) -> f64 {
        if self.visit_count == 0 { 0.0 }
        else { self.value_sum / self.visit_count as f64 }
    }
}

pub struct MctsNode {
    pub edges: Vec<MctsEdge>,
    pub visit_count: u32,
    pub is_terminal: bool,
    pub terminal_value: f32,
}

impl MctsNode {
    fn new_leaf() -> Self {
        MctsNode { edges: Vec::new(), visit_count: 0, is_terminal: false, terminal_value: 0.0 }
    }
    fn new_terminal(value: f32) -> Self {
        MctsNode { edges: Vec::new(), visit_count: 1, is_terminal: true, terminal_value: value }
    }
    fn is_expanded(&self) -> bool {
        !self.edges.is_empty() || self.is_terminal
    }
    fn select_child(&self, c_puct: f32) -> usize {
        let sqrt_n = (self.visit_count as f32).sqrt();
        let mut best = 0;
        let mut best_s = f64::NEG_INFINITY;
        for (i, e) in self.edges.iter().enumerate() {
            let s = e.q_value() + c_puct as f64 * e.prior as f64 * sqrt_n as f64 / (1.0 + e.visit_count as f64);
            if s > best_s { best_s = s; best = i; }
        }
        best
    }
}

// ── Pending leaf info ───────────────────────────────────

struct PendingLeaf {
    path: Vec<usize>,       // edge indices from root to parent of leaf
    encoded: [f32; ENCODED_SIZE],
    board: Board,           // board state at leaf (for expansion)
}

// ── Main search function (batch-parallel) ───────────────

pub fn search(
    board: &Board,
    config: &MctsConfig,
    ctx: &EvalContext,
    add_noise: bool,
) -> ([f32; 9], f32) {
    // Terminal check
    if let Some(result) = board.game_result() {
        return ([0.0; 9], terminal_value(result, board.side_to_move));
    }

    // Evaluate root
    let root_enc = encode_state(board);
    let root_resp = {
        let req = EvalRequest {
            encoded_state: root_enc,
            response_tx: ctx.resp_tx.clone(),
        };
        ctx.eval_tx.send(req).unwrap();
        ctx.resp_rx.recv().unwrap()
    };

    let mut root = MctsNode::new_leaf();
    expand_node(&mut root, board, &root_resp.policy);

    if add_noise {
        add_dirichlet_noise(&mut root, config.dirichlet_alpha, config.dirichlet_epsilon);
    }

    // Run simulations in batches
    let mut sims_done: u32 = 0;
    while sims_done < config.num_simulations {
        let batch_target = std::cmp::min(
            config.virtual_batch,
            (config.num_simulations - sims_done) as usize,
        );

        // Phase 1: Collect leaves by traversing tree with virtual loss
        let mut pending_leaves: Vec<PendingLeaf> = Vec::with_capacity(batch_target);
        let mut terminal_paths: Vec<(Vec<usize>, f64)> = Vec::new();

        for _ in 0..batch_target {
            collect_one_leaf(&mut root, board, config, &mut pending_leaves, &mut terminal_paths);
        }

        // Phase 2: Batch evaluate all leaves in ONE GPU call
        if !pending_leaves.is_empty() {
            let encoded: Vec<[f32; ENCODED_SIZE]> = pending_leaves.iter().map(|p| p.encoded).collect();
            let responses = ctx.evaluate_batch(&encoded);

            // Phase 3: Expand leaves and undo virtual loss + backpropagate
            for (leaf, resp) in pending_leaves.into_iter().zip(responses.into_iter()) {
                // Expand the leaf node
                let node = navigate_to_node(&mut root, &leaf.path);
                if !node.is_expanded() {
                    expand_node(node, &leaf.board, &resp.policy);
                }
                // Undo virtual loss and apply real value
                undo_virtual_loss_and_backprop(&mut root, &leaf.path, resp.value as f64);
            }
        }

        // Handle terminal nodes
        for (path, value) in terminal_paths {
            undo_virtual_loss_and_backprop(&mut root, &path, value);
        }

        sims_done += batch_target as u32;
    }

    // Extract policy from visit counts
    let mut policy = [0.0f32; 9];
    let mut total = 0u32;
    for e in &root.edges {
        policy[e.action as usize] = e.visit_count as f32;
        total += e.visit_count;
    }
    if total > 0 {
        for p in policy.iter_mut() { *p /= total as f32; }
    }

    let root_val = if root.visit_count > 0 {
        root.edges.iter().map(|e| e.value_sum).sum::<f64>() / root.visit_count as f64
    } else { 0.0 };

    (policy, root_val as f32)
}

/// Traverse tree to find one leaf, applying virtual loss along the way.
fn collect_one_leaf(
    root: &mut MctsNode,
    root_board: &Board,
    config: &MctsConfig,
    pending: &mut Vec<PendingLeaf>,
    terminals: &mut Vec<(Vec<usize>, f64)>,
) {
    let mut path: Vec<usize> = Vec::with_capacity(16);
    let mut board = *root_board;
    let mut node = root as *mut MctsNode;

    unsafe {
        loop {
            let n = &mut *node;

            if n.is_terminal {
                terminals.push((path, n.terminal_value as f64));
                return;
            }

            if !n.is_expanded() {
                // Unexpanded leaf — need NN evaluation
                pending.push(PendingLeaf {
                    path,
                    encoded: encode_state(&board),
                    board,
                });
                return;
            }

            // Select best child (PUCT)
            let idx = n.select_child(config.c_puct);
            path.push(idx);

            // Apply virtual loss: +1 visit, -1 value (pessimistic)
            n.edges[idx].visit_count += 1;
            n.edges[idx].value_sum -= 1.0;
            n.visit_count += 1;

            let action = n.edges[idx].action as usize;
            board.make_move(action);

            match &mut n.edges[idx].child {
                Some(child) => {
                    node = child.as_mut() as *mut MctsNode;
                }
                None => {
                    // No child yet — check terminal or queue for eval
                    if let Some(result) = board.game_result() {
                        let val = terminal_value(result, board.side_to_move.opposite());
                        n.edges[idx].child = Some(Box::new(MctsNode::new_terminal(val)));
                        terminals.push((path, val as f64));
                    } else {
                        n.edges[idx].child = Some(Box::new(MctsNode::new_leaf()));
                        pending.push(PendingLeaf {
                            path,
                            encoded: encode_state(&board),
                            board,
                        });
                    }
                    return;
                }
            }
        }
    }
}

/// Navigate from root to a node following edge indices
fn navigate_to_node<'a>(root: &'a mut MctsNode, path: &[usize]) -> &'a mut MctsNode {
    let mut node = root;
    for &idx in path {
        node = node.edges[idx].child.as_mut().expect("Missing child in path");
    }
    node
}

/// Undo virtual loss (-1 visit, +1 value) and apply real backpropagation
fn undo_virtual_loss_and_backprop(root: &mut MctsNode, path: &[usize], leaf_value: f64) {
    let mut node = root;
    // leaf_value is from LEAF's side-to-move perspective.
    // Path goes root→leaf (top-down). At path[0] (root's edge):
    //   odd path length  → root opposite side from leaf → start with -leaf_value
    //   even path length → root same side as leaf       → start with +leaf_value
    let mut value = if path.len() % 2 == 0 { leaf_value } else { -leaf_value };

    for &idx in path {
        // Undo virtual loss (-1.0) and apply real value
        node.edges[idx].value_sum += value + 1.0;

        value = -value;
        node = node.edges[idx].child.as_mut().expect("Missing child");
    }
    node.visit_count += 1;
}

// ── Helpers ─────────────────────────────────────────────

fn terminal_value(result: GameResult, side_to_move: Side) -> f32 {
    match result {
        GameResult::Win(side) => if side == side_to_move { 1.0 } else { -1.0 },
        GameResult::Draw => 0.0,
    }
}

fn expand_node(node: &mut MctsNode, board: &Board, policy: &[f32; 9]) {
    let mut moves = [0usize; NUM_PITS];
    let n = board.valid_moves_array(&mut moves);

    let mut sum = 0.0f32;
    for i in 0..n { sum += policy[moves[i]]; }

    node.edges.clear();
    node.edges.reserve(n);
    for i in 0..n {
        let prior = if sum < 1e-8 { 1.0 / n as f32 } else { policy[moves[i]] / sum };
        node.edges.push(MctsEdge {
            action: moves[i] as u8,
            prior,
            child: None,
            visit_count: 0,
            value_sum: 0.0,
        });
    }
}

fn add_dirichlet_noise(root: &mut MctsNode, alpha: f32, epsilon: f32) {
    let n = root.edges.len();
    if n == 0 { return; }
    let mut rng = rand::thread_rng();
    if let Ok(d) = Dirichlet::new_with_size(alpha as f64, n) {
        let noise: Vec<f64> = d.sample(&mut rng);
        for (i, e) in root.edges.iter_mut().enumerate() {
            e.prior = (1.0 - epsilon) * e.prior + epsilon * noise[i] as f32;
        }
    }
}

pub fn select_action(policy: &[f32; 9], temperature: f32, valid_moves: &[usize]) -> usize {
    if valid_moves.is_empty() { return 0; }
    if temperature < 0.01 {
        return *valid_moves.iter().max_by(|a, b| policy[**a].partial_cmp(&policy[**b]).unwrap()).unwrap();
    }
    let mut rng = rand::thread_rng();
    let inv_t = 1.0 / temperature;
    let mut probs = [0.0f64; 9];
    let mut sum = 0.0f64;
    for &m in valid_moves {
        let p = (policy[m] as f64).powf(inv_t as f64);
        probs[m] = p; sum += p;
    }
    if sum < 1e-10 { return valid_moves[rng.gen_range(0..valid_moves.len())]; }
    for p in probs.iter_mut() { *p /= sum; }
    let r: f64 = rng.gen();
    let mut c = 0.0;
    for &m in valid_moves { c += probs[m]; if r < c { return m; } }
    *valid_moves.last().unwrap()
}

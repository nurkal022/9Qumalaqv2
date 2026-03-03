/// Monte Carlo Tree Search (MCTS) with PUCT exploration.
///
/// Uses safe recursive descent for tree traversal (no raw pointers).

use crate::board::Board;
use crate::features::board_to_features;
use crate::network::{Network, NUM_ACTIONS};

const C_PUCT: f32 = 1.5;

#[derive(Clone)]
pub struct Node {
    pub mov: Option<usize>,
    pub prior: f32,
    pub visit_count: u32,
    pub total_value: f32,
    pub children: Vec<Node>,
    pub is_expanded: bool,
}

impl Node {
    pub fn new(mov: Option<usize>, prior: f32) -> Self {
        Self {
            mov,
            prior,
            visit_count: 0,
            total_value: 0.0,
            children: Vec::new(),
            is_expanded: false,
        }
    }

    #[inline]
    pub fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f32
        }
    }

    #[inline]
    pub fn puct_score(&self, parent_visits: u32) -> f32 {
        let exploration =
            C_PUCT * self.prior * (parent_visits as f32).sqrt() / (1.0 + self.visit_count as f32);
        self.q_value() + exploration
    }
}

pub fn expand_node(node: &mut Node, board: &Board, policy_logits: &[f32; NUM_ACTIONS]) {
    let legal_moves = board.legal_moves();

    if legal_moves.is_empty() {
        node.is_expanded = true;
        return;
    }

    let max_logit = legal_moves
        .iter()
        .map(|&m| policy_logits[m])
        .fold(f32::NEG_INFINITY, f32::max);

    let exp_sum: f32 = legal_moves
        .iter()
        .map(|&m| (policy_logits[m] - max_logit).exp())
        .sum();

    for &mov in &legal_moves {
        let prior = (policy_logits[mov] - max_logit).exp() / exp_sum;
        node.children.push(Node::new(Some(mov), prior));
    }

    node.is_expanded = true;
}

fn select_child_index(node: &Node) -> usize {
    let parent_visits = node.visit_count;
    node.children
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.puct_score(parent_visits)
                .partial_cmp(&b.puct_score(parent_visits))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
        .0
}

pub fn mcts_search(
    board: &Board,
    network: &Network,
    num_simulations: u32,
) -> (Vec<(usize, u32)>, f32) {
    let mut root = Node::new(None, 1.0);

    let features = board_to_features(board);
    let (logits, root_value) = network.predict(&features);
    expand_node(&mut root, board, &logits);
    root.visit_count = 1;
    root.total_value = root_value;

    for _ in 0..num_simulations {
        let mut sim_board = board.clone();
        simulate_nn(&mut root, &mut sim_board, network);
    }

    let move_visits: Vec<(usize, u32)> = root
        .children
        .iter()
        .map(|c| (c.mov.unwrap(), c.visit_count))
        .collect();

    let avg_value = if root.visit_count > 0 {
        root.total_value / root.visit_count as f32
    } else {
        0.0
    };

    (move_visits, avg_value)
}

pub fn mcts_search_random(
    board: &Board,
    num_simulations: u32,
) -> (Vec<(usize, u32)>, f32) {
    let mut root = Node::new(None, 1.0);

    let uniform_logits = [0.0f32; NUM_ACTIONS];
    expand_node(&mut root, board, &uniform_logits);
    root.visit_count = 1;
    root.total_value = 0.0;

    for _ in 0..num_simulations {
        let mut sim_board = board.clone();
        simulate_random(&mut root, &mut sim_board);
    }

    let move_visits: Vec<(usize, u32)> = root
        .children
        .iter()
        .map(|c| (c.mov.unwrap(), c.visit_count))
        .collect();

    let avg_value = if root.visit_count > 0 {
        root.total_value / root.visit_count as f32
    } else {
        0.0
    };

    (move_visits, avg_value)
}

/// Recursive simulation with neural network evaluation.
/// Returns the value from the perspective of the side to move at this node.
fn simulate_nn(node: &mut Node, board: &mut Board, network: &Network) -> f32 {
    if board.is_terminal() {
        let value = board.outcome_for_side(board.side_to_move_u8());
        node.visit_count += 1;
        node.total_value += value;
        return value;
    }

    if !node.is_expanded {
        // Leaf: expand and evaluate
        let features = board_to_features(board);
        let (logits, value) = network.predict(&features);
        expand_node(node, board, &logits);
        node.visit_count += 1;
        node.total_value += value;
        return value;
    }

    if node.children.is_empty() {
        // No legal moves (shouldn't happen normally)
        node.visit_count += 1;
        return 0.0;
    }

    // SELECT: pick child with highest PUCT
    let child_idx = select_child_index(node);
    let child_mov = node.children[child_idx].mov.unwrap();
    board.make_move(child_mov);

    // Recurse
    let child_value = simulate_nn(&mut node.children[child_idx], board, network);

    // The child's value is from the child's side perspective.
    // Our value is the negative (zero-sum game).
    let value = -child_value;
    node.visit_count += 1;
    node.total_value += value;
    value
}

/// Recursive simulation with random rollout evaluation.
fn simulate_random(node: &mut Node, board: &mut Board) -> f32 {
    if board.is_terminal() {
        let value = board.outcome_for_side(board.side_to_move_u8());
        node.visit_count += 1;
        node.total_value += value;
        return value;
    }

    if !node.is_expanded {
        let uniform_logits = [0.0f32; NUM_ACTIONS];
        expand_node(node, board, &uniform_logits);
        let value = random_rollout(board);
        node.visit_count += 1;
        node.total_value += value;
        return value;
    }

    if node.children.is_empty() {
        node.visit_count += 1;
        return 0.0;
    }

    let child_idx = select_child_index(node);
    let child_mov = node.children[child_idx].mov.unwrap();
    board.make_move(child_mov);

    let child_value = simulate_random(&mut node.children[child_idx], board);
    let value = -child_value;
    node.visit_count += 1;
    node.total_value += value;
    value
}

fn random_rollout(board: &Board) -> f32 {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut b = board.clone();
    let side = b.side_to_move_u8();
    let mut moves = 0;

    while !b.is_terminal() && moves < 300 {
        let legal = b.legal_moves();
        if legal.is_empty() {
            break;
        }
        let idx = rng.gen_range(0..legal.len());
        b.make_move(legal[idx]);
        moves += 1;
    }

    b.outcome_for_side(side)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = Node::new(Some(3), 0.5);
        assert_eq!(node.mov, Some(3));
        assert_eq!(node.prior, 0.5);
        assert_eq!(node.visit_count, 0);
        assert_eq!(node.q_value(), 0.0);
        assert!(!node.is_expanded);
    }

    #[test]
    fn test_expand_node() {
        let board = Board::new();
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut root = Node::new(None, 1.0);
        expand_node(&mut root, &board, &logits);

        assert!(root.is_expanded);
        assert_eq!(root.children.len(), 9);

        let prior_sum: f32 = root.children.iter().map(|c| c.prior).sum();
        assert!((prior_sum - 1.0).abs() < 0.001);

        let last_prior = root.children.last().unwrap().prior;
        let first_prior = root.children.first().unwrap().prior;
        assert!(last_prior > first_prior);
    }

    #[test]
    fn test_puct_favors_unvisited() {
        let mut parent = Node::new(None, 1.0);
        parent.visit_count = 100;
        parent.children.push(Node::new(Some(0), 0.5));
        parent.children.push(Node::new(Some(1), 0.5));

        parent.children[0].visit_count = 90;
        parent.children[0].total_value = 45.0;

        let score0 = parent.children[0].puct_score(100);
        let score1 = parent.children[1].puct_score(100);
        assert!(score1 > score0);
    }

    #[test]
    fn test_mcts_random_returns_valid_moves() {
        let board = Board::new();
        let (move_visits, _value) = mcts_search_random(&board, 50);

        assert!(!move_visits.is_empty());
        let total_visits: u32 = move_visits.iter().map(|(_, v)| *v).sum();
        assert!(total_visits > 0);

        let legal = board.legal_moves();
        for (mov, _) in &move_visits {
            assert!(legal.contains(mov));
        }
    }
}

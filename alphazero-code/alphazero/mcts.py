"""
Monte Carlo Tree Search with Neural Network guidance
AlphaZero-style MCTS
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from game import TogyzQumalaq, GameState


@dataclass
class MCTSConfig:
    """MCTS configuration"""
    num_simulations: int = 800  # Number of MCTS simulations per move
    c_puct: float = 1.5  # Exploration constant
    dirichlet_alpha: float = 0.3  # Dirichlet noise alpha (for root)
    dirichlet_epsilon: float = 0.25  # Fraction of noise to add
    temperature: float = 1.0  # Temperature for move selection
    

class MCTSNode:
    """MCTS tree node"""
    
    def __init__(self, prior: float):
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.children: Dict[int, 'MCTSNode'] = {}
        self.state: Optional[GameState] = None
    
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """Select child with highest UCB score"""
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        sqrt_parent_visits = math.sqrt(self.visit_count)
        
        for action, child in self.children.items():
            # UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            ucb_score = child.value + c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, policy: np.ndarray, valid_moves: np.ndarray):
        """Expand node with given policy"""
        for action in range(len(policy)):
            if valid_moves[action] > 0:
                self.children[action] = MCTSNode(prior=policy[action])
    
    def add_dirichlet_noise(self, alpha: float, epsilon: float):
        """Add Dirichlet noise to prior for exploration"""
        actions = list(self.children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        
        for i, action in enumerate(actions):
            self.children[action].prior = (
                (1 - epsilon) * self.children[action].prior + 
                epsilon * noise[i]
            )


class MCTS:
    """
    Monte Carlo Tree Search with Neural Network guidance
    """
    
    def __init__(self, model, config: MCTSConfig = None):
        self.model = model
        self.config = config or MCTSConfig()
        self.game = TogyzQumalaq()
    
    def search(self, state: GameState, add_noise: bool = True) -> np.ndarray:
        """
        Run MCTS and return visit count distribution (policy)
        
        Args:
            state: Current game state
            add_noise: Whether to add Dirichlet noise at root
        
        Returns:
            Policy vector (visit counts normalized)
        """
        root = MCTSNode(prior=0)
        
        # Expand root
        self.game.set_state(state)
        encoded_state = self.game.encode_state()
        policy, _ = self.model.predict(encoded_state)
        valid_moves = self.game.get_valid_moves()
        
        # Mask invalid moves and renormalize
        policy = policy * valid_moves
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # Fallback to uniform over valid moves
            policy = valid_moves / valid_moves.sum()
        
        root.expand(policy, valid_moves)
        root.state = state.copy()
        
        # Add exploration noise at root
        if add_noise:
            root.add_dirichlet_noise(
                self.config.dirichlet_alpha,
                self.config.dirichlet_epsilon
            )
        
        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            self.game.set_state(state)
            
            # Selection: traverse tree to leaf
            while node.expanded():
                action, node = node.select_child(self.config.c_puct)
                search_path.append(node)
                self.game.make_move(action)
            
            # Check if game is terminal
            winner = self.game.get_winner()
            
            if winner is not None:
                # Terminal node: use actual result
                if winner == 2:  # Draw
                    value = 0.0
                else:
                    # Value from perspective of player who just moved
                    current_player = self.game.state.current_player
                    if winner == current_player:
                        value = -1.0  # We lost (opponent won)
                    else:
                        value = 1.0  # We won
            else:
                # Non-terminal: expand and evaluate with network
                encoded = self.game.encode_state()
                policy, value = self.model.predict(encoded)
                valid = self.game.get_valid_moves()
                
                # Mask and normalize policy
                policy = policy * valid
                policy_sum = policy.sum()
                if policy_sum > 0:
                    policy = policy / policy_sum
                else:
                    policy = valid / valid.sum() if valid.sum() > 0 else np.ones(9) / 9
                
                node.expand(policy, valid)
                node.state = self.game.get_state()

                # Network outputs value from current player's perspective,
                # but backprop/select_child expects parent's perspective.
                # Negate to convert: child's perspective -> parent's perspective.
                value = -value
            
            # Backpropagation
            self._backpropagate(search_path, value)
        
        # Return visit counts as policy
        return self._get_policy(root)
    
    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        """Backpropagate value through search path"""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent
    
    def _get_policy(self, root: MCTSNode) -> np.ndarray:
        """Get policy from root visit counts"""
        policy = np.zeros(9, dtype=np.float32)
        
        for action, child in root.children.items():
            policy[action] = child.visit_count
        
        # Apply temperature
        if self.config.temperature == 0:
            # Deterministic: pick best
            best = np.argmax(policy)
            policy = np.zeros(9, dtype=np.float32)
            policy[best] = 1.0
        else:
            # Apply temperature
            policy = np.power(policy, 1.0 / self.config.temperature)
            policy = policy / policy.sum()
        
        return policy
    
    def get_action_probs(self, state: GameState, temperature: float = 1.0) -> Tuple[np.ndarray, MCTSNode]:
        """
        Get action probabilities from MCTS
        
        Args:
            state: Current game state
            temperature: Temperature for move selection
        
        Returns:
            Policy vector and root node
        """
        original_temp = self.config.temperature
        self.config.temperature = temperature
        
        policy = self.search(state, add_noise=(temperature > 0))
        
        self.config.temperature = original_temp
        
        return policy, None  # Return None for root (we don't need it externally)


class MCTSParallel:
    """
    Parallel MCTS for batch evaluation
    Groups multiple MCTS searches for batch GPU inference
    """
    
    def __init__(self, model, config: MCTSConfig = None, num_parallel: int = 8):
        self.model = model
        self.config = config or MCTSConfig()
        self.num_parallel = num_parallel
        self.games = [TogyzQumalaq() for _ in range(num_parallel)]
    
    def search_batch(self, states: List[GameState]) -> List[np.ndarray]:
        """
        Run MCTS for multiple states in parallel
        
        Args:
            states: List of game states
        
        Returns:
            List of policy vectors
        """
        results = []
        
        # Process in batches
        for i in range(0, len(states), self.num_parallel):
            batch_states = states[i:i + self.num_parallel]
            batch_policies = self._search_parallel(batch_states)
            results.extend(batch_policies)
        
        return results
    
    def _search_parallel(self, states: List[GameState]) -> List[np.ndarray]:
        """Run parallel MCTS for a batch of states"""
        n = len(states)
        roots = [MCTSNode(prior=0) for _ in range(n)]
        
        # Initialize roots with batch prediction
        encoded_states = []
        for i, state in enumerate(states):
            self.games[i].set_state(state)
            encoded_states.append(self.games[i].encode_state())
        
        encoded_batch = np.array(encoded_states)
        policies, _ = self.model.predict_batch(encoded_batch)
        
        # Expand roots
        for i in range(n):
            valid = self.games[i].get_valid_moves()
            policy = policies[i] * valid
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                policy = valid / valid.sum()
            
            roots[i].expand(policy, valid)
            roots[i].state = states[i].copy()
            roots[i].add_dirichlet_noise(
                self.config.dirichlet_alpha,
                self.config.dirichlet_epsilon
            )
        
        # Run simulations
        for _ in range(self.config.num_simulations):
            # Collect leaf nodes that need evaluation
            leaves_to_eval = []
            search_paths = []
            leaf_indices = []
            
            for i in range(n):
                node = roots[i]
                path = [node]
                self.games[i].set_state(states[i])
                
                # Selection
                while node.expanded():
                    action, node = node.select_child(self.config.c_puct)
                    path.append(node)
                    self.games[i].make_move(action)
                
                search_paths.append(path)
                
                # Check terminal
                winner = self.games[i].get_winner()
                if winner is not None:
                    # Terminal - immediate backprop
                    if winner == 2:
                        value = 0.0
                    else:
                        current = self.games[i].state.current_player
                        value = 1.0 if winner != current else -1.0
                    self._backpropagate(path, value)
                else:
                    # Need evaluation
                    leaves_to_eval.append(self.games[i].encode_state())
                    leaf_indices.append(i)
            
            if leaves_to_eval:
                # Batch evaluate leaves
                batch = np.array(leaves_to_eval)
                policies, values = self.model.predict_batch(batch)
                
                # Expand and backprop
                for j, i in enumerate(leaf_indices):
                    valid = self.games[i].get_valid_moves()
                    policy = policies[j] * valid
                    policy_sum = policy.sum()
                    if policy_sum > 0:
                        policy = policy / policy_sum
                    else:
                        policy = valid / valid.sum() if valid.sum() > 0 else np.ones(9) / 9
                    
                    node = search_paths[i][-1]
                    node.expand(policy, valid)
                    node.state = self.games[i].get_state()

                    # Negate: network value is from child's perspective,
                    # backprop expects parent's perspective
                    self._backpropagate(search_paths[i], -values[j])
        
        # Extract policies
        result_policies = []
        for root in roots:
            policy = np.zeros(9, dtype=np.float32)
            for action, child in root.children.items():
                policy[action] = child.visit_count
            
            if self.config.temperature == 0:
                best = np.argmax(policy)
                policy = np.zeros(9, dtype=np.float32)
                policy[best] = 1.0
            else:
                policy = np.power(policy, 1.0 / self.config.temperature)
                total = policy.sum()
                if total > 0:
                    policy = policy / total
            
            result_policies.append(policy)
        
        return result_policies
    
    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value


def select_action(policy: np.ndarray, temperature: float = 1.0) -> int:
    """
    Select action from policy
    
    Args:
        policy: Probability distribution over actions
        temperature: 0 = deterministic, >0 = stochastic
    
    Returns:
        Selected action index
    """
    if temperature == 0:
        return int(np.argmax(policy))
    
    # Sample from distribution
    return int(np.random.choice(len(policy), p=policy))


if __name__ == "__main__":
    # Test MCTS with a dummy model
    from model import create_model
    
    print("Testing MCTS...")
    
    # Create small model for testing
    model = create_model("small", device="cpu")
    
    config = MCTSConfig(
        num_simulations=100,  # Small for testing
        c_puct=1.5
    )
    
    mcts = MCTS(model, config)
    
    # Test on initial position
    game = TogyzQumalaq()
    state = game.get_state()
    
    print("Running MCTS on initial position...")
    policy = mcts.search(state)
    
    print(f"Policy: {policy}")
    print(f"Policy sum: {policy.sum():.3f}")
    print(f"Best move: pit {np.argmax(policy) + 1}")
    
    # Test parallel MCTS
    print("\nTesting Parallel MCTS...")
    parallel_mcts = MCTSParallel(model, config, num_parallel=4)
    
    states = [game.get_state() for _ in range(4)]
    policies = parallel_mcts.search_batch(states)
    
    print(f"Got {len(policies)} policies")
    for i, p in enumerate(policies):
        print(f"  State {i}: best move = pit {np.argmax(p) + 1}")
    
    print("\nMCTS tests passed!")


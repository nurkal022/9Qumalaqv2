"""
Тоғызқұмалақ - Game Logic for AlphaZero
Fast numpy-based implementation for self-play
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


class Player(IntEnum):
    WHITE = 0
    BLACK = 1


@dataclass
class GameState:
    """Immutable game state representation"""
    pits: np.ndarray  # Shape: (2, 9) - [player][pit]
    kazan: np.ndarray  # Shape: (2,) - captured stones
    tuzdyk: np.ndarray  # Shape: (2,) - tuzdyk position (-1 if none)
    current_player: int
    
    def copy(self) -> 'GameState':
        return GameState(
            pits=self.pits.copy(),
            kazan=self.kazan.copy(),
            tuzdyk=self.tuzdyk.copy(),
            current_player=self.current_player
        )


class TogyzQumalaq:
    """
    Тоғызқұмалақ game engine optimized for AlphaZero
    """
    
    # Constants
    NUM_PITS = 9
    INITIAL_STONES = 9
    TOTAL_STONES = 162  # 9 * 9 * 2
    WIN_THRESHOLD = 82
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> GameState:
        """Reset to initial position"""
        self.state = GameState(
            pits=np.full((2, 9), self.INITIAL_STONES, dtype=np.int32),
            kazan=np.zeros(2, dtype=np.int32),
            tuzdyk=np.full(2, -1, dtype=np.int8),
            current_player=Player.WHITE
        )
        return self.state
    
    def get_state(self) -> GameState:
        return self.state
    
    def set_state(self, state: GameState):
        self.state = state.copy()
    
    def get_valid_moves(self) -> np.ndarray:
        """
        Returns binary mask of valid moves (9 elements)
        """
        player = self.state.current_player
        opponent = 1 - player
        
        valid = np.zeros(9, dtype=np.float32)
        
        for i in range(9):
            # Can't play from opponent's tuzdyk on our side
            if self.state.tuzdyk[opponent] == i:
                continue
            if self.state.pits[player, i] > 0:
                valid[i] = 1.0
        
        return valid
    
    def get_valid_moves_list(self) -> List[int]:
        """Returns list of valid move indices"""
        return np.where(self.get_valid_moves() == 1)[0].tolist()
    
    def make_move(self, pit_index: int) -> Tuple[bool, Optional[int]]:
        """
        Execute a move. Returns (success, winner or None)
        Winner: 0=white, 1=black, 2=draw, None=game continues
        """
        player = self.state.current_player
        opponent = 1 - player
        
        stones = self.state.pits[player, pit_index]
        if stones == 0:
            return False, None
        
        # Check if valid (not opponent's tuzdyk)
        if self.state.tuzdyk[opponent] == pit_index:
            return False, None
        
        # Pick up stones
        self.state.pits[player, pit_index] = 0
        
        current_pit = pit_index
        current_side = player
        
        if stones == 1:
            # Special case: single stone moves to next pit
            current_pit += 1
            if current_pit > 8:
                current_pit = 0
                current_side = opponent
            
            # Check if landing on tuzdyk
            if current_side == opponent and self.state.tuzdyk[player] == current_pit:
                # Our tuzdyk on opponent's side - collect stone
                self.state.kazan[player] += 1
            elif current_side == player and self.state.tuzdyk[opponent] == current_pit:
                # Opponent's tuzdyk on our side - they collect
                self.state.kazan[opponent] += 1
            else:
                self.state.pits[current_side, current_pit] += 1
        else:
            # Normal case: distribute stones
            self.state.pits[current_side, current_pit] += 1
            stones -= 1
            
            while stones > 0:
                current_pit += 1
                if current_pit > 8:
                    current_pit = 0
                    current_side = 1 - current_side
                
                # Check tuzdyk
                if current_side == opponent and self.state.tuzdyk[player] == current_pit:
                    self.state.kazan[player] += 1
                elif current_side == player and self.state.tuzdyk[opponent] == current_pit:
                    self.state.kazan[opponent] += 1
                else:
                    self.state.pits[current_side, current_pit] += 1
                
                stones -= 1
        
        # Check capture and tuzdyk creation (only if landed on opponent's side, not on any tuzdyk)
        is_any_tuzdyk = (
            (current_side == opponent and self.state.tuzdyk[player] == current_pit) or
            (current_side == player and self.state.tuzdyk[opponent] == current_pit)
        )
        
        if current_side == opponent and not is_any_tuzdyk:
            count = self.state.pits[opponent, current_pit]
            
            # Tuzdyk creation: exactly 3 stones
            if count == 3 and self._can_create_tuzdyk(player, current_pit):
                self.state.tuzdyk[player] = current_pit
                self.state.kazan[player] += count
                self.state.pits[opponent, current_pit] = 0
            # Capture: even number of stones
            elif count % 2 == 0 and count > 0:
                self.state.kazan[player] += count
                self.state.pits[opponent, current_pit] = 0
        
        # Switch player
        self.state.current_player = opponent
        
        # Check for winner
        winner = self._check_winner()
        
        return True, winner
    
    def _can_create_tuzdyk(self, player: int, pit_index: int) -> bool:
        """Check if player can create tuzdyk at given pit"""
        # Already has tuzdyk
        if self.state.tuzdyk[player] != -1:
            return False
        # Can't make tuzdyk at pit 9 (index 8)
        if pit_index == 8:
            return False
        # Can't make tuzdyk at same position as opponent's tuzdyk
        opponent = 1 - player
        if self.state.tuzdyk[opponent] == pit_index:
            return False
        return True
    
    def _check_winner(self) -> Optional[int]:
        """
        Check if game is over.
        Returns: 0=white wins, 1=black wins, 2=draw, None=continues
        """
        # Win by threshold
        if self.state.kazan[Player.WHITE] >= self.WIN_THRESHOLD:
            return Player.WHITE
        if self.state.kazan[Player.BLACK] >= self.WIN_THRESHOLD:
            return Player.BLACK
        
        # Check if any side is empty
        white_empty = np.all(self.state.pits[Player.WHITE] == 0)
        black_empty = np.all(self.state.pits[Player.BLACK] == 0)
        
        if white_empty or black_empty:
            if self.state.kazan[Player.WHITE] > self.state.kazan[Player.BLACK]:
                return Player.WHITE
            elif self.state.kazan[Player.BLACK] > self.state.kazan[Player.WHITE]:
                return Player.BLACK
            else:
                return 2  # Draw
        
        return None
    
    def is_terminal(self) -> bool:
        """Check if game is over"""
        return self._check_winner() is not None
    
    def get_winner(self) -> Optional[int]:
        """Get winner (call only if is_terminal)"""
        return self._check_winner()
    
    def encode_state(self) -> np.ndarray:
        """
        Encode state for neural network input.
        Returns tensor of shape (channels, 9) where:
        - Channel 0: Current player's pits (normalized)
        - Channel 1: Opponent's pits (normalized)
        - Channel 2: Current player's kazan (normalized)
        - Channel 3: Opponent's kazan (normalized)
        - Channel 4: Current player's tuzdyk (one-hot)
        - Channel 5: Opponent's tuzdyk (one-hot)
        - Channel 6: Current player indicator (all 1s or 0s)
        
        Total: 7 channels x 9 positions = 63 values
        """
        player = self.state.current_player
        opponent = 1 - player
        
        # Normalization factor (max stones in a pit is ~50)
        NORM_FACTOR = 50.0
        KAZAN_NORM = 82.0
        
        channels = np.zeros((7, 9), dtype=np.float32)
        
        # Pits (normalized)
        channels[0] = self.state.pits[player] / NORM_FACTOR
        channels[1] = self.state.pits[opponent] / NORM_FACTOR
        
        # Kazan (broadcast to all positions)
        channels[2, :] = self.state.kazan[player] / KAZAN_NORM
        channels[3, :] = self.state.kazan[opponent] / KAZAN_NORM
        
        # Tuzdyk (one-hot encoding)
        if self.state.tuzdyk[player] >= 0:
            channels[4, self.state.tuzdyk[player]] = 1.0
        if self.state.tuzdyk[opponent] >= 0:
            channels[5, self.state.tuzdyk[opponent]] = 1.0
        
        # Current player indicator
        channels[6, :] = 1.0 if player == Player.WHITE else 0.0
        
        return channels
    
    def encode_state_flat(self) -> np.ndarray:
        """Flatten encoded state for simpler network"""
        return self.encode_state().flatten()
    
    @staticmethod
    def get_state_shape() -> Tuple[int, int]:
        """Returns shape of encoded state"""
        return (7, 9)
    
    @staticmethod
    def get_action_size() -> int:
        """Returns number of possible actions"""
        return 9
    
    def get_canonical_state(self) -> GameState:
        """
        Returns state from current player's perspective.
        This ensures the network always sees the board from the same orientation.
        """
        if self.state.current_player == Player.WHITE:
            return self.state.copy()
        
        # Flip perspective for black
        canonical = GameState(
            pits=self.state.pits[::-1].copy(),  # Swap rows
            kazan=self.state.kazan[::-1].copy(),
            tuzdyk=self.state.tuzdyk[::-1].copy(),
            current_player=Player.WHITE  # Always white's turn in canonical
        )
        return canonical
    
    def get_symmetries(self, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns symmetries of state-policy pair.
        For Togyz, there's no spatial symmetry (unlike Go/Chess),
        so we just return the original.
        """
        return [(self.encode_state(), policy)]
    
    def __str__(self) -> str:
        """Pretty print the board"""
        s = self.state
        lines = []
        lines.append("=" * 50)
        lines.append(f"  Black Kazan: {s.kazan[1]}  |  Tuzdyk: {s.tuzdyk[1] + 1 if s.tuzdyk[1] >= 0 else '-'}")
        lines.append("  " + " ".join(f"{s.pits[1, 8-i]:2d}" for i in range(9)))
        lines.append("  " + " ".join(f" {9-i}" for i in range(9)))
        lines.append("-" * 50)
        lines.append("  " + " ".join(f" {i+1}" for i in range(9)))
        lines.append("  " + " ".join(f"{s.pits[0, i]:2d}" for i in range(9)))
        lines.append(f"  White Kazan: {s.kazan[0]}  |  Tuzdyk: {s.tuzdyk[0] + 1 if s.tuzdyk[0] >= 0 else '-'}")
        lines.append(f"  Current: {'White' if s.current_player == 0 else 'Black'}")
        lines.append("=" * 50)
        return "\n".join(lines)


def play_random_game():
    """Play a random game for testing"""
    game = TogyzQumalaq()
    print(game)
    
    move_count = 0
    while not game.is_terminal():
        moves = game.get_valid_moves_list()
        if not moves:
            break
        
        move = np.random.choice(moves)
        success, winner = game.make_move(move)
        move_count += 1
        
        if move_count <= 5 or winner is not None:
            print(f"\nMove {move_count}: Pit {move + 1}")
            print(game)
    
    print(f"\nGame over after {move_count} moves!")
    winner = game.get_winner()
    if winner == 2:
        print("Result: Draw!")
    else:
        print(f"Winner: {'White' if winner == 0 else 'Black'}")
    
    return game


if __name__ == "__main__":
    play_random_game()


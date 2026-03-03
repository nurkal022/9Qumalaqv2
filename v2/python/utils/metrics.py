"""
Training metrics logging and tracking.
"""

import json
import os
from datetime import datetime


class MetricsLogger:
    """Log training metrics to JSON file for later visualization."""

    def __init__(self, log_path):
        self.log_path = log_path
        self.entries = []
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                self.entries = json.load(f)

    def log_training(self, generation, epoch, loss, value_loss, policy_loss, entropy, lr):
        entry = {
            "type": "training",
            "generation": generation,
            "epoch": epoch,
            "loss": loss,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy": entropy,
            "lr": lr,
            "timestamp": datetime.now().isoformat(),
        }
        self.entries.append(entry)
        self._save()

    def log_arena(self, generation, wins, losses, draws, elo_diff):
        entry = {
            "type": "arena",
            "generation": generation,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "elo_diff": elo_diff,
            "winrate": (wins + 0.5 * draws) / max(wins + losses + draws, 1),
            "timestamp": datetime.now().isoformat(),
        }
        self.entries.append(entry)
        self._save()

    def log_selfplay(self, generation, games, samples, avg_game_length):
        entry = {
            "type": "selfplay",
            "generation": generation,
            "games": games,
            "samples": samples,
            "avg_game_length": avg_game_length,
            "timestamp": datetime.now().isoformat(),
        }
        self.entries.append(entry)
        self._save()

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.entries, f, indent=2)

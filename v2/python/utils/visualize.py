"""
Visualization of training progress (loss, Elo by generation).
"""

import json
import argparse


def plot_training(log_path, output_prefix="v2/models/training"):
    """Generate training progress plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    with open(log_path, "r") as f:
        entries = json.load(f)

    # Filter by type
    training = [e for e in entries if e["type"] == "training"]
    arena = [e for e in entries if e["type"] == "arena"]

    if training:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        gens = [e["generation"] for e in training]
        losses = [e["loss"] for e in training]
        axes[0, 0].plot(gens, losses, "b-")
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Generation")
        axes[0, 0].set_ylabel("Loss")

        # Value Loss
        vloss = [e["value_loss"] for e in training]
        axes[0, 1].plot(gens, vloss, "r-")
        axes[0, 1].set_title("Value Loss")
        axes[0, 1].set_xlabel("Generation")

        # Policy Loss
        ploss = [e["policy_loss"] for e in training]
        axes[1, 0].plot(gens, ploss, "g-")
        axes[1, 0].set_title("Policy Loss")
        axes[1, 0].set_xlabel("Generation")

        # Entropy
        ent = [e["entropy"] for e in training]
        axes[1, 1].plot(gens, ent, "m-")
        axes[1, 1].set_title("Policy Entropy")
        axes[1, 1].set_xlabel("Generation")

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_loss.png", dpi=150)
        print(f"Saved loss plot to {output_prefix}_loss.png")

    if arena:
        fig, ax = plt.subplots(figsize=(10, 6))
        gens = [e["generation"] for e in arena]
        elos = [e["elo_diff"] for e in arena]
        winrates = [e["winrate"] * 100 for e in arena]

        ax.bar(gens, elos, alpha=0.7)
        ax.axhline(y=0, color="r", linestyle="--")
        ax.set_title("Elo Difference vs Previous Generation")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Elo Diff")

        # Add winrate labels
        for g, e, w in zip(gens, elos, winrates):
            ax.text(g, e + 5, f"{w:.0f}%", ha="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_elo.png", dpi=150)
        print(f"Saved Elo plot to {output_prefix}_elo.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="v2/models/metrics.json")
    parser.add_argument("--output", default="v2/models/training")
    args = parser.parse_args()
    plot_training(args.log, args.output)

"""
TogyzNet V2 - Dual-head ResNet for Gumbel AlphaZero

Inputs: [batch, 70] - board position features
Outputs:
  - policy_logits: [batch, 9] - raw logits (NOT softmax)
  - value: [batch, 1] - position evaluation [-1, +1]

Architecture: Input -> FC -> 6 ResBlocks -> Policy Head + Value Head
Parameters: ~150-200K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_SIZE = 70
NUM_ACTIONS = 9


class ResBlock(nn.Module):
    """Residual block with pre-activation (BN -> ReLU -> Linear)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(x))
        out = self.fc1(out)
        out = F.relu(self.bn2(out))
        out = self.fc2(out)
        return out + residual


class TogyzNetV2(nn.Module):
    """
    Dual-head network for Gumbel AlphaZero togyz kumalak.

    Args:
        input_size: Feature vector dimension (default: 70)
        hidden_size: Hidden layer width (default: 256)
        num_blocks: Number of residual blocks (default: 6)
    """

    def __init__(self, input_size=FEATURE_SIZE, hidden_size=256, num_blocks=6):
        super().__init__()

        # Input projection
        self.input_fc = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)

        # Trunk: residual blocks
        self.res_blocks = nn.ModuleList(
            [ResBlock(hidden_size) for _ in range(num_blocks)]
        )

        # Policy head: 9 move logits
        self.policy_bn = nn.BatchNorm1d(hidden_size)
        self.policy_fc1 = nn.Linear(hidden_size, 64)
        self.policy_fc2 = nn.Linear(64, NUM_ACTIONS)

        # Value head: scalar evaluation
        self.value_bn = nn.BatchNorm1d(hidden_size)
        self.value_fc1 = nn.Linear(hidden_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Trunk
        out = F.relu(self.input_bn(self.input_fc(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy head
        p = F.relu(self.policy_bn(out))
        p = F.relu(self.policy_fc1(p))
        policy_logits = self.policy_fc2(p)  # [batch, 9] raw logits

        # Value head
        v = F.relu(self.value_bn(out))
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))  # [batch, 1] in [-1, +1]

        return policy_logits, value

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_initial_model(output_path, hidden_size=256, num_blocks=6):
    """Create and save an initial random model."""
    model = TogyzNetV2(
        input_size=FEATURE_SIZE, hidden_size=hidden_size, num_blocks=num_blocks
    )
    torch.save(model.state_dict(), output_path)
    print(f"Saved initial model to {output_path}")
    print(f"Parameters: {model.count_parameters():,}")
    return model


if __name__ == "__main__":
    import sys
    import os

    model = TogyzNetV2()
    print(f"Parameters: {model.count_parameters():,}")

    # Test forward pass
    x = torch.randn(4, FEATURE_SIZE)
    policy, value = model(x)
    print(f"Policy shape: {policy.shape}")  # [4, 9]
    print(f"Value shape: {value.shape}")  # [4, 1]
    print(f"Policy sample: {F.softmax(policy[0], dim=0).detach().numpy()}")
    print(f"Value sample: {value[0].item():.4f}")

    # Save initial model if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        os.makedirs("v2/models", exist_ok=True)
        create_initial_model("v2/models/gen_0.pt")

"""
Neural Network for AlphaZero Тоғызқұмалақ
ResNet-style architecture with policy and value heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = F.relu(x)
        return x


class TogyzNet(nn.Module):
    """
    AlphaZero-style neural network for Тоғызқұмалақ
    
    Architecture:
    - Input: 7 channels x 9 positions
    - Residual tower: N residual blocks
    - Policy head: probability distribution over 9 moves
    - Value head: game outcome prediction [-1, 1]
    """
    
    def __init__(
        self,
        input_channels: int = 7,
        board_size: int = 9,
        num_res_blocks: int = 10,
        num_channels: int = 128
    ):
        super().__init__()
        
        self.board_size = board_size
        self.num_channels = num_channels
        
        # Initial convolution
        self.conv_input = nn.Conv1d(input_channels, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm1d(num_channels)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv1d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm1d(32)
        self.policy_fc = nn.Linear(32 * board_size, board_size)
        
        # Value head
        self.value_conv = nn.Conv1d(num_channels, 4, kernel_size=1)
        self.value_bn = nn.BatchNorm1d(4)
        self.value_fc1 = nn.Linear(4 * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, 7, 9)
        
        Returns:
            policy: Log probabilities of shape (batch, 9)
            value: Value estimate of shape (batch, 1)
        """
        # Initial conv
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * self.board_size)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 4 * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict policy and value for a single state
        
        Args:
            state: Encoded state of shape (7, 9)
        
        Returns:
            policy: Probability distribution over moves
            value: Value estimate
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            
            log_policy, value = self(x)
            policy = torch.exp(log_policy).cpu().numpy()[0]
            value = value.cpu().numpy()[0, 0]
        
        return policy, value
    
    def predict_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict policy and value for a batch of states
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(states)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            
            log_policy, value = self(x)
            policy = torch.exp(log_policy).cpu().numpy()
            value = value.cpu().numpy()[:, 0]
        
        return policy, value


class TogyzNetSmall(nn.Module):
    """
    Smaller, faster network for initial training/testing
    Can be used for quick iterations before full training
    """
    
    def __init__(self, input_channels: int = 7, board_size: int = 9):
        super().__init__()
        
        self.board_size = board_size
        
        # Simple MLP architecture
        flat_size = input_channels * board_size
        
        self.shared = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, board_size),
            nn.LogSoftmax(dim=1)
        )
        
        # Value head
        self.value = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Flatten input
        x = x.view(x.size(0), -1)
        
        shared = self.shared(x)
        policy = self.policy(shared)
        value = self.value(shared)
        
        return policy, value
    
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            
            log_policy, value = self(x)
            policy = torch.exp(log_policy).cpu().numpy()[0]
            value = value.cpu().numpy()[0, 0]
        
        return policy, value


class TogyzNetLarge(nn.Module):
    """
    Larger network for maximum strength
    Use this for final training after hyperparameter tuning
    """
    
    def __init__(
        self,
        input_channels: int = 7,
        board_size: int = 9,
        num_res_blocks: int = 20,
        num_channels: int = 256
    ):
        super().__init__()
        
        self.board_size = board_size
        self.num_channels = num_channels
        
        # Initial convolution
        self.conv_input = nn.Conv1d(input_channels, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm1d(num_channels)
        
        # Residual tower (deeper)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head with more capacity
        self.policy_conv = nn.Conv1d(num_channels, 64, kernel_size=1)
        self.policy_bn = nn.BatchNorm1d(64)
        self.policy_fc1 = nn.Linear(64 * board_size, 128)
        self.policy_fc2 = nn.Linear(128, board_size)
        
        # Value head with more capacity
        self.value_conv = nn.Conv1d(num_channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm1d(8)
        self.value_fc1 = nn.Linear(8 * board_size, 128)
        self.value_fc2 = nn.Linear(128, 64)
        self.value_fc3 = nn.Linear(64, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial conv
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 64 * self.board_size)
        policy = F.relu(self.policy_fc1(policy))
        policy = self.policy_fc2(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 8 * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = F.relu(self.value_fc2(value))
        value = torch.tanh(self.value_fc3(value))
        
        return policy, value
    
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            
            log_policy, value = self(x)
            policy = torch.exp(log_policy).cpu().numpy()[0]
            value = value.cpu().numpy()[0, 0]
        
        return policy, value


def create_model(size: str = "medium", device: str = "cuda") -> nn.Module:
    """
    Factory function to create model
    
    Args:
        size: "small", "medium", or "large"
        device: "cuda" or "cpu"
    
    Returns:
        Model instance
    """
    if size == "small":
        model = TogyzNetSmall()
    elif size == "medium":
        model = TogyzNet(num_res_blocks=10, num_channels=128)
    elif size == "large":
        model = TogyzNetLarge(num_res_blocks=20, num_channels=256)
    else:
        raise ValueError(f"Unknown model size: {size}")
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("Testing models...")
    
    for size in ["small", "medium", "large"]:
        model = create_model(size, device="cpu")
        params = count_parameters(model)
        print(f"{size.capitalize()} model: {params:,} parameters")
        
        # Test forward pass
        x = torch.randn(4, 7, 9)  # batch of 4
        policy, value = model(x)
        print(f"  Policy shape: {policy.shape}, Value shape: {value.shape}")
        
        # Test predict
        state = np.random.randn(7, 9).astype(np.float32)
        p, v = model.predict(state)
        print(f"  Predict: policy sum={p.sum():.3f}, value={v:.3f}")
    
    print("\nAll tests passed!")


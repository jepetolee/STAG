import torch
import torch.nn as nn
import torch.nn.functional as F

num_experts = 20


class MixedActor(nn.Module):
    def __init__(self, Size, hidden_size, output_size):
        super().__init__()

        expert_input_size = Size
        gate_input_size = Size

        self.layers = [
            (
                nn.Parameter(torch.empty(num_experts, expert_input_size, hidden_size)),
                nn.Parameter(torch.zeros(num_experts, 1, hidden_size)),
                torch.relu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, hidden_size, hidden_size)),
                nn.Parameter(torch.zeros(num_experts, 1, hidden_size)),
                torch.relu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, hidden_size, hidden_size)),
                nn.Parameter(torch.zeros(num_experts, 1, hidden_size)),
                torch.relu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, hidden_size, output_size)),
                nn.Parameter(torch.zeros(num_experts, 1, output_size)),
                torch.tanh,
            ),
        ]

        for index, (weight, bias, activation) in enumerate(self.layers):

            for w in weight:
                nn.init.orthogonal_(w, gain=1.0)

            self.register_parameter(f"w{index}", weight)
            self.register_parameter(f"b{index}", bias)

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(gate_input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, num_experts),
        )

    def forward(self, x):
        coefficients = F.softmax(self.gate(x), dim=1).t().unsqueeze(-1)
        out = x.clone()

        for (weight, bias, activation) in self.layers:
            out = activation(
                out.matmul(weight)  # (N, B, H), B = Batch, H = hidden
                .add(bias)  # (N, B, H)
                .mul(coefficients)  # (B, H)
                .sum(dim=0)
            )

        return out

import torch


class RSSMState:
    def __init__(self, mean, standard_deviation, stochastic_state, deterministic_state):
        self.mean = torch.tensor(mean, dtype=torch.float64)
        self.standard_deviation = torch.tensor(standard_deviation, dtype=torch.float64)
        self.stochastic_state = torch.tensor(stochastic_state, dtype=torch.float64)
        self.deterministic_state = torch.tensor(deterministic_state, dtype=torch.float64)

    def get_feature(self):
        return torch.cat((self.stochastic_state, self.deterministic_state), dim=-1)

    def get_distribution(self):
        return torch.cat((self.mean, self.standard_deviation), dim=-1)


def stack_states(states: list, dim):
    return RSSMState(torch.stack([state.mean for state in states], dim=dim),
                     torch.stack([state.standard_deviation for state in states], dim=dim),
                     torch.stack([state.stochastic_state for state in states], dim=dim),
                     torch.stack([state.deterministic_state for state in states], dim=dim),)

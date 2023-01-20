import torch
import torch.distributions as Distribution

class RSSMState:
    def __init__(self, mean, standard_deviation, stochastic_state, deterministic_state):
        self.mean = mean.clone().detach()
        self.standard_deviation = standard_deviation.clone().detach()
        self.stochastic_state = stochastic_state.clone().detach()
        self.deterministic_state = deterministic_state.clone().detach()

    def get_feature(self):
        return torch.cat((self.stochastic_state, self.deterministic_state), dim=-1)

    def get_distribution(self):
        return Distribution.independent.Independent(Distribution.Normal(self.mean, self.standard_deviation), 1)


def stack_states(states: list, dim):
    return RSSMState(torch.stack([state.mean[0] for state in states], dim=dim),
                     torch.stack([state.standard_deviation[0] for state in states], dim=dim),
                     torch.stack([state.stochastic_state[0] for state in states], dim=dim),
                     torch.stack([state.deterministic_state[0] for state in states], dim=dim),)

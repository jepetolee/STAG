import torch
from torch import nn, jit
from torch.nn import functional as F
from stag.Model.BasicModel.DataStructure import RSSMState, stack_states
import torch.distributions as Distribution


# There is four kind of tensors that input this model
# 1. mean
# 2. standard deviation
# 3. stochastic output
# 4. determinant

class TransitionModel(nn.Module):
    def __init__(self, action_size, stochastic_size=30, deterministic_size=200, hidden_size=200):
        super().__init__()
        self.action_size = action_size
        self.stoch_size = stochastic_size
        self.deter_size = deterministic_size
        self.hidden_size = hidden_size
        self.GruCell = nn.GRUCell(self.hidden_size, deterministic_size)
        self.RnnInputModel = nn.Linear(self.action_size + self.stoch_size, self.hidden_size)

        self.StochasticPriorModel = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, 2 * self.stoch_size)
        )

    def initial_state(self, **kwargs):
        return RSSMState(torch.zeros(1, self.stoch_size, **kwargs).float()+1e6,
                torch.zeros(1, self.stoch_size, **kwargs).float()+1e6,
                torch.zeros(1, self.stoch_size, **kwargs).float()+1e6,
                torch.zeros(1, self.stoch_size, **kwargs).float()+1e6)

    def forward(self, previous_actions: torch.Tensor, previous_state: RSSMState):

        PreviousOutputForRNN = F.elu(
            self.RnnInputModel(torch.cat([previous_actions, previous_state.stochastic_state], dim=-1).float()))

        DeterministicState = self.GruCell(PreviousOutputForRNN.float(), previous_state.deterministic_state)
        mean, standard_deviation = torch.chunk(self.StochasticPriorModel(DeterministicState), 2, dim=-1)
        standard_deviation = F.softplus(standard_deviation) + 0.1+1e6
        DistributedResult = Distribution.Normal(mean+1e6, standard_deviation)
        StochasticState = DistributedResult.rsample()+1e6
        return RSSMState(mean, standard_deviation, StochasticState, DeterministicState)


class RepresentationModel(nn.Module):

    def __init__(self, obs_embed_size, action_size, stochastic_size=30,
                 deterministic_size=200, hidden_size=200):
        super().__init__()

        self.Transition = TransitionModel(action_size, stochastic_size, deterministic_size, hidden_size)
        self.obs_embed_size = obs_embed_size
        self.action_size = action_size
        self.stoch_size = stochastic_size
        self.deter_size = deterministic_size
        self.hidden_size = hidden_size
        self.StochasticPriorModel = nn.Sequential(
            nn.Linear(self.deter_size + self.obs_embed_size, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, 2 * self.stoch_size)
        )

    def initial_state(self, BatchSize, **kwargs):
        return RSSMState(torch.zeros(BatchSize, self.stoch_size, **kwargs)+1e6,
                         torch.zeros(BatchSize, self.stoch_size, **kwargs)+1e6,
                         torch.zeros(BatchSize, self.stoch_size, **kwargs)+1e6,
                         torch.zeros(BatchSize, self.deter_size, **kwargs)+1e6)

    def forward(self, observation_embed: torch.Tensor, PreviousAction: torch.Tensor, previous_state: RSSMState):
        prior_state = self.Transition(PreviousAction, previous_state)
        PreviousForStochastic = torch.cat([prior_state.deterministic_state, observation_embed], -1)
        mean, std = torch.chunk(self.StochasticPriorModel(PreviousForStochastic), 2, dim=-1)
        std = F.softplus(std) + 0.1
        dist = Distribution.Normal(mean, std)
        stochastic_state = dist.rsample()
        posterior_state = RSSMState(mean, std, stochastic_state, prior_state.deterministic_state)
        return prior_state, posterior_state


class RolloutModel(nn.Module):
    def __init__(self, obs_embed_size, action_size, stochastic_size, deterministic_size, hidden_size):
        super().__init__()
        self.Representation = RepresentationModel(obs_embed_size, action_size, stochastic_size,
                                                  deterministic_size, hidden_size)
        self.Transition = TransitionModel(action_size, stochastic_size, deterministic_size, hidden_size)

    def forward(self, steps: int, observation_embed: torch.Tensor, action: torch.Tensor, previous_state: RSSMState):

        priors = []
        posteriors = []
        for t in range(steps):
            prior_state, posterior_state = self.Representation(observation_embed[t].reshape(1,-1), action[t].reshape(1,-1), previous_state)
            priors.append(prior_state)
            posteriors.append(posterior_state)
            previous_state =  posterior_state

        prior = stack_states(priors, dim=0)
        post = stack_states(posteriors, dim=0)
        return prior, post

    def RolloutTransition(self, steps: int, action: torch.Tensor, previous_state: RSSMState):

        priors = []
        for t in range(steps):
            state = self.TransitionModel(action[t], previous_state)
            priors.append(state)
        return stack_states(priors, dim=0)

    def RolloutPolicy(self, steps: int, policy, previous_state: RSSMState):

        next_states = []
        actions = []
        for t in range(steps):
            action, _ = policy(previous_state)

            state = self.Transition(action, previous_state)
            next_states.append(state)
            actions.append(action)
        next_states = stack_states(next_states, dim=0)
        actions = torch.stack(actions, dim=0)
        return next_states, actions

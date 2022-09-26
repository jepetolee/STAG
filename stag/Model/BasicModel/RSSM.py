import torch
from torch import nn, jit
from torch.nn import functional as F
from typing import List, Optional
import torch.distributions as Distribution


# There is four kind of tensors that input this model
# 1. mean
# 2. standard deviation
# 3. stochastic output
# 4. determinant

class TransitionModel(jit.ScriptModule):
    def __init__(self, action_size, stochastic_size=30, deterministic_size=200, hidden_size=200):
        super().__init__()
        self.action_size = action_size
        self.stoch_size = stochastic_size
        self.deter_size = deterministic_size
        self.hidden_size = hidden_size
        self.GruCell = nn.GRUCell(hidden_size, deterministic_size)
        self.RnnInputModel = nn.Linear(self._action_size + self._stoch_size, self._hidden_size)

        self.StochasticPriorModel = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, 2 * self.stoch_size)
        )

    @jit.script_method
    def initial_state(self, **kwargs):
        return (torch.zeros(1, self.stoch_size, **kwargs),
                torch.zeros(1, self.stoch_size, **kwargs),
                torch.zeros(1, self.stoch_size, **kwargs),
                torch.zeros(1, self.stoch_size, **kwargs))

    @jit.script_method
    def forward(self, previous_actions: torch.Tensor, stochastic_output: torch.Tensor,
                DeterministicValue: torch.Tensor):
        PreviousOutputForRNN = F.elu(self.RnnInputModel(torch.cat([previous_actions, stochastic_output], dim=-1)))
        DeterministicState = self.GruCell(PreviousOutputForRNN, DeterministicValue)
        mean, standard_deviation = torch.chunk(self.StochasticPriorModel(DeterministicState), 2, dim=-1)
        standard_deviation = F.softplus(standard_deviation) + 0.1
        DistributedResult = Distribution.Normal(mean, standard_deviation)
        Stochasticstate = DistributedResult.rsample()
        return mean, standard_deviation, Stochasticstate, DeterministicState


class RepresentationModel(jit.ScriptModule):

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

    @jit.script_method
    def initial_state(self, **kwargs):
        return (torch.zeros(1, self.stoch_size, **kwargs),
                torch.zeros(1, self.stoch_size, **kwargs),
                torch.zeros(1, self.stoch_size, **kwargs),
                torch.zeros(1, self.stoch_size, **kwargs))

    @jit.script_method
    def forward(self, observation_embed: torch.Tensor, PreviousAction: torch.Tensor,
                StochasticState: torch.Tensor, DeterministicState: torch.Tensor):
        prior_state = self.Transition(PreviousAction, StochasticState, DeterministicState)
        PreviousForStochastic = torch.cat([prior_state[3], observation_embed], -1)
        mean, std = torch.chunk(self.StochasticPriorModel(PreviousForStochastic), 2, dim=-1)
        std = F.softplus(std) + 0.1
        dist = nn.ELU(mean, std)
        stochastic_state = dist.rsample()
        return prior_state, mean, std, stochastic_state, prior_state[3]


class RolloutModel(jit.ScriptModule):
    def __init__(self, obs_embed_size, action_size, stochastic_size, deterministic_size, hidden_size):
        super().__init__()
        self.Representation = RepresentationModel(obs_embed_size, action_size, stochastic_size,
                                                  deterministic_size, hidden_size)
        self.Transition = TransitionModel(action_size, stochastic_size, deterministic_size, hidden_size)

    def forward(self, steps: int, observation_embed: torch.Tensor, action: torch.Tensor,
                action_size: torch.Tensor, mean: torch.Tensor, standard_deviation: torch.Tensor,
                Stochasticstate: torch.Tensor, DeterministicState: torch.Tensor):

        priors = []
        posteriors = []
        for t in range(steps):
            prior_state, mean,standard_deviation, Stochasticstate, DeterministicState = self.RepresentationModel(observation_embed[t], action[t], action_size,
                                                                     mean,standard_deviation, Stochasticstate, DeterministicState)
            priors.append(prior_state)
            posteriors.append((mean,standard_deviation, Stochasticstate,DeterministicState))

        return priors, posteriors

    def RolloutTransition(self, steps: int, action: torch.Tensor, mean: torch.Tensor, standard_deviation: torch.Tensor,
                Stochasticstate: torch.Tensor, DeterministicState: torch.Tensor):

        priors = []
        for t in range(steps):
            state = self.TransitionModel(action[t], mean, standard_deviation, Stochasticstate, DeterministicState)
            priors.append(state)
        return priors

    def RolloutPolicy(self, steps: int, policy,  mean: torch.Tensor, standard_deviation: torch.Tensor,
                Stochasticstate: torch.Tensor, DeterministicState: torch.Tensor):


        next_states = []
        actions = []
       # state = buffer_method(state, 'detach')
        for t in range(steps):
            action, _ = policy( mean, standard_deviation, Stochasticstate, DeterministicState)
            state = self.TransitionModel(action,  mean, standard_deviation, Stochasticstate, DeterministicState)
            next_states.append(state)
            actions.append(action)
        actions = torch.stack(actions, dim=0)
        return next_states, actions
import torch
from tqdm import trange
from RL_Model import *


# need to check all tensors tensor
class Dreamer:
    def __init__(self, device, train_steps, dtype=torch.float64, learning_rate=0.78):
        self.trading_model = TradingModel(output_size=3)
        self.train_steps = train_steps
        self.device = device
        self.type = dtype
        self.kl_scale = 1
        self.free_nats = 3
        self.discount = 0.99
        self.horizon = 15
        self.discount_lambda = 0.95
        self.learning_rate = learning_rate
        self.ModelParameters = list(self.trading_model.observation_encoder.parameters()) + list(
            self.trading_model.observation_decoder.parameters()) \
                               + list(self.trading_model.reward_model.parameters()) + list(
            self.trading_model.representation.parameters()) \
                               + list(self.trading_model.transition.parameters())
        self.gradient_clip = 100

    def RunModel(self, crypto_chart):
        return self.trading_model(crypto_chart)

    def GetLoss(self, observations, actions, rewards):
        Model = self.trading_model
        batch_size = observations[0]

        observations = observations.type(self.type) / 255.0 - 0.5
        embed = Model.observation_encoder(observations)

        prev_state = Model.representation.initial_state(batch_size, device=self.device, dtype=self.type)
        prior, post = Model.rollout.forward(batch_size, embed, actions, prev_state)

        feat = post.get_feature()
        image_pred = Model.observation_decoder(feat)
        reward_pred = Model.reward_model(feat)
        reward_loss = -torch.mean(reward_pred.log_prob(rewards))
        image_loss = -torch.mean(image_pred.log_prob(observations))

        prior_dist = prior.get_distribution()
        post_dist = post.get_distribution()
        div = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
        div = torch.max(div, div.new_full(div.size(), self.free_nats))
        model_loss = self.kl_scale * div + reward_loss + image_loss

        image_distribution, _ = Model.rollout.RolloutPolicy(self.horizon, Model.policy, post)
        imag_feat = image_distribution.get_distribution()

        image_reward = torch.mean(Model.reward_model(imag_feat))
        value = torch.mean(Model.value_model(imag_feat))

        reward = image_reward[:-1]
        discount_arr = self.discount * torch.ones_like(image_reward)
        value = value[:-1]
        bootstrap = value[-1]

        next_values = torch.cat([value[1:], bootstrap[None]], 0)
        target = reward + discount_arr[:-1] * next_values * (1 - self.discount_lambda)
        timesteps = list(range(reward.shape[0] - 1, -1, -1))
        outputs = []
        accumulated_reward = bootstrap
        for t in timesteps:
            inp = target[t]
            discount_factor = reward[t]
            accumulated_reward = inp + discount_factor * self.discount_lambda * accumulated_reward
            outputs.append(accumulated_reward)
        returns = torch.flip(torch.stack(outputs), [0])
        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        actor_loss = -torch.mean(discount * returns)

        with torch.no_grad():
            value_feat = imag_feat[:-1].detach()
            value_discount = discount.detach()
            value_target = returns.detach()

        value_pred = Model.value_model(value_feat)
        log_prob = value_pred.log_prob(value_target)
        value_loss = -torch.mean(value_discount * log_prob.unsqueeze(2))

        return model_loss, actor_loss, value_loss

    def OptimizeModel(self, chart_datas, actions, rewards):
        print("optimizing is starting.....")
        # setting optimizers

        ModelOptimizer = torch.optim.Adam(self.ModelParameters, lr=self.learning_rate)
        RewardOptimizer = torch.optim.Adam(self.trading_model.reward_model.parameters(), lr=self.learning_rate)
        ValueOptimizer = torch.optim.Adam(self.trading_model.value_model.parameters(), lr=self.learning_rate)

        for i in trange(self.train_steps):
            model_loss, actor_loss, value_loss = self.GetLoss(chart_datas, actions, rewards)

            model_loss.backward()
            actor_loss.backward()
            value_loss.backward()

            torch.nn.utils.clip_grad_norm(self.ModelParameters, self.gradient_clip)
            torch.nn.utils.clip_grad_norm(self.trading_model.reward_model.parameters(), self.gradient_clip)
            torch.nn.utils.clip_grad_norm(self.trading_model.value_model.parameters(), self.gradient_clip)

            ModelOptimizer.step()
            RewardOptimizer.step()
            ValueOptimizer.step()

    def train_data(self, reward,): # incomplete
        return

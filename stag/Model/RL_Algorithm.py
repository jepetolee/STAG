import torch

from RL_Model import *


class Dreamer:
    def __init__(self, device, dtype=torch.float64):
        self.trading_model = TradingModel(output_size=3)
        self.device = device
        self.type = dtype
        self.kl_scale = 1
        self.free_nats = 3
        self.discount =0.99
        self.horizon = 15
        self.discount_lambda = 0.95


    def hypothesis(self, crypto_chart):
        return self.trading_model(crypto_chart)

    def loss(self, observation, action, reward):
        Model = self.trading_model
        lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(observation, 3)
        batch_size = batch_t * batch_b

        observation = observation.type(self.type) / 255.0 - 0.5
        embed = Model.observation_encoder(observation)

        prev_state = Model.representation.initial_state(batch_b, device=self.device, dtype=self.type)
        prior, post = Model.rollout.rollout_representation(batch_t, embed, action, prev_state)

        feat = post.get_feature()
        image_pred = Model.observation_decoder(feat)
        reward_pred = Model.reward_model(feat)
        reward_loss = -torch.mean(reward_pred.log_prob(reward))
        image_loss = -torch.mean(image_pred.log_prob(observation))

        prior_dist = prior.get_distribution()
        post_dist = post.get_distribution()
        div = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
        div = torch.max(div, div.new_full(div.size(), self.free_nats))
        model_loss = self.kl_scale * div + reward_loss + image_loss

        with torch.no_grad():
            flat_post = buffer_method(post, 'reshape', batch_size, -1)
        with FreezeParameters(self.model_modules):
            imag_dist, _ = Model.rollout.RolloutPolicy(self.horizon,Model.policy,flat_post)
        imag_feat = imag_dist.get_distribution()

        with FreezeParameters(self.model_modules + self.value_modules):
            imag_reward = Model.reward_model(imag_feat).mean
            value = Model.value_model(imag_feat).mean

        discount_arr = self.discount * torch.ones_like(imag_reward)
        returns = self.compute_return(imag_reward[:-1], value[:-1], discount_arr[:-1],
                                      bootstrap=value[-1], lambda_=self.discount_lambda)
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

        with torch.no_grad():
            prior_ent = torch.mean(prior_dist.entropy())
            post_ent = torch.mean(post_dist.entropy())
            loss_info = LossInfo(model_loss, actor_loss, value_loss, prior_ent, post_ent, div, reward_loss, image_loss,
                                 pcont_loss)

        return model_loss, actor_loss, value_loss, loss_info

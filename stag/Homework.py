import torch
import torch.nn as nn
import torch.optim as optim
import gym
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# GPT-2 기반의 Transformer 정책
class TransformerPolicy(nn.Module):
    def __init__(self, model_name="gpt2"):
        super(TransformerPolicy, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def forward(self, input_ids):
        return self.model(input_ids).logits[:, -1, :]


# PPO 알고리즘
def ppo_update(policy, optimizer, states, actions, rewards, advantages, epsilon_clip=0.2, epochs=10):
    for _ in range(epochs):
        logits = policy(states)
        action_probs = torch.softmax(logits, dim=-1)
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        old_logits = states.clone().detach()
        old_logits.requires_grad = False
        old_action_probs = torch.softmax(old_logits, dim=-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        ratio = selected_action_probs / (old_action_probs + 1e-8)

        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - epsilon_clip, 1.0 + epsilon_clip) * advantages
        surrogate_loss = -torch.min(surrogate1, surrogate2).mean()

        optimizer.zero_grad()
        surrogate_loss.backward()
        optimizer.step()


# RLHF 훈련 루프
def rlhf_train(env, policy, optimizer, num_episodes=2, human_feedback_episodes=10):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    for episode in range(num_episodes):
        # 강화 학습 데이터 수집
        states = []
        actions = []
        rewards = []
        advantages = []

        state = env.reset()
        done = False

        while not done:
            input_ids = tokenizer.encode(state, return_tensors="pt")
            action = policy(input_ids).argmax().item()

            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # 얻은 보상을 사용하여 어드밴티지 계산
        returns = []
        advantage = 0
        for r in reversed(rewards):
            advantage = 0.99 * advantage + r
            returns.insert(0, advantage)

        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - torch.tensor(rewards, dtype=torch.float32)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO를 사용하여 정책 업데이트
        ppo_update(policy, optimizer, states, actions, rewards, advantages)

        # 일정 주기로 인간 피드백 수집 및 모델 훈련
        if episode % human_feedback_episodes == 0:
            human_feedback = collect_human_feedback(env, policy)
            for state, human_action in human_feedback:
                input_ids = tokenizer.encode(state, return_tensors="pt")
                action_probs = policy(input_ids)
                loss = nn.CrossEntropyLoss()(action_probs.unsqueeze(0), torch.tensor([human_action]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 훈련된 모델을 사용하여 다시 PPO 업데이트
            ppo_update(policy, optimizer, states, actions, rewards, advantages)

        if episode % 10 == 0:
            total_reward = sum(rewards)
            print(f"Episode {episode}, Total Reward: {total_reward}")


# 인간 피드백 수집
def collect_human_feedback(env, policy, num_episodes=5):
    human_feedback = []
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            input_ids = tokenizer.encode(state, return_tensors="pt")
            action = policy(input_ids).argmax().item()

            # 인간 피드백 수신
            human_action = int(input("Enter the correct action: "))
            human_feedback.append((state, human_action))

            next_state, _, done, _ = env.step(action)
            state = next_state
    return human_feedback


if __name__ == "__main__":
    # 인간 피드백을 받아들이는 환경을 사용자 정의하십시오.
    class HumanFeedbackEnv:
        def __init__(self):
            self.state = 0
            self.done = False

        def reset(self):
            self.state = input()
            self.done = False
            return self.state

        def step(self, action):
            if self.done:
                raise ValueError("Episode is done. Please reset the environment.")

            # 특정한 행동을 취하면 보상으로 2를 받음
            if action == "CUTTY":
                reward = 2
            else:
                reward = 0

            # 환경 상태 업데이트

            self.state = str(self.state) + str(action)

            # 간단한 종료 조건 설정
            if action == "END":
                self.done = True

            return self.state, reward, self.done, {}


    env = HumanFeedbackEnv()

    # GPT-2 기반의 Transformer 정책 및 PPO 옵티마이저 초기화
    policy = TransformerPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # RLHF 훈련 실행
    rlhf_train(env, policy, optimizer, num_episodes=2, human_feedback_episodes=10)

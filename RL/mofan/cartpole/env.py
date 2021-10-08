# tt
# 2021.9.30
# Main loop for CartPole game with DQN and gym

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from DQN import DQN

def demo():
    env = gym.make("CartPole-v0")  # Using cartpole env in gym
    env = env.unwrapped  # Remove some limits Todo (tt): figure out what kind of limits
    env.reset()
    # for _ in range(1000):
    #     env.render()
    #     env.step(env.action_space.sample())

    print(env.action_space.n)  # 查看这个环境中可用的 action 有多少个
    print(env.observation_space)  # 查看这个环境中可用的 state 的 observation
    print(env.observation_space.high)  # 查看 observation 最高取值
    print(env.observation_space.low)  # 查看 observation 最低取值

    observation = env.reset()
    print(f"observation: {observation}")
    observation -= env.observation_space.low
    print(f"observation: {observation}")
    observation /= env.observation_space.high
    observation /= 2.0
    print(f"observation: {observation}")

def train(num_episode=100, max_step=1000, e=0.8, gamma=0.8, memory_step=20):
    dqn = DQN(lr=1e-2)
    env = gym.make("CartPole-v0")  # Using cartpole env in gym
    env = env.unwrapped  # Remove some limits Todo (tt): figure out what kind of limits
    actions = range(env.action_space.n)

    costs = []

    def normalize(ob):
        nonlocal env
        ob = ob[0] # Since ob may be tensor(1, xxxx, xxx ... )
        ob -= env.observation_space.low
        ob /= env.observation_space.high
        ob /= 2.0
        return ob


    for ep in range(1, num_episode + 1):
        step = 0
        observation = env.reset() # Get 1st observation
        while True:
            env.render()
            step += 1
            observation = torch.Tensor([observation])
            observation[0] = normalize(observation)
            # 1. Choose an action with learning net
            chosen = -1
            if np.random.random() > e: # e-greedy
                chosen = np.random.choice(actions)
            else:
                max_q_value = -1.0e6
                for action in actions:
                    input = torch.cat((observation, torch.Tensor([[action]])), dim=1)
                    q_value = dqn.get_predict(input)[0]
                    if q_value > max_q_value:
                        max_q_value = q_value
                        chosen = action

            pred_value = dqn.get_predict(torch.cat((observation, torch.Tensor([[chosen]])), dim=1)) # Do it one more time to ensure correct bp

            # 2. Perform this action and get the corresponding state
            observation_, _, done, _ = env.step(chosen)
            x, x_dot, theta, theta_dot = observation_

            # 3. Get reward
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = torch.Tensor([r1 + r2]).cuda()

            # 4. Get actual value (reward + the max next value)
            actual_value = torch.Tensor([-1.0]).cuda()
            if done:
                pass # Died
            else:
                observation = torch.Tensor([observation_])
                observation[0] = normalize(observation)
                next_value = torch.Tensor([-1.0e6])
                for action in actions:
                    input = torch.cat((observation, torch.Tensor([[action]])), dim=1)
                    q_value = dqn.get_next_value(input)
                    next_value = q_value if q_value[0] > next_value[0] else next_value
                actual_value = reward + gamma * next_value

            cost = dqn.learn(pred_value, actual_value)
            costs.append(cost.cpu().detach())

            if done or step % memory_step == 0:
                dqn.store_memory()

            if done or step == max_step:
                print(f"Episode{ep} finished in {step} steps")
                break

            observation = observation_
    plt.plot(costs)
    plt.show()



if __name__ == "__main__":
    train(num_episode=1000)
    # demo()

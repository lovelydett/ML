'''
Author: tt
Date: 2021.10.8
Description: Try gym basic operations and play mountain car game
'''

import gym

class Agent:
    def __init__(self, env):
        pass
    def choose_action(self, observation, is_egreedy=False):
        '''
        Decide what to do now
        :param observation: current observation
        :param is_egreedy: should we explore (in training)
        :return: the action to take
        '''
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
                0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 +0.06
        action = 2 if lb < velocity < ub else 0
        return action
    def learn(self, observation, action, reward):
        pass

def check_envs():
    from gym import envs
    env_specs = envs.registry.all()
    env_ids = [s.id for s in env_specs]
    print(f"envs: {env_ids}")

def mountain_car():
    # Make an environment
    env = gym.make("MountainCar-v0")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}") # So possible actions are {0, 1, 2}
    print(f"Num of actions: {env.action_space.n}")
    print(f"Observation ub: {env.observation_space.high}")
    print(f"Observation lb: {env.observation_space.low}")

    # Initialize the environment, return first observation, an np.array
    first_ob = env.reset()
    print(f"first_ob: {first_ob}")

    # Get all possible actions
    actions = env.action_space

    # Get an agent
    agent = Agent(env)

    # Start playing
    print("--------Start playing--------")
    total_reward = 0
    for _ in range(100):
        observation = env.reset()
        while True:
            env.render() # Display game GUI
            action = agent.choose_action(observation)
            next_observation, reward, done, _ = env.step(action) # Do action and get new state descriptors
            total_reward += reward
            agent.learn(observation, action, reward)
            if done:
                break
            observation = next_observation
    print(f"Average episode reward for 100 episodes: {total_reward}")
    env.close()

if __name__ == "__main__":
    # check_envs()
    mountain_car()
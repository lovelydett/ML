# tt
# 2021.9.15
# A very simple Q-Learning example

import numpy as np
import pandas as pd
import time

def demo():
    n_states = 12  # 1-dimensional world width
    actions = ['left', 'right']  # possible actions
    epsilon = 0.8  # greedy
    alpha = 0.2  # learning rate
    gamma = 0.9  # decreasing rate of reward
    MAX_EPISODES = 100

    # initialize a Q-Table: state - action mapping
    table = pd.DataFrame(
        np.random.normal(0, 0.5, (n_states, len(actions))),     # q_table normal distribution
        columns=actions,    # columns
    )

    # current state (location)
    state = 0

    # pick and execute an action at a state
    def choose_action():
        possible_actions = table.iloc[state, :]
        # at epsilon portion of time we greedily pick the largest action, the other time we randomly pick
        idx = np.random.randint(0, len(actions)) if np.random.uniform() > epsilon else possible_actions.argmax()
        return actions[idx]

    # return the reward of an action at a state
    def get_reward(action):
        if state == n_states - 1 and action == "right":
            return 1
        return 0

    def print_current_state():
        for i in range(0, state):
            print("-", end='')
        print("o",end='')
        for i in range(state + 1, n_states):
            print("-", end='')
        if state != n_states:
            print("T", end='')
        print("")

    # main loop: learning process here!
    for i in range(0, MAX_EPISODES):
        state = 0 # reset state
        counter = 0 # count how many times it moves
        while state != n_states:
            # print_current_state()
            counter += 1
            # 1. pick an action
            action = "right" if state == 0 else choose_action()
            # 2. get reward for this action at this state
            reward = get_reward(action)
            # 3. get predicted Q value for this action at this state
            q_pred = table.loc[state, action] # 3. get predicted Q value for this action at this state

            new_state = state + (1 if action == "right" else -1)
            # 4. get actual Q value for this
            q_target = 0
            if new_state == n_states: # finish this episode
                q_target = reward
            else: # not yet finish, we use current reward plus the max possible reward in new_state!
                q_target = reward + table.iloc[new_state, :].max() * gamma

            # 5. update Q table for (state, action) by learning
            table.loc[state, action] += alpha * (q_target - q_pred)

            state = new_state
            # print_current_state()
            # time.sleep(0.3)

        print(f"Episode{i + 1}: {counter} actions made to reach goal")
    print(f"Final Q-Table:\n{table}")

if __name__ == "__main__":
    demo()
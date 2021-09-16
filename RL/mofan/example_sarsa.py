# tt
# 2021.9.15
# A very simple Sarsa example
# Sarsa decides A' for S' when deciding A at S, and use the feed back to update Q(S, A)
# Sarsa is on-policy since it already decides its next state and next action at current state and action

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
        np.random.normal(0, 0.8, (n_states, len(actions))),     # q_table normal distribution
        columns=actions,    # columns
    )

    # current state (location)
    state = 0

    # pick and execute an action at a state
    def choose_action(state):
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
        state = 0
        # get initial action
        action = "right"
        counter = 0
        while state != n_states:
            counter += 1
            # 1. take this action
            next_state = state + (1 if action == "right" else -1)
            if next_state == n_states:
                break # we finished
            # 2. pick next action for new state, this has actually decided next action to take
            next_action = "right" if next_state == 0 else choose_action(next_state)
            # 3. update Q(S, A) by current reward and next predict Q(S', A')
            cur_pred = table.loc[state, action]
            next_pred = table.loc[next_state, next_action]
            cur_reward = get_reward(action)
            table.loc[state, action] += alpha * (cur_reward + gamma * next_pred - cur_pred)
            # 4. update state and action since we actually are gonna take next_action
            state = next_state
            action = next_action

            # print_current_state()
            # time.sleep(0.1)

        print(f"Episode{i + 1}: {counter} actions made to reach goal")
    print(f"Final Q-Table:\n{table}")

if __name__ == "__main__":
    demo()
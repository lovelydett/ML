# tt
# 2021.9.15
# A very simple Sarsa-lambda example
# Sarsa-lambda or Sarsa(lambda) is when reward received at current (S, A)
# not only update Q(S, A), but also update those steps that led us here at a rate lambda
# lambda == 0: only update Q(S, A)
# lambda > 0: update previous Q(S, A) at decaying rate (1 - lambda)

import numpy as np
import pandas as pd
import time

def demo():
    n_states = 12  # 1-dimensional world width
    actions = ['left', 'right']  # possible actions
    epsilon = 0.8  # greedy
    alpha = 0.2  # learning rate
    gamma = 0.9  # decreasing rate of reward
    lamb = 0.6 # Sarsa(lamb)
    MAX_EPISODES = 100

    # initialize a Q-Table: state - action mapping
    q_table = pd.DataFrame(
        np.random.normal(0, 0.8, (n_states, len(actions))),     # q_table normal distribution
        columns=actions,    # columns
    )

    # initialize an E-Table: decaying rates for each (S, A)
    e_table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )

    # current state (location)
    state = 0

    # pick and execute an action at a state
    def choose_action(state):
        possible_actions = q_table.iloc[state, :]
        # at epsilon portion of time we greedily pick the largest action, the other time we randomly pick
        idx = np.random.randint(0, len(actions)) if np.random.uniform() > epsilon else possible_actions.argmax()
        return actions[idx]

    # return the reward of an action at a state
    def get_reward(action):
        if state == n_states - 1 and action == "right":
            return 1
        if state == 1 and action == "left":
            return -1
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
            # 3. get actual and pred values to calculate delta
            cur_pred = q_table.loc[state, action]
            next_pred = q_table.loc[next_state, next_action]
            cur_reward = get_reward(action)
            delta = cur_reward + gamma * next_pred - cur_pred
            # 4. mark current (S, A) as the last step
            e_table.loc[state, action] += 1
            # 5. update all steps with delta
            for s in range(n_states):
                for a in actions:
                    q_table.loc[s, a] += alpha * delta * e_table.loc[s, a]
                    e_table.loc[s, a] *= lamb
            # 6. update state and action since we actually are gonna take next_action
            state = next_state
            action = next_action

            # print_current_state()
            # time.sleep(0.1)

        print(f"Episode{i + 1}: {counter} actions made to reach goal")
    print(f"Final Q-Table:\n{q_table}")

if __name__ == "__main__":
    demo()
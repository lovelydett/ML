'''
Author: tt
Date: 2021.10.20
Description: Walk in a maze with Sarsa
'''

import numpy as np
import matplotlib.pyplot as plt

def main():

    # Initial policy params: [up, right, down, left]
    def get_theta():
        return np.array([[np.nan, 1, 1, np.nan],  # s0
                        [np.nan, 1, np.nan, 1],  # s1
                        [np.nan, np.nan, 1, 1],  # s2
                        [1, 1, 1, np.nan],  # s3
                        [np.nan, np.nan, 1, 1],  # s4
                        [1, np.nan, np.nan, np.nan],  # s5
                        [1, np.nan, np.nan, np.nan],  # s6
                        [1, 1, np.nan, np.nan],  # s7
                        ])

    # Convert theta to policy PI(s, a), which is a probability
    def softmax_convert_theta_to_pi(theta):
        beta = 1.0
        theta = np.exp(beta * theta)
        m, n = theta.shape
        pi = np.zeros((m, n))
        for i in range(m):
            row_total = np.nansum(theta[i])
            for j in range(n):
                pi[i][j] = theta[i][j] / row_total
        pi = np.nan_to_num(pi)
        return pi

    # Get action and next state given current state and policy
    # [up, right, down, left]
    moves = [-3, +1, +3, -1]
    def get_action(Q, pi, state, e):
        # At probability e, explore based on policy PI(a | s)
        if np.random.random() < e:
            action = np.random.choice(range(len(moves)), p=pi[state, :])
        # Else greedily exploit the argmax action for current state
        else:
            action = np.nanargmax(Q[state, :])
        return action

    def get_next_state(state, action):
        return state + moves[action]


    def Sarsa(MAX_EPISODE=100, gamma=0.9, lr=1e-2):
        # 8 states, each has 4 actions
        theta = get_theta()
        # Q-table must multiply by theta to introduce nan!
        Q = np.random.normal(0, 1, theta.shape) * theta
        e = 0.9
        pi = softmax_convert_theta_to_pi(theta)  # Convert theta params to a policy PI(s, a)
        for ep in range(1, MAX_EPISODE + 1):
            state = 0
            step = 0
            action = get_action(Q, pi, state, e) # Get an initial action
            while state != 8:
                step += 1
                next_state = get_next_state(state, action)
                next_action = get_action(Q, pi, next_state, e) if next_state != 8 else -1
                if next_state == 8:
                    actual_gain = 1
                else:
                    actual_gain =  0 + gamma * Q[next_state][next_action]

                TD_error = actual_gain - Q[state][action]
                Q[state][action] += lr * (TD_error)

                action = next_action
                state = next_state

            print(f"Episode{ep}: {step} steps to arrive")
            print(f"Current Q-table:\n{Q}")

            e /= 2 # Reduce e-greedy,

    Sarsa(MAX_EPISODE=1000)

if __name__ == "__main__":
    main()
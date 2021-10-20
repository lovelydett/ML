'''
Author: tt
Date: 2021.10.20
Description: Walk in a maze with Policy Gradient
'''

import numpy as np
import matplotlib.pyplot as plt

def main():
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    plt.plot([1, 1], [0, 1], color='red', linewidth=2)
    plt.plot([1, 2], [2, 2], color='red', linewidth=2)
    plt.plot([2, 2], [2, 1], color='red', linewidth=2)
    plt.plot([2, 3], [1, 1], color='red', linewidth=2)

    plt.text(0.5, 2.5, 'S0', size=14, ha='center')
    plt.text(1.5, 2.5, 'S1', size=14, ha='center')
    plt.text(2.5, 2.5, 'S2', size=14, ha='center')
    plt.text(0.5, 1.5, 'S3', size=14, ha='center')
    plt.text(1.5, 1.5, 'S4', size=14, ha='center')
    plt.text(2.5, 1.5, 'S5', size=14, ha='center')
    plt.text(0.5, 0.5, 'S6', size=14, ha='center')
    plt.text(1.5, 0.5, 'S7', size=14, ha='center')
    plt.text(2.5, 0.5, 'S8', size=14, ha='center')
    plt.text(0.5, 2.3, 'START', ha='center')
    plt.text(2.5, 0.3, 'GOAL', ha='center')

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    labelbottom='off', right='off', left='off', labelleft='off')

    line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)
    # plt.show()

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
    def get_action_and_next_state(pi, state):
        # [up, right, down, left]
        moves = [-3, +1, +3, -1]
        # Choose an action with current policy PI(s, a)
        action = np.random.choice(range(len(moves)), p=pi[state, :])
        return action, state + moves[action]

    # Do policy gradient once, with a history of an episode
    def learn_from_history(theta, pi, history, lr=1e-1):
        total_step = len(history)
        m, n = theta.shape
        delta_theta = theta.copy()

        for i in range(m):
            N_si = len([1 for each in history if each[0] == i])
            for j in range(n):
                N_si_aj = len([1 for each in history if each == [i, j]])
                delta_theta[i][j] = (N_si_aj - pi[i][j] * N_si) / total_step
        theta += lr * delta_theta
        return theta


    def train(MAX_EPISODE=100):
        theta = get_theta()
        for ep in range(1, MAX_EPISODE + 1):
            pi = softmax_convert_theta_to_pi(theta) # Convert theta params to a policy PI(s, a)
            state = 0
            history = []
            while True:
                action, next_state = get_action_and_next_state(pi, state)
                history.append([state, action])
                state = next_state
                if state == 8:
                    break
            history.append([state, np.nan])  # Last (s, a) pair should also remembered

            # Do policy gradient when an episode finishes, update params (thetas)
            theta = learn_from_history(theta, pi, history)

            print(f"Episode{ep}: {len(history)} steps to arrive")
            print(f"Current policy:\n{pi}")

    train(MAX_EPISODE=1000)

if __name__ == "__main__":
    main()
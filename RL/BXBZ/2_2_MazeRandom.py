'''
Author: tt
Date: 2021.10.20
Description: Randomly walk in a maze
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
    plt.show()

    # Initial policy params: [up, right, down, left]
    theta_0 = np.array([[np.nan, 1, 1, np.nan],  # s0
                        [np.nan, 1, np.nan, 1],  # s1
                        [np.nan, np.nan, 1, 1],  # s2
                        [1, 1, 1, np.nan],  # s3
                        [np.nan, np.nan, 1, 1],  # s4
                        [1, np.nan, np.nan, np.nan],  # s5
                        [1, np.nan, np.nan, np.nan],  # s6
                        [1, 1, np.nan, np.nan],  # s7
                        ])

    # Convert theta to policy PI(s, a), which is a probability
    def simple_convert_theta_to_pi(theta):
        m, n = theta.shape
        pi = np.zeros((m, n))
        for i in range(m):
            row_total = np.nansum(theta[i])
            for j in range(n):
                pi[i][j] = theta[i][j] / row_total
        pi = np.nan_to_num(pi)
        return pi

    # Get next state given current state and policy
    def get_next_state(pi, state):
        # [up, right, down, left]
        moves = [-3, +1, +3, -1]
        # Choose an action with current policy PI(s, a)
        next_move = np.random.choice(moves, p=pi[state, :])
        return state + next_move

    # Run game
    # Get initial policy PI(s, a)
    pi_0 = simple_convert_theta_to_pi(theta_0)
    print(f"Initial policy:\n{pi_0}")
    state = 0
    steps = 0
    while True:
        state = get_next_state(pi_0, state)
        steps += 1
        if state == 8:
            break
    print(f"Run {steps} steps to arrive")


if __name__ == "__main__":
    main()
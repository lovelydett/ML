'''
Author: tt
Date: 2021.10.20
Description: Walk in a maze with Q-Learning
'''

import numpy as np

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


    def QLearning(MAX_EPISODE=100, gamma=0.9, lr=1e-2, e=0.5):
        # 8 states, each has 4 actions
        theta = get_theta()
        # Q-table must multiply by theta to introduce nan!
        Q = np.random.normal(0, 1, theta.shape) * theta
        pi = softmax_convert_theta_to_pi(theta)  # Convert theta params to a policy PI(s, a)
        for ep in range(1, MAX_EPISODE + 1):
            state = 0
            step = 0
            while state != 8:
                step += 1
                action = get_action(Q, pi, state, e) # Choose an action with e-greedy
                next_state = get_next_state(state, action) # Get next state for current action
                max_next_action = get_action(Q, pi, next_state, e=0) if next_state != 8 else -1 # e=0 means no need to explore in argmax(a)Q(s', a) part
                if next_state == 8:
                    actual_gain = 1
                else:
                    actual_gain =  0 + gamma * Q[next_state][max_next_action]

                TD_error = actual_gain - Q[state][action]
                Q[state][action] += lr * (TD_error)

                state = next_state

            print(f"Episode{ep}: {step} steps to arrive, e-greedy = {e}")

            if ep > MAX_EPISODE // 2:
                e *= 0.999 # Reduce e-greedy

        print(f"Final Q-table:\n{Q}")
        V = np.nansum(pi * Q, axis=1)
        print(f"Final V(s) table:\n{V}")

    QLearning(MAX_EPISODE=1000)

if __name__ == "__main__":
    main()
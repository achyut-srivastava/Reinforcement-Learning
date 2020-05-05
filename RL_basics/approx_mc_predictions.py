import numpy as np
import matplotlib.pyplot as plt

from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_policy, print_values
from monte_carlo_random import random_action, play_game, SMALL_ENOUGH, GAMMA, ALL_POSSIBLE_ACTIONS

LEARNING_RATE = 0.001

if __name__=="__main__":
    grid = standard_grid()

    print("Rewards: ")
    print_values(grid.rewards, grid)

    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'L',
        (2, 2): 'U',
        (2, 3): 'L',
    }

    theta = np.random.randn(4)/2
    # V = theta.dot(x)
    def s2x(s):
        return np.array([s[0]-1, s[1]-1.5, s[0]*s[1]-3, 1])

    delta = []
    t = 1.0
    for it in range(10000):
        if it % 100 == 0:
            t += 10e-3
        biggest_change = 0
        alpha = LEARNING_RATE / t
        state_return = play_game(grid, policy)
        seen_states = set()
        for s, G in state_return:
            if s not in seen_states:
                old_theta = theta.copy()
                x = s2x(s)
                V_hat = theta.dot(x)
                theta += alpha*(G - V_hat)*x
                biggest_change = max(biggest_change, np.abs(old_theta - theta).sum())
                seen_states.add(s)
        delta.append(biggest_change)

    plt.plot(delta)
    plt.show()

    V = {}
    states = grid.all_state()
    for s in states:
        if s in grid.actions:
            V[s] = theta.dot(s2x(s))
        else:
            V[s] = 0

    print_values(V, grid)
    print_policy(policy, grid)




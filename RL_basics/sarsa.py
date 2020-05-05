import numpy as np
import matplotlib.pyplot as plt

from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_policy, print_values
from monte_carlo_es import max_dict
from td0_predictions import random_action

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'L', 'R', 'D')

if __name__ =="__main__":
    grid = negative_grid(-0.1)

    print("Rewards: ")
    print_values(grid.rewards, grid)

    Q = {}
    update_counts = {}
    update_counts_sa = {}
    states = grid.all_state()
    for s in states:
        Q[s]={}
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0
            update_counts_sa[s][a] = 1.0

    t = 1.0
    delta = []
    for it in range(10000):
        if it%100 == 0:
            t += 10e-3

        s = (2, 0)
        grid.set_state(s)

        a = max_dict(Q[s])[0]
        a = random_action(a, eps = 0.5/t)

        biggest_change = 0
        while not grid.game_over():
            r = grid.move(a)
            s2 = grid.current_state()

            a2 = max_dict(Q[s2])[0]
            a2 = random_action(a2, eps=0.5/t)

            alpha = ALPHA/update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005

            old_qsa = Q[s][a]
            Q[s][a] = Q[s][a] + alpha*(r + GAMMA*Q[s2][a2] - Q[s][a])
            biggest_change = max(biggest_change, np.abs(old_qsa-Q[s][a]))

            update_counts[s] = update_counts.get(s, 0) + 1

            s = s2
            a = a2
            delta.append(biggest_change)

    plt.plot(delta)
    plt.show()

    V= {}
    policy ={}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    print_values(V, grid)
    print_policy(policy, grid)
    print(update_counts)


    pass


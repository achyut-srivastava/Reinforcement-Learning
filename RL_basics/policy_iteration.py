import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

if __name__ =="__main__":
    grid = negative_grid(-.10)

    print("rewards: ")
    print_values(grid.rewards, grid)

    states = grid.all_state()

    print("\ninitial initalization: ")
    policy ={}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    print_policy(policy, grid)

    # Value Initialization
    V = {}
    for s in states:
        V[s] = 0

    while True:
        #policy evaluation
        while True:
            for s in states:
                delta = 0
                old_Vs = V[s]
                if s in policy:
                    a = policy[s]
                    grid.set_state(s)
                    r = grid.move(a)
                    V[s] = r + GAMMA*(V[grid.current_state()])
                    delta = max(delta, np.abs(old_Vs-V[s]))
            if delta < SMALL_ENOUGH:
                break

        is_policy_converged = True
        for s in states:
            if s in policy:
                old_a = policy[s]
                best_value = float('-inf')
                for a in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA *(V[grid.current_state()])
                    if v > best_value:
                        best_value = v
                        policy[s] = a
                if policy[s] != old_a:
                    is_policy_converged = False

        if is_policy_converged:
            break

    print_values(V, grid)
    print("\n\n")
    print_policy(policy, grid)


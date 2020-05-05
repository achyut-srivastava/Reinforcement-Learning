import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

if __name__ == "__main__":
    grid = negative_grid(-.1)

    print("rewards: ")
    print_values(grid.rewards, grid)

    states = grid.all_state()

    print("\ninitial initalization: ")
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    print_policy(policy, grid)

    # Value Initialization
    V = {}
    for s in states:
        V[s] = 0

    while True:
        delta = 0
        for s in states:
            old_Vs = V[s]
            if s in policy:
                old_a = policy[s]
                new_V = float('-inf')
                new_a = None
                for a in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA * (V[grid.current_state()])
                    if v > new_V:
                        new_V = v
                        new_a = a
                V[s] = new_V
                policy[s] = new_a
                delta = max(delta, np.abs(old_Vs-V[s]))
        if delta < SMALL_ENOUGH:
            break

    print("\n\n")
    print_values(V, grid)
    print("\n\n")
    print_policy(policy, grid)
    pass


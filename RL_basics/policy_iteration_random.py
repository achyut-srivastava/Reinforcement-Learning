import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

if __name__ == "__main__":
    grid = negative_grid(-1)

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
        # policy evaluation
        while True:
            delta = 0
            for s in states:
                old_Vs = V[s]
                new_Vs=0
                if s in policy:
                    for a in ALL_POSSIBLE_ACTIONS:
                        if a == policy[s]:
                            p_a = 0.5
                        else:
                            p_a = (0.5 / 3)
                        grid.set_state(s)
                        r = grid.move(a)
                        new_Vs += p_a * (r + GAMMA * (V[grid.current_state()]))
                    V[s] = new_Vs
                    delta = max(delta, np.abs(old_Vs - V[s]))
            if delta < SMALL_ENOUGH:
                break

        is_policy_converged = True
        for s in states:
            if s in policy:
                old_a = policy[s]
                best_value = float('-inf')
                for a in ALL_POSSIBLE_ACTIONS:
                    v = 0
                    for a2 in ALL_POSSIBLE_ACTIONS:
                        if a2 == a:
                            p_a = 0.5
                        else:
                            p_a = (0.5 / 3)
                        grid.set_state(s)
                        r = grid.move(a2)
                        v += p_a * (r + GAMMA * (V[grid.current_state()]))
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


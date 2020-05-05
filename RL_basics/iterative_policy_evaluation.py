import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid

SMALL_ENOUGH = 10e-4


def print_values(V, g):
    for i in range(g.rows):
        print("-" * 20)
        for j in range(g.cols):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" %v, end = "")
            else:
                print("%.2f|" %v, end="")
        print("")
    print("\n")


def print_policy(P, g):
    for i in range(g.rows):
        print("-" * 20)
        for j in range(g.cols):
            p = P.get((i, j), ' ')
            print(" %s |" %p, end = "")
        print("")
    print("\n")

if __name__=="__main__":
    grid = standard_grid()
    states = grid.all_state()

    # random actions
    V={}
    for s in states:
        V[s] = 0
    gamma = 1 # discount factor

    while True:
        for s in states:
            delta = 0
            old_Vs = V[s]
            if s in grid.actions:
                new_Vs = 0
                best = float('-inf')
                pa_s = 1.0/len(grid.actions[s])
                for a in grid.actions[s]:
                    grid.set_state(s)
                    r = grid.move(a)
                    new_Vs += pa_s * (r + gamma * V[grid.current_state()])
                V[s] = new_Vs
                delta = max(delta, np.abs(old_Vs - V[s]))
        if delta < SMALL_ENOUGH:
            break
    #
    print("Values for uniformly random actions: ")
    print_values(V, grid)
    print("\n\n")

    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }
    print_policy(policy, grid)

    print("Fixed policy/actions: ")
    print_policy(policy, grid)
    print("\n\n")

    V = {}
    for s in states:
        V[s] = 0

    gamma = 0.9

    count = 0
    while True:
        delta = 0
        count += 1
        for s in states:
            old_Vs = V[s]
            if s in policy:
                grid.set_state(s)
                V[s] = grid.move(policy[s]) + gamma * (V[grid.current_state()])
                delta = max(delta, np.abs(old_Vs-V[s]))

        if delta < SMALL_ENOUGH:
            break

    print("Fixed policy values: ")
    print_values(V, grid)
    print("\n\n")
    print(count)
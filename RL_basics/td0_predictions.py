import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_policy, print_values

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_OUTCOMES = ('U', 'L', 'R', 'D')


def random_action(a, eps=0.1):
    if np.random.random() < (1-eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_OUTCOMES)


def play_game(grid, policy):
    s = (2, 0)
    grid.set_state(s)
    state_return = [(s, 0)]
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        state_return.append((s, r))
    return state_return


if __name__ == "__main__":
    grid = negative_grid()
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

    V = {}
    for s in grid.all_state():
        V[s] = 0

    for i in range(7000):
        state_return = play_game(grid, policy)
        for s_p in range(len(state_return)-1):
            s, _ = state_return[s_p]
            s1, r = state_return[s_p+1]
            V[s] = V[s] + ALPHA * (r + GAMMA * V[s1] - V[s])
    print_values(V, grid)
    print_policy(policy, grid)


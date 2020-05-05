import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_policy, print_values

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_STATES = ('U', 'L', 'D', 'R')


def play_game(grid, policy):

    start_states = list(grid.actions.keys())
    start_id = np.random.choice(len(grid.actions))

    grid.set_state(start_states[start_id])
    s = grid.current_state()

    states_rewards = [(s, 0)]
    while not grid.game_over():
        a = policy[s]
        r = grid.move(a)
        s = grid.current_state()
        states_rewards.append([s,r])

    G = 0
    first = True
    states_returns = []
    for (s,r) in reversed (states_rewards):
        if first: first = False
        else: states_returns.append((s, G))
        G = r + GAMMA * G
    states_returns.reverse()
    return states_returns


if __name__ == "__main__":
    grid = standard_grid()
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
    returns = {}
    states = grid.all_state()
    for s in states:
        if s in grid.actions:
            returns[s] = []
        else: V[s] = 0

    for t in range(100):
        states_returns = play_game(grid, policy)
        seen_states= set()
        for (s, G) in states_returns:
            if s not in seen_states:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                seen_states.add(s)

    print_values(V, grid)
    print_policy(policy, grid)

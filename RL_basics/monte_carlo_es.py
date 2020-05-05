import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_policy, print_values

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'L', 'D', 'R')


def play_game(grid, policy):

    start_states = list(grid.actions.keys())
    start_id = np.random.choice(len(start_states))
    grid.set_state(start_states[start_id])

    a = np.random.choice(ALL_POSSIBLE_ACTIONS)
    s = grid.current_state()

    states_actions_rewards = [(s, a, 0)]
    seen_states =set()
    num_steps = 0
    seen_states.add(grid.current_state())
    while True:
        num_steps += 1
        old_s = grid.current_state()
        r = grid.move(a)
        s = grid.current_state()
        if old_s == s:
            states_actions_rewards.append((s, None, -100))
            break
        elif s in seen_states:
            break
        elif grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = policy[s]
            states_actions_rewards.append((s,a,r))
        seen_states.add(s)
    G = 0
    first = True
    states_actions_returns = []
    for (s, a, r) in reversed (states_actions_rewards):
        if first: first = False
        else: states_actions_returns.append((s, a, G))
        G = r + GAMMA * G
    states_actions_returns.reverse()
    return states_actions_returns


def max_dict(d):
    max_key = list(d.keys())[0]
    max_value = d[max_key]
    for k, v in d.items():
        if v > max_value:
            max_value = v
            max_key = k
    return max_key, max_value

if __name__ == "__main__":
    grid = standard_grid()
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    V = {}
    Q = {}
    returns = {}
    states = grid.all_state()
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0
                returns[(s,a)] = []
        else: pass

    for t in range(10000):
        states_actions_returns = play_game(grid, policy)
        seen_states_actions_pairs= set()
        for s, a, G in states_actions_returns:
            sa = (s,a)
            if sa not in seen_states_actions_pairs:
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                seen_states_actions_pairs.add(sa)

        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]
            # V[s] = max_dict(Q[s])[1]

    print_policy(policy, grid)

    #
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]
    print_values(V, grid)
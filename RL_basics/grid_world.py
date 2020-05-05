import numpy as np
import matplotlib.pyplot as plt

# our grid
# _ _ _ +1
# _ X _ -1
# S _ _ _


class Grid:
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        self.rewards = rewards
        self.actions = actions

    def all_state(self):
        return set(self.rewards.keys() | self.actions.keys())

    def is_terminal(self, s):
        return s not in self.actions

    def current_state(self):
        return self.i, self.j

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def move(self, action):
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i += -1
            elif action == 'D':
                self.i += 1
            elif action == 'L':
                self.j += -1
            elif action == 'R':
                self.j += 1
        return self.rewards.get((self.i, self.j), 0)

    def undo_move(self, action):
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i += -1
        elif action == 'L':
            self.j += 1
        elif action == 'R':
            self.j += -1
        assert(self.current_state() in self.all_state())

    def game_over(self):
        return (self.i, self.j)not in self.actions


def standard_grid():
    g = Grid(3, 4, (2, 0))
    rewards = {
               (0, 3): 1,
               (1, 3): -1
               }
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'R', 'D'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('U', 'L', 'R'),
        (2, 3): ('U', 'L')
    }
    g.set(rewards, actions)
    return g


def negative_grid(step_cost=-0.1):
    g = standard_grid()
    g.rewards.update({
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost
    })
    return g

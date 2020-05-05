import numpy as np

LENGTH = 3


class Environment:
    def __init__(self):
        self.board = np.zeros((LENGTH, LENGTH))
        self.x = -1
        self.o = 1
        self.total_states = 3**(LENGTH*LENGTH)
        self.winner = None
        self.ended = False

    def game_over(self, force_recalculate=False):
        if not force_recalculate and self.ended:
            return self.ended

        # row check
        for i in range(LENGTH):
            for player in (self.x, self.o):
                 if self.board[i].sum() == player*LENGTH:
                    self.winner = player
                    self.ended = True
                    return True

        # column check
        for i in range(LENGTH):
            for player in (self.x, self.o):
                 if self.board[:,i].sum() == player*LENGTH:
                    self.winner = player
                    self.ended = True
                    return True

        #left diagonal check
        for i in range(LENGTH):
            for player in (self.x, self.o):
                if self.board.trace() == player * LENGTH:
                    self.winner = player
                    self.ended = True
                    return True

        #right diagonal check

        for player in (self.x, self.o):
            if np.fliplr(self.board).trace() == player * LENGTH:
                self.winner = player
                self.ended = True
                return True

            if np.all((self.board == 0 ) == False):
                self.winner = None
                self.ended = True
                return True

        self.winner = None
        return False

    def get_state(self):
        h = 0
        k = 0
        for i in range(LENGTH):
            for j in range(LENGTH):
                if self.board[i, j] == 0:
                    v = 0
                elif self.board[i, j] == self.x:
                    v = 1
                elif self.board[i, j] == self.o:
                    v = 2
                h += (3 ** k) * v
                k += 1
        return h

    def draw_board(self):
        for i in range(LENGTH):
            print("*"*10)
            for j in range(LENGTH):
                print("  ", end="")
                if self.board[i, j] == self.x:
                    print("X ", end="")
                elif self.board[i, j] == self.o:
                    print("O ", end="")
                else:
                    print("  ", end="")
            print("")
        print("*"*10)

    def is_empty(self, i ,j):
        return self.board[i, j] == 0

    def get_reward(self, sym):
        if not self.game_over():
            return 0

        return 1 if self.winner==sym else 0



class Agent:
    def __init__(self, eps=0.1, alpha=0.5):
        self.eps = eps
        self.alpha = alpha
        self.verbose = False
        self.state_history = []

    def setV(self, V):
        self.V = V

    def set_symbol(self, sym):
        self.sym = sym

    def set_verbose(self, v):
        self.verbose = v

    def reset_history(self):
        self.state_history = []

    def take_action(self, env):
        best_state = None
        possible_move =[]
        r = np.random.rand()
        if r < self.eps:
            if self.verbose: print("Random cell")
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i, j):
                        possible_move.append((i,j))

            idx = np.random.choice(len(possible_move))
            next_move = possible_move[idx]

        else:
            if self.verbose: print(":Greedy action:")
            pos2values={}
            next_move=None
            best_value = -1
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i, j):
                        env.board[i, j] = self.sym
                        state = env.get_state()
                        env.board[i, j] = 0
                        pos2values[(i, j)] = self.V[state]
                        if self.V[state] > best_value:
                            best_value = self.V[state]
                            best_state = state
                            next_move = (i , j)

            if self.verbose:
                for i in range(LENGTH):
                    print("*"*10)
                    for j in range(LENGTH):
                        if env.is_empty(i, j):
                            # print the value
                            print(" %.2f|" % pos2values[(i, j)], end="")
                        else:
                            print("  ", end="")
                            if env.board[i, j] == env.x:
                                print("X  |", end="")
                            elif env.board[i, j] == env.o:
                                print("O  |", end="")
                            else:
                                print("   |", end="")
                    print("")
                print("*"*10)

        env.board[next_move[0], next_move[1]] = self.sym

    def update_state_history(self, state):
        self.state_history.append(state)

    def update(self, env):
        reward = env.get_reward(self.sym)
        target = reward
        for prev in reversed(self.state_history):
            value = self.V[prev] + self.alpha * (target - self.V[prev])
            self.V[prev] = value
            target = value
        self.reset_history()

class Human:
    def __init__(self):
        pass

    def take_action(self, env):
        while True:
            move = input("Enter the coordinate i, j: ")
            i, j = move.split(',')
            i = int(i)
            j = int(j)
            if env.is_empty(i, j):
                env.board[i, j] = self.sym
                break

    def set_symbol(self, sym):
        self.sym = sym

    def update_state_history(self, state):
        pass

    def update(self, env):
        pass



def play_game(p1, p2, env, draw=False):
    current_player = None
    while not env.game_over():
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1

        if current_player == p1 and draw:
            env.draw_board()
        if current_player == p2 and draw:
            env.draw_board()

        current_player.take_action(env)

        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)

    if draw:
        env.draw_board()

    p1.update(env)
    p2.update(env)


def initialV_x(env, state_winner_triples):
    V = np.zeros(env.total_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.x:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v
    return V

def initialV_o(env, state_winner_triples):
    V = np.zeros(env.total_states)
    for state, winner, ended in state_winner_triples:
        if ended:
            if winner == env.o:
                v = 1
            else:
                v = 0
        else: v = 0.5
        V[state] = v
    return V

def get_hash_and_winner(env, i=0, j=0):
    results = []
    for permute in (0, env.x, env.o):
        env.board[i, j] = permute
        if j == 2:
            if i == 2:
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += get_hash_and_winner(env, i+1, 0)
            pass
        else:
            results += get_hash_and_winner(env, i, j+1)
    return results


if __name__ == "__main__":
    p1 = Agent()
    p2 = Agent()
    env = Environment()
    state_winner_triples = get_hash_and_winner(env)
    Vx = initialV_x(env, state_winner_triples)
    p1.setV(Vx)
    Vo = initialV_o(env, state_winner_triples)
    p2.setV(Vo)

    p1.set_symbol(env.x)
    p2.set_symbol(env.o)

    T=10000
    for i in range(T):
        if (i%500 == 0):
            print(i)
        play_game(p1,p2,Environment())

    human = Human()
    human.set_symbol(env.o)

    while True:
        p1.set_verbose(True)
        play_game(p1, human, Environment(), draw=2)

        answer = input("Play again? [Y/n]: ")
        if answer and answer.lower()[0] == 'n':
            break
import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2,.5,0.75]

class Bandit:
    def __init__(self, p):
        self.p = p
        self.current_mean = 0.
        self.count = 0.

    def pull_lever(self):
        return np.random.randn() < self.p

    def update(self, x):
        self.count += 1
        self.current_mean = (x + (self.count -1)*self.current_mean)/(self.count)

def experiments():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rewards = np.zeros(NUM_TRIALS)
    optimal_j = np.argmax([b.p for b in bandits])
    num_of_exploited = 0
    num_of_explored = 0
    num_optimal = 0
    print("Optimal j :", optimal_j)
    for i in range(NUM_TRIALS):
        # epsilon greedy
        
        if np.random.random() < EPS:
            num_of_explored += 1
            j = np.random.randint(len(bandits))
        else:
            num_of_exploited += 1
            j = np.argmax([b.current_mean for b in bandits])

        if j == optimal_j:
            num_optimal += 1

        x = bandits[j].pull_lever()
        # print("return x :",x)

        rewards[i] = x

        bandits[j].update(x)

    for b in bandits:
        print("mean_estimate: ",b.current_mean)

    print("Num of times explored: ", num_of_explored)
    print("Num of times exploited: " , num_of_exploited)
    print("Num of times optimal value: ", num_optimal)
    print("total reward: ",rewards.sum())
    print("Overall win rate: ",rewards.sum()/NUM_TRIALS)

    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.show()


if __name__ == "__main__":
    experiments()
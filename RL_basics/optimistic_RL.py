import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS = 100000

class banditArm:
    def __init__(self, p):
        self.p = p
        self.p_mean = 5.
        self.count = 1.

    def pull_lever(self):
        return np.random.random() < self.p

    def update(self,x):
        self. count+=1
        self.p_mean = (x + (self.count-1)*self.p_mean)/self.count

def experiments(m1,m2,m3):
    bandits = [banditArm(m1), banditArm(m2), banditArm(m3)]
    rewards = np.zeros(NUM_TRIALS)
    optimal_bandit = np.argmax([b.p for b in bandits])
    for i in range(NUM_TRIALS):
        j = np.argmax([b.p_mean for b in bandits])
        rewards[i] = bandits[j].pull_lever()

        bandits[j].update(bandits[j].pull_lever())
    for b in bandits:
        print(f"Mean is {b.p_mean}")

    print(f"Total rewards is: {rewards.sum()}")
    print(f"Overall win rate is : {rewards.sum()/NUM_TRIALS}")
    print(f"Number of time each bandit is selected : {[b.count for b in bandits]}")
    cum_sum = np.cumsum(rewards)
    win_rate = cum_sum/(np.arange(NUM_TRIALS)+1)

    plt.plot(win_rate, label ='win rate')
    plt.plot(np.ones(NUM_TRIALS) * np.max([m1,m2,m3]), label='optimal')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    m1, m2, m3 = 0.2,0.5,0.7
    experiments(m1, m2, m3)
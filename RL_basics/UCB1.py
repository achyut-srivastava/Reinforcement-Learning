import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS =10000
BANDIT_PROBS = [0.2,0.8,0.6]


class Bandit:
    def __init__(self, p):
        self.p = p
        self.p_mean = 0
        self.p_count =0

    def pullLever(self):
        return np.random.random() < self.p

    def update(self, x):
        self.p_count += 1
        self.p_mean = (x + (self.p_count - 1)* self.p_mean)/self.p_count

def experience():
    rewards = np.zeros(NUM_TRIALS)
    bandits = [Bandit(p) for p in BANDIT_PROBS]

    for i in range(3):
        rewards[i] = bandits[i].pullLever()
        bandits[i].update(bandits[i].pullLever())

    for b in bandits:
        print(f" Bandit mean is: {b.p_mean}")
    for i in range(3, NUM_TRIALS):
        j = np.argmax([(b.p_mean+ np.sqrt((2*np.log(i))/b.p_count))for b in bandits])

        rewards[i] = bandits[j].pullLever()
        bandits[j].update(bandits[j].pullLever())

    for b in bandits:
        print(f" Bandit final mean is: {b.p_mean}")

    cum_sum = np.cumsum(rewards)
    win_rate = rewards.sum()/NUM_TRIALS
    print(f" Total rewards is: {rewards.sum()}")
    print(f" Number of time each bandit is selected: {[b.p_count for b in bandits]}")
    print(f" Win rate: {win_rate}")

if __name__ == "__main__":
    experience()
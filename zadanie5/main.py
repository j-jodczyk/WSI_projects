# Zaimplementuj algorytm Q-learning. Następnie, wykorzystując proste środowisko (np. Taxi-v3),
# zbadaj wpływ hiperparametrów na działanie algorytmu (np. wpływ strategii eksploracji, współczynnik uczenia).
# agent i srodowisko
# dyskonto i learning rate + wpływ strategii eksploracji (Bolzmanowska i zachlanna)
# czas dochodzenia do nagrody - max 4 str

from re import L
import numpy as np
import gym

class Agent():
    def __init__(self, actions: int, states: int, learning_rate: float, discount_rate: float) -> None:
        self.Q_table = np.zeros((states, actions)) #holds maximum expected future reward for each action at each state
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        # Q_table[action][state] = E(G|action, state)

        # Q learning algorith
        #   choose action
        #   perform action
        #   measure revard
        #   update Q

        # exploration_rate - rate of steps we'll do randomly, start with = 1
        # we generate random number, if number > exploration_rate -> exploatation (what we already know)
        #   else - explore

    def update_table(self, state, action, reward, next_step):
        # update:
        # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s’,a’) — Q(s,a)]
        self.Q_table[state, action] = self.Q_table[state, action] + self.learning_rate *\
                                    (reward + self.discount_rate * np.max(self.Q_table[next_step, :] \
                                    - self.Q_table[state, action]))

    def choose_action(self) -> None:
        pass

def main():
    env = gym.make('Taxi-v3')
    env.render()

    actions = env.action_space.n
    states = env.observation_space.n

    #params
    train_episodes = 50000
    test_episodes = 100
    steps = 100
    learning_rate = 0.7
    discount_rate = 0.6             # the larger the dr, the more agent cares about long term reward

    exploration_rate = 1            # 1 is max
    min_er = 0.01                   # min exploration rate
    exploration_decay_rate = 0.01

    agent = Agent(actions, states, learning_rate, discount_rate)

    # cumulative expected reward : G = sum(gamma**k * reward_in_state[k])
    # learning methods
    # monte carlo - collect revards at the end of episode and calulate maximum expected future reward
    #   max(expected future reward at state) = former estimation + learning_rate*(G - former estimation)
    # temporal difference - esimate reward at each step
    #   max(expected future reward at state) = former estimation + learinig_rate*(previous reward + gamma*discounted value on the next step - previous estimate)

    # approaches to RL
    # value based - goal to optimize value function
    # policy based - optimize the policy function without using value function
    #   deterministic policy
    #   stochastic policy - output a distribution probability over actions
    # model based - model the environment

    for episode in range(train_episodes):
        state = env.reset()
        done = False

        for step in range(steps):
            # decision if we explore or exploit
            exploration = np.random.random(1)[0]
            if exploration > exploration_rate:
                action = np.argmax(agent.Q_table[state, :])
            else:
                action = env.action_space.sample()

            # execute action and see what happens
            next_step, reward, done, info = env.step(action)

            # update Q_table
            agent.update_table(state, action, reward, next_step)

            state = next_step

            if done:
                break

        # decay exploration_rate
        exploration_rate = min_er + (1 - min_er) * np.exp(-exploration_decay_rate*episode)

    all_rewards = []

    for episode in range(test_episodes):
        state = env.reset()
        done = False
        rewards = 0

        for step in range(steps):
            action = np.argmax(agent.Q_table[state, :])
            next_step, reward, done, info = env.step(action)

            rewards += reward

            if done:
                all_rewards.append(rewards)
                break

            state = next_step
    env.close()
    print(f"score: {sum(all_rewards)/test_episodes}")

if __name__=="__main__":
    main()
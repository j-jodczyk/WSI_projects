# Zaimplementuj algorytm Q-learning. Następnie, wykorzystując proste środowisko (np. Taxi-v3),
# zbadaj wpływ hiperparametrów na działanie algorytmu (np. wpływ strategii eksploracji, współczynnik uczenia).
# agent i srodowisko
# dyskonto i learning rate + wpływ strategii eksploracji (Bolzmanowska i zachlanna)
# czas dochodzenia do nagrody - max 4 str


from matplotlib import pyplot as plt
import numpy as np
import gym

class Agent():
    def __init__(self, actions: int, states: int, learning_rate: float, discount_rate: float, policy: str) -> None:
        self.Q_table = np.zeros((states, actions)) #holds maximum expected future reward for each action at each state
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.policy = policy

    def choose_action(self, state, param, env):
        if self.policy == "greedy":
            exploration = np.random.random(1)[0]
            if exploration > param: # large param -> explore
                action = np.argmax(self.Q_table[state, :]) # exploit
            else:
                action = env.action_space.sample() # explore
        else:
            exp_sum = np.sum([np.exp(self.Q_table[state, a]/param) for a in range(len(self.Q_table[state]))])
            action_probs = [(np.exp(self.Q_table[state, a]/param))/exp_sum for a in range(len(self.Q_table[state]))]
            action = np.random.choice(list(range(len(self.Q_table[state, :]))), p=action_probs)
        return action


    def update_table(self, state, action, reward, next_step):
        # update:
        # Q(s,a):= Q(s,a) + lr [R(s,a) + dr * max Q(s’,a’) — Q(s,a)]
        self.Q_table[state, action] = self.Q_table[state, action] + self.learning_rate *\
                                    (reward + self.discount_rate * np.max(self.Q_table[next_step, :] \
                                    - self.Q_table[state, action]))

def main():
    env = gym.make('Taxi-v3')

    actions = env.action_space.n
    states = env.observation_space.n

    #params
    train_episodes = 10000
    test_episodes = 100
    steps = 100
    learning_rate = 1
    discount_rate = 0.9            # the larger the dr, the more agent cares about long term reward
    exploration_rate = 0.5
    temperature = 30
    policy = "boltzmann"           # boltzmann or greedy


    avg_steps = 0
    avg_reward = 0
    for i in range(20):

        agent = Agent(actions, states, learning_rate, discount_rate, policy)

        for episode in range(train_episodes):
            state = env.reset()
            done = False

            for step in range(steps):
                # decision if we explore or exploit
                param = exploration_rate if agent.policy == "greedy" else temperature
                action = agent.choose_action(state, param, env)
                # execute action and see what happens
                next_step, reward, done, info = env.step(action)

                # update Q_table
                agent.update_table(state, action, reward, next_step)

                state = next_step

                if done:
                    break

        # tests:
        rewards = []
        for episode in range(test_episodes):
            state = env.reset()
            done = False

            steps_eval = 0
            reward_eval = 0

            for step in range(steps):
                action = np.argmax(agent.Q_table[state, :])
                next_state, reward, done, info = env.step(action)

                reward_eval += reward

                if done:
                    agent.Q_table[next_state, :] = 0
                    rewards.append(reward_eval)
                    steps_eval += step
                    break

                state = next_state

        env.close()
        avg_steps += steps_eval
        avg_reward += sum(rewards)/test_episodes

    print(f"Avg steps: {avg_steps/20}")
    print(f"Avg reward: {avg_reward/20}")


if __name__=="__main__":
    main()
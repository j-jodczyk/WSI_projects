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
    discount_rate = 0.1            # the larger the dr, the more agent cares about long term reward
    exploration_rate = 0.1
    temperature = 0.1
    policy = "boltzmann"           # boltzmann or greedy


    avg_steps = 0
    avg_reward = 0
    for i in range(5):

        agent = Agent(actions, states, learning_rate, discount_rate, policy)

        rewards = []
        steps_eval = 0
        for episode in range(train_episodes):
            state = env.reset()
            done = False

            reward_eval = 0

            for step in range(steps):

                # decision if we explore or exploit
                param = exploration_rate if agent.policy == "greedy" else temperature
                action = agent.choose_action(state, param, env)
                # execute action and see what happens
                next_state, reward, done, info = env.step(action)

                reward_eval += reward

                # update Q_table
                agent.update_table(state, action, reward, next_state)

                state = next_state

                if done:
                    agent.Q_table[next_state, :] = 0
                    rewards.append(reward_eval)
                    steps_eval += step
                    break
            if step == 99:
                steps_eval += step

        env.close()
        avg_steps += steps_eval/train_episodes
        avg_reward += sum(rewards)

    print(f"Avg steps: {avg_steps/5}")
    print(f"Avg reward: {avg_reward/5}")


if __name__=="__main__":
    main()
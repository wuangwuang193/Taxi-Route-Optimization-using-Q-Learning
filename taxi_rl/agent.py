
import numpy as np





class QLearningAgent:

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.1,
        discount_factor=0.99
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.alpha = learning_rate
        self.gamma = discount_factor

        self.q_table = np.zeros(
            (state_size, action_size)
        )

    def choose_action(self, state, epsilon):

        if np.random.random() < epsilon:

            action = np.random.randint(
                self.action_size
            )

        else:

            action = np.argmax(
                self.q_table[state]
            )

        return action

    def learn(
        self,
        state,
        action,
        reward,
        next_state,
        terminated
    ):

        current_q = self.q_table[state, action]

        if terminated:
            next_max_q = 0

        else:
            next_max_q = np.max(
                self.q_table[next_state]
            )

        new_q = current_q + self.alpha * (
            reward
            + self.gamma * next_max_q
            - current_q
        )

        self.q_table[state, action] = new_q

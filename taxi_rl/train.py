
import os
import numpy as np
import pandas as pd

from taxi_rl.environment import TaxiEnvironment
from taxi_rl.agent import QLearningAgent


def train_agent(
    episodes=10000,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995
):

    # 1. Khởi tạo Environment
    env = TaxiEnvironment()

    # 2. Lấy số lượng State và Action
    state_size = env.env.observation_space.n
    action_size = env.env.action_space.n

    # 3. Khởi tạo Agent
    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        discount_factor=discount_factor
    )

    # 4. Lưu kết quả huấn luyện
    rewards_history = []
    steps_history = []

    print("===== START TRAINING =====")

    # 5. Vòng lặp Training
    for episode in range(episodes):

        # Khởi tạo episode mới
        state, info = env.reset()

        total_reward = 0
        steps = 0

        terminated = False
        truncated = False

        # 6. Agent tương tác với Environment
        while not (terminated or truncated):

            # Agent chọn hành động
            action = agent.choose_action(
                state,
                epsilon
            )

            # Thực hiện hành động
            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            # Agent học từ kết quả
            agent.learn(
                state,
                action,
                reward,
                next_state,
                terminated
            )

            # Cập nhật trạng thái
            state = next_state

            total_reward += reward
            steps += 1

        # 7. Lưu kết quả episode
        rewards_history.append(total_reward)
        steps_history.append(steps)

        # 8. Giảm epsilon
        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

        # 9. In tiến trình mỗi 1000 episodes
        if (episode + 1) % 1000 == 0:

            avg_reward = np.mean(
                rewards_history[-1000:]
            )

            avg_steps = np.mean(
                steps_history[-1000:]
            )

            print(
                f"Episode: {episode + 1}/{episodes} | "
                f"Average Reward: {avg_reward:.2f} | "
                f"Average Steps: {avg_steps:.2f} | "
                f"Epsilon: {epsilon:.4f}"
            )

    # 10. Lưu Q-table
    os.makedirs("outputs/models", exist_ok=True)

    np.save(
        "outputs/models/q_table.npy",
        agent.q_table
    )

    # 11. Lưu lịch sử Training
    os.makedirs("outputs/logs", exist_ok=True)

    history = pd.DataFrame({
        "episode": range(1, episodes + 1),
        "reward": rewards_history,
        "steps": steps_history
    })

    history.to_csv(
        "outputs/logs/training_log.csv",
        index=False
    )

    env.close()

    print("\n===== TRAINING COMPLETED =====")
    print("Q-table saved successfully!")
    print("Training log saved successfully!")

    return agent, history

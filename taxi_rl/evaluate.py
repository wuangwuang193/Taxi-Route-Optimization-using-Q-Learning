
import os
import numpy as np

from taxi_rl.environment import TaxiEnvironment
from taxi_rl.agent import QLearningAgent


def evaluate_agent(episodes=1000):

    # 1. Khởi tạo Environment
    env = TaxiEnvironment()

    # 2. Lấy số lượng State và Action
    state_size = env.env.observation_space.n
    action_size = env.env.action_space.n

    # 3. Khởi tạo Agent
    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size
    )

    # 4. Kiểm tra file Q-table
    model_path = "outputs/models/q_table.npy"

    if not os.path.exists(model_path):
        env.close()
        raise FileNotFoundError(
            "Khong tim thay Q-table. "
            "Hay chay Training truoc!"
        )

    # 5. Load kiến thức đã học
    agent.q_table = np.load(model_path)

    if agent.q_table.shape != (state_size, action_size):
        env.close()
        raise ValueError("Kich thuoc Q-table khong hop le!")

    print("===== START EVALUATION =====")
    print("Loaded Q-table:", agent.q_table.shape)

    # 6. Các biến lưu kết quả
    rewards_history = []
    steps_history = []

    success_count = 0

    try:

        # 7. Chạy Evaluation
        for episode in range(episodes):

            state, info = env.reset()

            total_reward = 0
            steps = 0

            terminated = False
            truncated = False

            while not (terminated or truncated):

                # Chọn hành động tốt nhất đã học
                action = agent.choose_action(
                    state,
                    epsilon=0.0
                )

                # Thực hiện hành động
                next_state, reward, terminated, truncated, info = (
                    env.step(action)
                )

                # Cập nhật trạng thái
                state = next_state

                total_reward += reward
                steps += 1

            # Lưu kết quả
            rewards_history.append(total_reward)
            steps_history.append(steps)

            # Taxi-v3 kết thúc thành công khi terminated=True
            if terminated:
                success_count += 1

        # 8. Tính các chỉ số
        success_rate = (
            success_count / episodes
        ) * 100

        average_reward = np.mean(rewards_history)
        average_steps = np.mean(steps_history)

        # 9. In kết quả
        print("\n===== EVALUATION RESULTS =====")

        print(f"Total Episodes: {episodes}")

        print(
            f"Successful Episodes: "
            f"{success_count}/{episodes}"
        )

        print(f"Success Rate: {success_rate:.2f}%")

        print(f"Average Reward: {average_reward:.2f}")

        print(f"Average Steps: {average_steps:.2f}")

        return {
            "success_rate": success_rate,
            "average_reward": average_reward,
            "average_steps": average_steps,
            "success_count": success_count,
            "total_episodes": episodes
        }

    finally:
        env.close()

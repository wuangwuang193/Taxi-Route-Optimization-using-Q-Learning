
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from taxi_rl.environment import TaxiEnvironment
from taxi_rl.agent import QLearningAgent


# Tên các hành động trong Taxi-v3
ACTION_NAMES = {
    0: "DOWN",
    1: "UP",
    2: "RIGHT",
    3: "LEFT",
    4: "PICKUP",
    5: "DROPOFF"
}

# Tên các địa điểm
LOCATION_NAMES = ["R", "G", "Y", "B"]


def get_state_description(env, state):

    # Giải mã State
    info = env.decode_state(state)

    taxi_row = info["taxi_row"]
    taxi_col = info["taxi_col"]
    passenger = info["passenger"]
    destination = info["destination"]

    # Xác định vị trí hành khách
    if passenger == 4:
        passenger_location = "IN TAXI"
    else:
        passenger_location = LOCATION_NAMES[passenger]

    return (
        f"Taxi: ({taxi_row}, {taxi_col}) | "
        f"Passenger: {passenger_location} | "
        f"Destination: {LOCATION_NAMES[destination]}"
    )


def run_visual_simulation(delay=0.5):

    # =====================================
    # 1. KHỞI TẠO ENVIRONMENT
    # =====================================

    env = TaxiEnvironment(render_mode="rgb_array")

    state_size = env.env.observation_space.n
    action_size = env.env.action_space.n

    # =====================================
    # 2. KHỞI TẠO AGENT
    # =====================================

    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size
    )

    # =====================================
    # 3. LOAD Q-TABLE
    # =====================================

    project_dir = Path(__file__).resolve().parent.parent

    model_path = (
        project_dir / "outputs" / "models" / "q_table.npy"
    )

    if not model_path.exists():
        env.close()
        raise FileNotFoundError(
            "Khong tim thay q_table.npy. "
            "Hay chay Training truoc!"
        )

    agent.q_table = np.load(model_path)

    if agent.q_table.shape != (state_size, action_size):
        env.close()
        raise ValueError("Kich thuoc Q-table khong hop le!")

    # =====================================
    # 4. RESET ENVIRONMENT
    # =====================================

    state, info = env.reset()

    total_reward = 0
    steps = 0

    terminated = False
    truncated = False

    # =====================================
    # 5. TẠO GIAO DIỆN MATPLOTLIB
    # =====================================

    plt.ion()

    fig, ax = plt.subplots(figsize=(8, 8))

    fig.suptitle(
        "TAXI AI - GRAPHICAL SIMULATION",
        fontsize=16,
        fontweight="bold"
    )

    # Lấy hình ảnh bản đồ ban đầu
    frame = env.render()

    image = ax.imshow(frame)

    ax.axis("off")

    ax.set_title(
        get_state_description(env, state),
        fontsize=11
    )

    # Hiển thị cửa sổ
    plt.show(block=False)

    print("===== TAXI AI GRAPHICAL SIMULATION =====")
    print("Q-table loaded:", agent.q_table.shape)

    try:

        # =====================================
        # 6. AI BẮT ĐẦU DI CHUYỂN
        # =====================================

        while not (terminated or truncated):

            # Chọn hành động tốt nhất
            action = agent.choose_action(
                state,
                epsilon=0.0
            )

            # Thực hiện hành động
            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            # Cập nhật thông tin
            steps += 1
            total_reward += reward

            state = next_state

            # Lấy hình ảnh mới
            frame = env.render()

            # Cập nhật bản đồ
            image.set_data(frame)

            # Cập nhật tiêu đề
            state_description = get_state_description(
                env,
                state
            )

            ax.set_title(
                f"{state_description}\n"
                f"Action: {ACTION_NAMES[action]} | "
                f"Reward: {reward} | "
                f"Total Reward: {total_reward} | "
                f"Steps: {steps}",
                fontsize=10
            )

            # Cập nhật giao diện
            fig.canvas.draw_idle()

            plt.pause(delay)

        # =====================================
        # 7. HIỂN THỊ KẾT QUẢ
        # =====================================

        if terminated:
            result = "SUCCESS - PASSENGER DELIVERED!"
        else:
            result = "FAILED - TIME LIMIT REACHED!"

        ax.set_title(
            f"{result}\n"
            f"Total Steps: {steps} | "
            f"Total Reward: {total_reward}",
            fontsize=12,
            fontweight="bold"
        )

        fig.canvas.draw_idle()

        print("\n===== SIMULATION RESULTS =====")
        print("Total Steps:", steps)
        print("Total Reward:", total_reward)
        print("Result:", result)

    finally:

        env.close()

        plt.ioff()

        # Giữ cửa sổ để người dùng quan sát kết quả
        plt.show()


if __name__ == "__main__":

    run_visual_simulation(delay=0.5)

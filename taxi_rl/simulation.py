
from pathlib import Path
import time
import numpy as np

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

# Tên các địa điểm trên bản đồ
LOCATION_NAMES = ["R", "G", "Y", "B"]


def describe_state(env, state):

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
        f"Taxi position: ({taxi_row}, {taxi_col})\n"
        f"Passenger: {passenger_location}\n"
        f"Destination: {LOCATION_NAMES[destination]}"
    )


def simulate_agent(delay=0.5):

    # 1. Khởi tạo Environment có chế độ hiển thị
    env = TaxiEnvironment(render_mode="ansi")

    # 2. Lấy kích thước State và Action
    state_size = env.env.observation_space.n
    action_size = env.env.action_space.n

    # 3. Khởi tạo Agent
    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size
    )

    # 4. Load Q-table
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

    q_table = np.load(model_path)

    if q_table.shape != (state_size, action_size):
        env.close()
        raise ValueError("Kich thuoc Q-table khong hop le!")

    agent.q_table = q_table

    print("=" * 50)
    print("TAXI AI LIVE SIMULATION")
    print("=" * 50)

    print("Loaded Q-table:", agent.q_table.shape)

    # 5. Reset Environment
    state, info = env.reset()

    total_reward = 0
    steps = 0

    terminated = False
    truncated = False

    try:

        print("\nINITIAL STATE")
        print(describe_state(env, state))
        print("\nINITIAL MAP")
        print(env.render())

        # 6. AI bắt đầu di chuyển
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

            steps += 1
            total_reward += reward

            print("\n" + "=" * 50)
            print(f"STEP: {steps}")
            print("ACTION:", ACTION_NAMES[action])
            print("REWARD:", reward)

            # Hiển thị trạng thái mới
            print("\nCURRENT STATE")
            print(describe_state(env, next_state))

            # Hiển thị bản đồ mới
            print("\nCURRENT MAP")
            print(env.render())

            state = next_state

            time.sleep(delay)

        # 7. Kết quả Simulation
        print("\n" + "=" * 50)
        print("SIMULATION COMPLETED")
        print("=" * 50)

        print("Total Steps:", steps)
        print("Total Reward:", total_reward)

        if terminated:
            print("RESULT: SUCCESS - Passenger delivered!")

        elif truncated:
            print("RESULT: FAILED - Time limit reached!")

    finally:
        env.close()


if __name__ == "__main__":

    simulate_agent(delay=0.5)


from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def visualize_training():

    # 1. Xác định đường dẫn dữ liệu
    project_dir = Path(__file__).resolve().parent.parent

    data_path = project_dir / "outputs" / "logs" / "training_log.csv"

    output_dir = project_dir / "outputs" / "figures"

    # 2. Kiểm tra dữ liệu
    if not data_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file: {data_path}"
        )

    # 3. Tạo thư mục lưu biểu đồ
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Đọc dữ liệu CSV
    df = pd.read_csv(data_path)

    required_columns = {"episode", "reward", "steps"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "File CSV thieu cac cot episode, reward hoac steps."
        )

    if df.empty:
        raise ValueError("File CSV khong co du lieu.")

    print("===== TRAINING DATA =====")
    print(df.head())

    print("\nNumber of Episodes:", len(df))

    # 5. Tính Moving Average Reward
    window_size = 100

    df["moving_average"] = (
        df["reward"]
        .rolling(window=window_size, min_periods=1)
        .mean()
    )

    # =====================================
    # CHART 1: REWARD PER EPISODE
    # =====================================

    plt.figure(figsize=(12, 5))

    plt.plot(
        df["episode"],
        df["reward"],
        label="Reward per Episode"
    )

    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "reward_per_episode.png",
        dpi=300
    )

    # =====================================
    # CHART 2: MOVING AVERAGE REWARD
    # =====================================

    plt.figure(figsize=(12, 5))

    plt.plot(
        df["episode"],
        df["moving_average"],
        label=f"Moving Average ({window_size} Episodes)"
    )

    plt.title("Moving Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "moving_average_reward.png",
        dpi=300
    )

    # =====================================
    # CHART 3: STEPS PER EPISODE
    # =====================================

    plt.figure(figsize=(12, 5))

    plt.plot(
        df["episode"],
        df["steps"],
        label="Steps per Episode"
    )

    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Number of Steps")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "steps_per_episode.png",
        dpi=300
    )

    # 6. Hiển thị tất cả biểu đồ
    plt.show()

    # 7. Thông báo hoàn thành
    print("\n===== VISUALIZATION COMPLETED =====")
    print("Charts saved in:", output_dir)


if __name__ == "__main__":
    visualize_training()

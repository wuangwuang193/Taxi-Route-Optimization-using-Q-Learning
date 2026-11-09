
from taxi_rl.evaluate import evaluate_agent


def main():

    results = evaluate_agent(
        episodes=1000
    )

    print("\n===== FINAL SUMMARY =====")

    print(
        "Success Rate:",
        f"{results['success_rate']:.2f}%"
    )

    print(
        "Average Reward:",
        f"{results['average_reward']:.2f}"
    )

    print(
        "Average Steps:",
        f"{results['average_steps']:.2f}"
    )


if __name__ == "__main__":
    main()

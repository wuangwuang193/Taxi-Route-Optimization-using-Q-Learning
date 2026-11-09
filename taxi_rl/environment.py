
import gymnasium as gym


class TaxiEnvironment:

    def __init__(self, render_mode=None):
        """
        Khởi tạo môi trường Taxi-v3.
        """

        self.env = gym.make(
            "Taxi-v3",
            render_mode=render_mode
        )

        self.state = None

        # Các địa điểm trên bản đồ
        self.locations = ["R", "G", "Y", "B"]

    def reset(self):
        """
        Khởi tạo một episode mới.
        """

        self.state, info = self.env.reset()

        return self.state, info

    def step(self, action):
        """
        Thực hiện hành động của taxi.
        """

        next_state, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        self.state = next_state

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info
        )

    def render(self):
        """
        Hiển thị bản đồ Taxi.
        """

        return self.env.render()

    def decode_state(self, state):
        """
        Giải mã trạng thái thành thông tin dễ hiểu.
        """

        taxi_row, taxi_col, passenger, destination = (
            self.env.unwrapped.decode(state)
        )

        return {
            "taxi_row": taxi_row,
            "taxi_col": taxi_col,
            "passenger": passenger,
            "destination": destination
        }

    def close(self):
        """
        Đóng môi trường.
        """

        self.env.close()

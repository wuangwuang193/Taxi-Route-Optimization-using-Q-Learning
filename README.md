# 🚕 Taxi Route Optimization using Q-Learning

## 1. Giới thiệu dự án

**Taxi Route Optimization** là dự án mô phỏng bài toán điều hướng taxi và vận chuyển hành khách bằng phương pháp Học tăng cường (Reinforcement Learning), sử dụng thuật toán **Q-Learning**.

Dự án được xây dựng dựa trên môi trường **Taxi-v3** của thư viện Gymnasium. Trong môi trường này, một chiếc taxi di chuyển trên bản đồ dạng lưới, tìm hành khách, đón khách và đưa hành khách đến đúng điểm đến.

Thông qua quá trình huấn luyện, tác tử (Agent) học cách lựa chọn hành động phù hợp nhằm tối đa hóa tổng phần thưởng và hoàn thành nhiệm vụ với số bước di chuyển ít nhất có thể.

## 2. Mục tiêu dự án

- Tìm hiểu nguyên lý hoạt động của Học tăng cường (Reinforcement Learning).
- Tìm hiểu và triển khai thuật toán Q-Learning.
- Xây dựng tác tử có khả năng học cách điều hướng trong môi trường Taxi-v3.
- Tối ưu hóa quá trình đón và trả hành khách.
- Đánh giá hiệu quả của mô hình thông qua phần thưởng, số bước thực hiện và tỷ lệ hoàn thành nhiệm vụ.
- Trực quan hóa quá trình huấn luyện và mô phỏng hoạt động của tác tử.

## 3. Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python | Ngôn ngữ lập trình chính |
| Gymnasium | Xây dựng môi trường Taxi-v3 |
| NumPy | Xử lý dữ liệu và Q-table |
| Pandas | Tổng hợp và phân tích kết quả |
| Matplotlib | Trực quan hóa dữ liệu |
| Pygame | Hỗ trợ mô phỏng trực quan (nếu sử dụng trong phiên bản hiện tại) |

## 4. Môi trường Taxi-v3

Taxi-v3 là môi trường mô phỏng một chiếc taxi hoạt động trên bản đồ dạng lưới.

Tác tử cần thực hiện các nhiệm vụ:

1. Di chuyển taxi đến vị trí của hành khách.
2. Đón hành khách.
3. Di chuyển đến điểm đến được yêu cầu.
4. Trả hành khách tại đúng vị trí.

**Không gian hành động (Action Space):**

- Di chuyển xuống.
- Di chuyển lên.
- Di chuyển sang phải.
- Di chuyển sang trái.
- Đón hành khách.
- Trả hành khách.

**Không gian trạng thái (State Space):**

Môi trường Taxi-v3 có 500 trạng thái và 6 hành động có thể lựa chọn tại mỗi trạng thái.

## 5. Thuật toán Q-Learning

Q-Learning là thuật toán Học tăng cường không cần mô hình (Model-free Reinforcement Learning), cho phép tác tử học cách lựa chọn hành động thông qua quá trình tương tác với môi trường.

Dự án sử dụng Q-table để lưu trữ giá trị của từng cặp trạng thái và hành động.

Công thức cập nhật Q-value:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha
\left[r + \gamma \max_{a'}Q(s',a') - Q(s,a)\right]
\]

Trong đó:

- \(s\): Trạng thái hiện tại.
- \(a\): Hành động được lựa chọn.
- \(r\): Phần thưởng nhận được.
- \(s'\): Trạng thái tiếp theo.
- \(\alpha\): Tốc độ học (Learning Rate).
- \(\gamma\): Hệ số chiết khấu (Discount Factor).

### Chiến lược Epsilon-Greedy

Dự án sử dụng chiến lược Epsilon-Greedy để cân bằng giữa:

- **Exploration:** Thử nghiệm những hành động mới.
- **Exploitation:** Lựa chọn hành động có Q-value cao nhất đã học được.

Giá trị Epsilon giảm dần theo quá trình huấn luyện để tác tử chuyển từ khám phá sang khai thác kiến thức đã học.

## 6. Cấu hình huấn luyện

Các tham số được sử dụng trong lần huấn luyện đã ghi nhận:

| Tham số | Giá trị |
|---|---:|
| Số tập huấn luyện (Episodes) | 10.000 |
| Learning Rate (Alpha) | 0.1 |
| Discount Factor (Gamma) | 0.99 |
| Epsilon ban đầu | 1.0 |
| Epsilon tối thiểu | 0.01 |
| Epsilon Decay | 0.995 |
| Số trạng thái | 500 |
| Số hành động | 6 |

## 7. Kết quả thực nghiệm

Sau khi huấn luyện, mô hình được đánh giá trên 1.000 lượt mô phỏng.

| Chỉ số đánh giá | Kết quả |
|---|---:|
| Tổng số lượt đánh giá | 1.000 |
| Số lượt hoàn thành nhiệm vụ | 1.000 |
| Tỷ lệ thành công | 100% |
| Phần thưởng trung bình | 7.90 |
| Số bước trung bình | 13.10 |
| Kích thước Q-table | 500 × 6 |

Kết quả cho thấy tác tử đã học được chiến lược điều hướng phù hợp trong môi trường Taxi-v3 và hoàn thành nhiệm vụ vận chuyển hành khách trong các lượt đánh giá đã thực hiện.

*Lưu ý: Đây là kết quả từ lần chạy thực nghiệm đã ghi nhận; kết quả có thể thay đổi khi huấn luyện lại với cấu hình hoặc điều kiện khác.*

## 8. Cấu trúc dự án

```text
TaxiRouteOptimization/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── taxi_rl/
│   ├── agent.py
│   ├── environment.py
│   ├── train.py
│   ├── evaluate.py
│   ├── simulation.py
│   ├── visual_simulation.py
│   ├── visualization.py
│   └── animation.py
│
├── outputs/
│   └── figures/
│
└── tests/
```

## 9. Hướng dẫn cài đặt và sử dụng

### Bước 1: Clone repository

```bash
git clone <LINK_GITHUB_CUA_BAN>
```

### Bước 2: Di chuyển vào thư mục dự án

```bash
cd TaxiRouteOptimization
```

### Bước 3: Tạo môi trường ảo

```bash
python -m venv venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
venv\Scripts\activate
```

### Bước 4: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 5: Chạy chương trình

Tùy theo chức năng được triển khai trong phiên bản hiện tại:

```bash
python main.py
```

## 10. Hướng phát triển

Trong tương lai, dự án có thể được mở rộng theo các hướng:

- So sánh Q-Learning với SARSA và các thuật toán Học tăng cường khác.
- Thử nghiệm các cấu hình Learning Rate, Discount Factor và Epsilon khác nhau.
- Cải thiện giao diện mô phỏng hoạt động của taxi.
- Bổ sung biểu đồ theo dõi quá trình huấn luyện và đánh giá mô hình.
- Nghiên cứu áp dụng các thuật toán Deep Q-Network (DQN).
- Mở rộng sang các bài toán điều hướng phức tạp hơn.

## 11. Giới hạn của dự án

Dự án hiện tại sử dụng môi trường Taxi-v3 có bản đồ dạng lưới và quy tắc di chuyển cố định. Vì vậy, đây là mô hình mô phỏng bài toán điều hướng và vận chuyển hành khách, chưa phải hệ thống tối ưu tuyến đường taxi ngoài thực tế.

Dự án chưa xét đến các yếu tố như tình trạng giao thông, khoảng cách địa lý thực tế, thời gian di chuyển hoặc nhiều xe taxi hoạt động đồng thời.

## 12. Tác giả

**Tên dự án:** Taxi Route Optimization using Q-Learning  
**Lĩnh vực:** Data Science / Reinforcement Learning  
**Ngôn ngữ:** Python

---

⭐ Nếu thấy dự án hữu ích, hãy để lại một Star cho repository!
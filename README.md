markdown
# Đồ Án Robot Ứng Dụng Học Máy: Nhận Diện Màu Giấy & Mô Phỏng Điều Khiển

Đồ án thực hiện bài toán nhận diện màu sắc của giấy trong điều kiện ánh sáng thay đổi bằng thuật toán Học máy (Machine Learning) và mô phỏng thuật toán điều khiển hành vi của Robot di chuyển theo quỹ đạo hình tam giác dựa trên kết quả nhận diện thời gian thực qua Webcam.

##  Thành Viên Thực Hiện
* **Họ và tên:** [Lê Thị Hoàng Yến]
* **Mã số sinh viên (MSSV):** [2421060224]
* **Đề tài phát triển:** Nhận dạng màu sắc trong điều kiện ánh sáng thay đổi và mô phỏng điều khiển (Phát triển từ đề tài mã A4).

##  Cấu Trúc Thư Mục Project Trên GitHub
* `train_py`: Mã nguồn sử dụng để huấn luyện mô hình Machine Learning từ tập dữ liệu.
* `cam_py`: Mã nguồn kiểm tra (test) độc lập tính năng nhận diện màu sắc trực tiếp qua Webcam.
* `mô phỏng_py`: Mã nguồn chính của đồ án, tích hợp luồng nhận diện từ Webcam và thuật toán điều khiển robot chấm tròn di chuyển mượt mà về các đỉnh của hình tam giác.
* `color_model.pkl`: File lưu trữ trọng số mô hình đã huấn luyện thành công (bộ não AI).

##  Thư Viện Sử Dụng (Requirements)
Project sử dụng các thư viện Python chuẩn theo khuyến khích của giảng viên:
Project sử dụng các thư viện Python tiêu chuẩn sau để xây dựng hệ thống:
* `opencv-python` (`cv2`) - Xử lý hình ảnh webcam và vẽ đồ họa mô phỏng.
* `numpy` - Xử lý mảng dữ liệu ảnh và tính toán toán học.
* `joblib` - Nạp và lưu trữ file mô hình Học máy (`.pkl`).
* `os` - Thư viện hệ điều hành mặc định của Python để quản lý đường dẫn file.
  
##  Hướng Dẫn Cài Đặt Và Chạy Chương Trình

### Bước 1: Cài đặt các thư viện cần thiết
Mở Terminal / Command Prompt tại thư mục dự án trên máy tính và chạy lệnh sau để cài đặt:
```bash
pip install opencv-python numpy scikit-learn
```

### Bước 2: Khởi chạy file mô phỏng chính
Đảm bảo máy tính của bạn đã kết nối với Webcam ổn định, sau đó chạy lệnh:
```bash
python mô phỏng_.py
```

## 📺 Video Demo Thực Tế (1 - 2 Phút)
* Đường dẫn xem video kết quả vận hành: [https://www.youtube.com/watch?v=0T1cx0_3jLI]
* Đường dẫn xem video mô phỏng :[https://youtu.be/Vz7noqT1RoQ]

# Tài Liệu Kỹ Thuật - DetectCaptcha

Dự án này là một hệ thống nhận diện CAPTCHA dạng SVG/PNG đơn giản sử dụng thuật toán so khớp hình dáng (Shape Matching) với thuật toán K-Nearest Neighbors (1-NN) dựa trên khoảng cách Chamfer.

## 1. Cấu Trúc Dự Án

```
detect_captcha/
├── main.py         # API Gateway (FastAPI) - Điểm tiếp nhận request
├── solver.py       # Core Logic - Bộ não xử lý nhận diện
├── utils.py        # Utilities - Các hàm xử lý ảnh và vector
├── train.py        # Training Script - Công cụ học từ dữ liệu nhãn
├── database.json   # Knowledge Base - Dữ liệu đã học (Model)
└── requirements.txt # Các thư viện phụ thuộc
```

## 2. Chi Tiết Các Module

### 2.1. `utils.py` - Bộ xử lý đầu vào (Preprocessing)
Module này chịu trách nhiệm biến đổi dữ liệu thô (ảnh hoặc mã SVG) thành dạng dữ liệu vector chuẩn hóa mà thuật toán có thể hiểu được.

*   **`process_png_content(image_content: bytes) -> list[dict]`**:
    *   **Input**: Byte thô của file ảnh PNG.
    *   **Logic**:
        1.  Dùng OpenCV (`cv2`) decode ảnh.
        2.  **Invert Colors**: Chuyển ảnh chữ đen/nền trắng thành chữ trắng/nền đen để phù hợp với thuật toán tìm biên (contour) của OpenCV.
        3.  **Denoise**: Dùng phép toán hình thái học (Morphological Open) để loại bỏ các điểm nhiễu nhỏ, vệt bẩn.
        4.  **Find Contours**: Tìm bao đóng (contours) của các ký tự.
        5.  **Filter**: Loại bỏ các contour quá nhỏ (nhiễu).
        6.  **Sắp xếp**: Sắp xếp các contour theo trục X (từ trái sang phải) để đảm bảo thứ tự ký tự.
    *   **Output**: Danh sách các tập điểm (point cloud) đại diện cho từng ký tự.

*   **`parse_svg_paths(svg_content: str) -> list[dict]`**:
    *   **Input**: Chuỗi XML của file SVG.
    *   **Logic**:
        1.  Dùng Regex trích xuất tất cả thẻ `<path>`.
        2.  **Lọc nhiễu**: Bỏ qua các path có `fill="none"` hoặc có thuộc tính `stroke` (thường là đường gạch ngang gây nhiễu trong CAPTCHA).
        3.  Parse thuộc tính `d="..."` để lấy danh sách tọa độ.
    *   **Output**: Danh sách các tập điểm của từng ký tự, đã sắp xếp theo trục X.

*   **`normalize_points(points, size=100) -> list[tuple]`**:
    *   **Mục đích**: Chống biến dạng do tỉ lệ (Scale Invariant).
    *   **Logic**: Co dãn (scale) và dịch chuyển (translate) tập điểm của một ký tự sao cho nó nằm vừa vặn vào trung tâm của một khung hình vuông `100x100`. Điều này giúp chữ 'A' nhỏ và chữ 'A' lớn có biểu diễn giống nhau.

### 2.2. `solver.py` - Logic Nhận Diện (Recognition Engine)
Đây là nơi chứa class `CaptchaSolver`, thực hiện việc so khớp mẫu.

*   **`calculate_distance(pts1, pts2) -> float`**:
    *   **Thuật toán**: Chamfer Distance (biến thể hai chiều đối xứng).
    *   **Mô tả**: Với mỗi điểm trên hình 1, tìm điểm gần nhất trên hình 2 và tính bình phương khoảng cách. Làm tương tự ngược lại từ hình 2 sang hình 1.
    *   **Ý nghĩa**: Đo độ "sai khác" về hình dáng giữa 2 ký tự. Giá trị càng nhỏ, 2 ký tự càng giống nhau.

*   **`solve(input_data) -> str`**:
    *   **Luồng xử lý**:
        1.  Nhận đầu vào (SVG hoặc List Vectors từ PNG).
        2.  Với mỗi ký tự (contour) tách được:
            *   Chuẩn hóa (Normalize) về size 100x100.
            *   Duyệt qua toàn bộ **Knowledge Base** (`database.json`).
            *   Tính khoảng cách tới từng mẫu trong DB.
            *   Chọn nhãn (Label) của mẫu có khoảng cách bé nhất (Nearest Neighbor).
        3.  Ghép các nhãn lại thành chuỗi kết quả.

### 2.3. `train.py` - Quy trình Huấn Luyện (Training)
*   **`rebuild_database(labeled_dir)`**:
    *   **Input**: Folder chứa các file SVG đã đặt tên theo đúng nội dung (VD: `AB12CD.svg`).
    *   **Logic**:
        1.  Xóa sạch database cũ.
        2.  Đọc từng file SVG, tách các ký tự.
        3.  Gán nhãn cho từng ký tự dựa vào vị trí của nó trong tên file.
        4.  Lưu vector đặc trưng vào `database.json`.
    *   **Lợi ích**: Cho phép cập nhật model cực nhanh chỉ bằng cách thêm file mẫu vào folder và chạy lại script.

### 2.4. `main.py` - Giao Tiếp API
*   Cung cấp endpoint `POST /solve` hỗ trợ linh hoạt các kiểu input:
    *   **File Upload**: Upload file ảnh/SVG trực tiếp.
    *   **URL**: Gửi link ảnh, server tự tải về và xử lý.
    *   **Raw Body**: Gửi nội dung SVG text hoặc Binary trong body request.
*   Tự động phát hiện kiểu dữ liệu (`application/json`, `image/png`, text) để gọi hàm xử lý phù hợp.

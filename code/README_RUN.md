# Chạy lại bài lab

Từ thư mục gốc repo, dùng môi trường Conda base:

```bash
source /home/sakana/miniconda3/etc/profile.d/conda.sh
conda activate base
python submission_2A202602810/code/run_notebook.py
```

Hoặc mở `lab.ipynb` từ thư mục `submission_2A202602810/code/`, chọn kernel Python của Conda base và Restart & Run All.
Notebook nhận GPU nếu CUDA sẵn sàng, có fallback CPU. Tất cả các thí nghiệm dùng 20 epoch.
Trên Colab/Kaggle, cần đặt thư mục nộp cùng repo chứa `data/`, `scripts/`; mẫu Excel đã có trong thư mục code,
đổi REPO_ROOT trong ô đầu nếu lưu dữ liệu ở chỗ khác. Cài thiếu thư viện bằng requirements.txt.

- Dữ liệu chia theo metadata gốc, không thay đổi train/eval.
- Chuẩn hoá và chọn cấu hình chỉ dùng train/validation.
- Checkpoint ở epoch có val loss thấp nhất, xếp cấu hình bằng macro-F1 val.
- FP16 là phép đo precision bổ sung, không tham gia chọn cấu hình cuối đã khoá.
- Dự đoán cuối là seed 1; seed 2/3 baseline chỉ dùng đo nhiễu.
- Dự đoán của lần chạy lại giống hệt thì dùng lại kết quả chấm chính thức; không cần chấm lặp.
- Chạy `python code/verify_submission.py` để kiểm tra cấu trúc, số liệu và output notebook.

Module train không đọc tensor eval. Tập eval chỉ được đưa lên thiết bị ở Part 4.
Module results_table giữ nguyên bốn sheet, header và công thức của mẫu Excel;
đồng thời lưu cached value tính từ số đo để file đọc được khi chưa mở bằng Excel.
Các phép đo sức khoẻ và phân tích per-class được thêm vào cuối sheet Legend với exp_id.
Không có checkpoint hoặc dữ liệu trong file nén nộp bài.

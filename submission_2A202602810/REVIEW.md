# Đối chiếu hướng dẫn trong 9 ảnh temp/

Ngày kiểm tra: 2026-10-03. Đã đọc cả chín ảnh từ 13-24-48 đến 13-25-18.

| Mục trong ảnh | Kết quả | Bằng chứng |
| --- | --- | --- |
| Thư mục submission đúng MSSV, code ở code/, đường dẫn notebook đúng | Đạt | submission_2A202602810/code/lab.ipynb; hỗ trợ chạy từ thư mục code gốc hoặc code bản nộp |
| Metadata gốc, train 464809, eval 116203; val phân tầng seed 42 | Đạt | data.py; train còn 371847, val 92962; metadata không bị chỉnh |
| X float32, y int64 0..6; fit chỉ phần train, giữ 44 cột binary | Đạt | _read, fit_standardizer, apply_standardizer; đã kiểm tra lại trực tiếp trong Conda base |
| M-base đúng shape, 47879 tham số, bias/ReLU/dropout đúng, logits thô | Đạt | model.py và output Part 1 |
| Khởi tạo He thực sự; loss bước 0 thật, ghi nhớ 20 mẫu, gradient | Đạt | health_checks.json, notebook, figures/health_overfit20.png; loss He 2.269062, không ép thành ln7 |
| Cùng run_experiment, eval mode khi đo, preclip norm, best checkpoint, time/memory/diverged | Đạt | train.py và 17 results JSON; tất cả hoàn thành 20 epoch |
| Baseline SGD momentum .9, CE/He, wd=0, batch512, FP32, không dropout/clip; lr chọn bằng val | Đạt | base-s1; quét .01/.05, chọn .05 |
| Baseline nhiều seed, mean/std và 2sigma | Đạt | base-s1..3, Seeds; F1 mean 0.834477, std 0.001914, 2sigma 0.003827 |
| Thí nghiệm có dự đoán trước, số liệu, PNG, đối chiếu dưới lần chạy; optimizer >=2lr | Đạt sau chỉnh trình bày | 17 run, đủ 7 chủ đề; notebook đã thay nhận xét chung bằng số liệu và giải thích cụ thể |
| Zeros: kiểm tra gradient theo dự đoán và đường cong | Đạt sau bổ sung bằng chứng | zero_init_gradient_check.json, cell kiểm tra mới, Legend: gradient lớp ẩn 0, bias cuối 0.496712 |
| Chỉ chọn cấu hình bằng val, eval cho baseline và cuối | Đạt | selection.json; seed1 Adam .003; FP16 bổ sung bị loại khỏi chọn cấu hình |
| CSV row_id,pred, đủ 116203 dòng không trùng, nhãn nguyên 0..6 | Đạt | verify_submission.py; ma trận từ CSV khớp JSON chính thức |
| Excel giữ Legend/Experiments/Seeds/Summary, cột và công thức, eval chỉ hai cấu hình | Đạt | đối chiếu công thức từng ô với mẫu; cached values không có lỗi |
| Báo cáo theo mẫu, khoảng 4 trang, số liệu gắn exp_id, per-class/confusion, câu hỏi loss không giảm | Đạt về nội dung; đã rút gọn | REPORT.md còn 1568 từ, hai ảnh nhúng và liên kết các ảnh còn lại; số trang phụ thuộc cách trình bày Markdown |
| Notebook có output, code đầy đủ, không checkpoint/data trong ZIP | Đạt | phiên bản huấn luyện đã chạy hết; giữ nguyên outputs, thêm một cell chẩn đoán zeros đã chạy riêng; tổng 16 cell code có output/thứ tự thực thi |
| Fork cá nhân và nộp link trên hệ thống học | Cần xác nhận bước nộp | origin=https://github.com/nthanhwork/K4-Track4-Day1-Neuralnetwork.git; nhánh preview tracking origin/preview theo refs cục bộ, main ahead origin/main 1 commit; chưa kiểm tra trạng thái nộp LMS |

## Kết luận

Đạt các yêu cầu kỹ thuật nhìn thấy trong ảnh. Không coi số trang Markdown hoặc trạng thái nộp LMS là đã được xác nhận tuyệt đối.
Không thay đổi cấu hình cuối, trọng số/dự đoán hoặc số chấm eval để xử lý các điểm trình bày.

- Official eval: accuracy 0.913032, macro-F1 0.873024.
- Các nhận xét bổ sung/rút gọn báo cáo dùng đúng lịch sử đã đo; không huấn luyện lại hoặc chọn theo eval.
- Bộ sinh báo cáo và runner đã được cập nhật để giữ các cải tiến khi chạy lại.
- Không mở Excel/LibreOffice GUI trong lần kiểm tra này; đã kiểm tra công thức nguyên vẹn và cache tính từ kết quả bằng openpyxl.

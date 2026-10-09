# Baseline Qwen cho TextVQA: ảnh gốc và ảnh suy thoái

Repo này **chỉ suy luận, không huấn luyện**. Model `Qwen/Qwen2.5-VL-3B-Instruct` trả lời cùng 1.000 câu hỏi TextVQA hai lần: với ảnh gốc (`clean_image_path`) và ảnh suy thoái `realistic_mix/L2` (`degraded_image_path`). Hai lượt dùng cùng prompt, model, cấu hình xử lý ảnh và giải mã greedy. Cuối cùng, chương trình chấm điểm và so sánh từng câu hỏi.

Dataset suy thoái: [Data_suy_thoai (Text VQA)](https://www.kaggle.com/datasets/anhthu128/data-suy-thoai-text-vqa).

## 1. Chuẩn bị dữ liệu trên máy H200

**Dataset Kaggle không tự mount trên máy H200 bên ngoài Kaggle.** Người chạy phải tải/copy nó về máy chủ hoặc dùng ổ dữ liệu do quản trị viên gắn sẵn. Chức năng **Add Input** chỉ tự gắn dữ liệu trong Kaggle Notebook. Repo GitHub này chỉ chứa mã, không chứa ảnh hay model.

Cần **hai nguồn ảnh**:

1. Dataset suy thoái: `textvqa_realistic_v2/main/final_main_manifest.csv` và `textvqa_realistic_v2/main/images/realistic_mix/L2/*.png`.
2. **Ảnh TextVQA gốc** đã dùng để tạo bộ suy thoái. Dataset suy thoái không kèm ảnh gốc. `clean_image_path` trong manifest ghi đường dẫn tại lúc tạo dữ liệu; đường dẫn đó có thể không tồn tại trên máy H200.

Ví dụ cấu trúc trên máy chủ:

```text
~/data/
├── data-suy-thoai-text-vqa/
│   └── textvqa_realistic_v2/
│       └── main/
│           ├── final_main_manifest.csv
│           └── images/realistic_mix/L2/*.png
└── textvqa-clean/
    └── ... các ảnh TextVQA gốc tương ứng ...
```

Có thể tải dataset suy thoái từ trang Kaggle bằng trình duyệt rồi chuyển file sang máy chủ. Nếu muốn tải thẳng trên máy chủ, dùng **Kaggle CLI** trong một môi trường riêng:

```bash
python3.11 -m venv "$HOME/.venvs/kaggle-cli"
"$HOME/.venvs/kaggle-cli/bin/python" -m pip install kaggle
"$HOME/.venvs/kaggle-cli/bin/kaggle" auth login --no-launch-browser
mkdir -p "$HOME/data/data-suy-thoai-text-vqa"
"$HOME/.venvs/kaggle-cli/bin/kaggle" datasets download anhthu128/data-suy-thoai-text-vqa \
  -p "$HOME/data/data-suy-thoai-text-vqa" --unzip
```

Làm theo liên kết đăng nhập CLI in ra. **Không đưa token Kaggle vào GitHub.** Cú pháp tải theo [tài liệu Kaggle CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md), xác thực theo [hướng dẫn Kaggle](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication). Nếu dữ liệu đã nằm trên ổ được gắn sẵn, bỏ qua bước tải và dùng đường dẫn của ổ đó.

Ảnh gốc cần được cung cấp riêng. Sau khi đặt dữ liệu, khai báo trong shell sẽ chạy thí nghiệm:

```bash
export TEXTVQA_DATASET_ROOT="$HOME/data/data-suy-thoai-text-vqa"
export TEXTVQA_CLEAN_ROOT="$HOME/data/textvqa-clean"
export HF_HOME="$HOME/.cache/huggingface"
test -f "$TEXTVQA_DATASET_ROOT/textvqa_realistic_v2/main/final_main_manifest.csv"
```

`TEXTVQA_DATASET_ROOT` là **thư mục cha của `textvqa_realistic_v2`**. `TEXTVQA_CLEAN_ROOT` có thể có thư mục con. Chương trình tìm ảnh gốc theo đúng tên file; nếu thiếu hoặc trùng tên sẽ dừng. Người chạy cần bảo đảm nội dung ảnh đúng `image_id`, vì chỉ trùng tên file chưa chứng minh đúng ảnh.

## 2. Lấy mã và cài môi trường

Máy đích: Linux, Python 3.11, CUDA và **đúng một GPU NVIDIA H200 được hiển thị**. Nếu repo Private, tài khoản GitHub của thầy cần được cấp quyền truy cập.

```bash
git clone https://github.com/ncat218/textvqa-qwen-baseline.git
cd textvqa-qwen-baseline
bash run.sh setup
```

`setup` tạo `.venv` trong repo bằng `uv` (nếu có) hoặc `venv` + pip. Các phụ thuộc trực tiếp được ghim phiên bản trong `pyproject.toml` và `requirements.txt`. **Chưa có lock file cho toàn bộ phụ thuộc bắc cầu trên Linux/H200**; nếu cần tái lập môi trường tuyệt đối, tạo và lưu lock file trên máy đích trước lượt chạy chính thức.

Tải model **một lần, rõ ràng**. Các lệnh suy luận chạy offline, không tự tải model:

```bash
.venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct", revision="main")'
```

Để cố định phiên bản model, thay `model_revision: main` trong **cả hai** file YAML bằng cùng một commit hash trên Hugging Face, rồi dùng chính commit đó trong lệnh tải. Xem [model card Qwen](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) và [tài liệu Transformers](https://huggingface.co/docs/transformers/v4.50.0/en/model_doc/qwen2_5_vl).

## 3. Chạy thí nghiệm

Trong **cùng cửa sổ shell** đã đặt các biến môi trường ở mục 1:

```bash
bash run.sh validate       # kiểm tra manifest và ảnh; chưa nạp model
bash run.sh smoke          # 16 câu hỏi clean/degraded; nạp model một lần
bash run.sh base_clean     # đủ 1.000 câu với ảnh gốc
bash run.sh base_degraded  # cùng 1.000 câu với ảnh realistic_mix/L2
bash run.sh score_baseline # chấm hai lượt và so sánh theo từng câu
```

**Chỉ chạy đủ 1.000 câu sau khi `validate` và `smoke` thành công.** Smoke test in ước lượng thời gian của cặp lượt chạy đầy đủ; không dùng điểm của 16 câu để sửa prompt hay cấu hình. `score_baseline` chỉ chấm các kết quả đã có, không nạp model.

Manifest phải có đúng 1.000 dòng, `question_id` duy nhất, 10 đáp án tham chiếu mỗi câu, `condition=realistic_mix` và `level=L2`. Các cột bắt buộc: `image_id`, `question_id`, `question`, `answers_json` hoặc `answers`, `clean_image_path`, `degraded_image_path`, `condition`, `level`, `config_sha256`, `source_manifest_sha256`. Hai cột cũ `clean_path` và `blur_path` **không được dùng**. Chương trình ánh xạ đường dẫn ảnh suy thoái cũ sang vị trí mới trên máy H200 mà không sửa manifest gốc.

## 4. Kết quả

Mỗi lượt nằm trong `outputs/base_clean/` hoặc `outputs/base_realistic_mix_l2/`:

```text
run_metadata.json      thông tin môi trường, model và hash
manifest_used.csv       manifest dùng cho lượt chạy
predictions.jsonl       dự đoán từng câu
predictions.csv         dự đoán dạng bảng
runtime.csv             thời gian từng câu
errors.jsonl            lỗi từng câu (rỗng nếu không lỗi)
summary.json            số dòng và thống kê thời gian
score_report.json       điểm tổng sau khi chấm
scored_predictions.csv  điểm từng câu
```

Mỗi dự đoán có `image_id`, `question_id`, `question`, `image_path`, `condition_name`, `model_id`, `model_revision`, `prompt_template_id`, `raw_prediction`, `normalized_prediction`, `generation_seconds`, `preprocess_seconds`, `total_seconds`, `status`, `error_message`. Dòng lỗi vẫn được ghi; nếu có lỗi, chương trình không báo điểm tổng gây hiểu nhầm.

`outputs/comparison_clean_vs_realistic_mix_l2/` chứa `paired_scores.csv`, `latency_comparison.csv`, `comparison_report.json`: điểm clean/degraded, chênh lệch điểm phần trăm, mức giảm tương đối, số câu điểm giảm/bằng/tăng và thời gian. Chương trình từ chối so sánh hai lượt khi manifest, ID, model, prompt, cấu hình giải mã hoặc môi trường chạy khác nhau.

Điểm dùng chuẩn hóa đáp án kiểu VQA rồi tính `min(số đáp án tham chiếu khớp / 3, 1)` trên 10 đáp án, **đúng công thức trong đặc tả thí nghiệm**. Đây là công thức đồng thuận VQA giản lược; không gọi là bộ chấm TextVQA chính thức nếu chưa đối chiếu evaluator chính thức.

## 5. Kiểm tra không cần GPU và giới hạn hiện tại

```bash
PYTHONPATH=src python3.11 -m unittest discover -s tests -v
```

Đã kiểm tra cú pháp Python và 4 phép thử với ảnh/manifest giả; chúng không tải Qwen hoặc dùng CUDA. **Chưa xác nhận** đường dẫn dữ liệu thật, tải model, chạy H200, thời gian và điểm thực tế. Bước `validate` rồi `smoke` trên máy H200 sẽ kiểm tra các phần này trước lượt chạy đầy đủ.

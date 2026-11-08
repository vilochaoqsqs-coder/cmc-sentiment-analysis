"""
Script phân loại chủ đề và gán nhãn cảm xúc (Tích cực, Bình thường, Tiêu cực)
cho toàn bộ 1,083 bản ghi trong `data/cmc_dataset_clean.csv`, xuất ra
`data/cmc_dataset_labeled.csv` để phục vụ huấn luyện và hiển thị trên Dashboard.

Cách chạy:
    python data/label_dataset.py
"""

import csv
from pathlib import Path
import sys
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from model.sentiment import PhoBERTSentimentAnalyzer

INPUT_CSV_PATH = BASE_DIR / "data" / "cmc_dataset_clean.csv"
OUTPUT_CSV_PATH = BASE_DIR / "data" / "cmc_dataset_labeled.csv"

TOPIC_RULES: Dict[str, List[str]] = {
    "Hoạt động & CLB": [
        "câu lạc bộ", "clb", "casting", "tuyển thành viên", "gen ", "fest",
        "sự kiện", "ban nhạc", "nhảy", "múa", "khởi nghiệp", "hatch", "gim",
        "the ix", "okc", "rebels", "esport", "free fire", "ctf", "giải đấu",
        "tình nguyện", "club day",
    ],
    "Đời sống & Nhà trọ": [
        "trọ", "roommate", "roomate", "ở ghép", "phòng ở", "tiền phòng",
        "vệ sinh khép kín", "vạn phúc", "hà đông", "căng tin", "đồ ăn",
        "gửi xe", "xe điện", "ship", "billiards", "minecraft", "server",
    ],
    "Tìm đồ & Hỗ trợ": [
        "để quên", "nhặt được", "đánh rơi", "tìm lại", "thẻ sinh viên",
        "móc khoá", "chìa khoá", "ipad", "sạc dự phòng", "điện thoại",
        "khảo sát", "xin lại", "hậu tạ",
    ],
    "Cơ sở vật chất & CNTT": [
        "cơ sở vật chất", "phòng thực hành", "phòng học", "tầng ", "cơ sở 1",
        "cơ sở 2", "wifi", "mạng", "điều hòa", "thư viện", "thang máy",
        "itest", "đăng ký môn học", "cổng thông tin",
    ],
    "Học vụ & Giảng viên": [
        "giảng viên", "thầy", "cô", "môn", "học phần", "tín chỉ", "lịch thi",
        "lịch học", "bài tập", "hạn nộp bài", "điểm", "qua môn", "trượt môn",
        "công tác sinh viên", "học bổng", "thực tập", "tiếng anh", "toán",
    ],
}


def classify_topic(text: str) -> str:
    """Phân loại chủ đề bài viết/bình luận sinh viên dựa trên từ khóa đặc trưng."""
    t = text.lower()
    best_topic = "Giao lưu & Khác"
    best_score = 0
    for topic, keywords in TOPIC_RULES.items():
        score = sum(1 for kw in keywords if kw in t)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic


def main() -> None:
    if not INPUT_CSV_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_CSV_PATH}")

    analyzer = PhoBERTSentimentAnalyzer(use_pretrained_backbone=False)

    with open(INPUT_CSV_PATH, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    labeled_rows = []
    label_counts = {"Tích cực": 0, "Bình thường": 0, "Tiêu cực": 0}
    topic_counts: Dict[str, int] = {}

    for row in rows:
        content = row.get("content", "")
        res = analyzer.predict(content)
        topic = classify_topic(content)

        label_counts[res.label] = label_counts.get(res.label, 0) + 1
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

        labeled_rows.append(
            {
                "id": row.get("id", ""),
                "type": row.get("type", "post"),
                "time": row.get("time", ""),
                "topic": topic,
                "content": content,
                "cleaned_content": res.preprocessed_text,
                "likes": int(row.get("likes", 0) or 0),
                "label": res.label,
                "sentiment": res.label,
                "confidence": res.confidence,
            }
        )

    fieldnames = [
        "id",
        "type",
        "time",
        "topic",
        "content",
        "cleaned_content",
        "likes",
        "label",
        "sentiment",
        "confidence",
    ]
    with open(OUTPUT_CSV_PATH, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled_rows)

    print("=" * 55)
    print("HOÀN TẤT GÁN NHÃN CẢM XÚC & PHÂN LOẠI CHỦ ĐỀ")
    print("=" * 55)
    print(f"- File đầu vào  : {INPUT_CSV_PATH.name} ({len(rows)} dòng)")
    print(f"- File đầu ra   : {OUTPUT_CSV_PATH.name}")
    print("- Phân bố cảm xúc:")
    for lbl, cnt in label_counts.items():
        pct = (cnt / len(rows) * 100) if rows else 0
        print(f"    + {lbl:<12}: {cnt:>4} dòng ({pct:.1f}%)")
    print("- Phân bố chủ đề:")
    for tp, cnt in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    + {tp:<22}: {cnt:>4} dòng")
    print("=" * 55)


if __name__ == "__main__":
    main()


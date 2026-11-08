"""
Script đọc toàn bộ các file dataset_*.json trong thư mục `data/`, trích xuất
bài đăng chính (post, bao gồm cả sharedPost nếu có) cùng toàn bộ bình luận (comment),
chuẩn hóa teencode qua `data/teencode.json` và xuất ra file `data/cmc_dataset_clean.csv`.

Cách chạy:
    python data/extract_json_to_csv.py
"""

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEENCODE_PATH = DATA_DIR / "teencode.json"
OUTPUT_CSV_PATH = DATA_DIR / "cmc_dataset_clean.csv"


def load_teencode_normalizer(teencode_path: Path):
    """Tải từ điển teencode.json và tạo hàm chuẩn hóa từ viết tắt theo ranh giới từ."""
    if not teencode_path.exists():
        return lambda s: s

    with open(teencode_path, mode="r", encoding="utf-8") as f:
        raw = json.load(f)

    teencode_dict = {str(k).strip().lower(): str(v).strip() for k, v in raw.items() if str(k).strip()}
    if not teencode_dict:
        return lambda s: s

    sorted_keys = sorted(teencode_dict.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted_keys) + r")(?!\w)",
        flags=re.IGNORECASE,
    )

    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFC", str(text))

        def _replace(match: re.Match) -> str:
            word = match.group(0)
            return teencode_dict.get(word.lower(), word)

        return re.sub(r"\s+", " ", pattern.sub(_replace, text)).strip()

    return _normalize


def clean_whitespace(text: Any) -> str:
    """Chuẩn hóa khoảng trắng và xuống dòng thành chuỗi gọn gàng."""
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def is_valid_text(text: str, min_words: int = 3) -> bool:
    """Kiểm tra văn bản không trống và có ít nhất `min_words` từ."""
    if not text:
        return False
    words = text.split()
    return len(words) >= min_words


def parse_likes(val: Any) -> int:
    """Chuyển đổi giá trị likes sang số nguyên an toàn."""
    if val is None:
        return 0
    try:
        return int(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0


def extract_comments_from_post(post: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Lấy danh sách toàn bộ bình luận từ cả trường `comments` và `topComments`
    (bao gồm cả bình luận trong `sharedPost` và bình luận phản hồi lồng nhau).
    """
    raw_comments: List[Dict[str, Any]] = []
    for key in ("comments", "topComments"):
        items = post.get(key)
        if isinstance(items, list):
            raw_comments.extend(items)

    shared = post.get("sharedPost")
    if isinstance(shared, dict):
        for key in ("comments", "topComments"):
            items = shared.get(key)
            if isinstance(items, list):
                raw_comments.extend(items)

    all_comments: List[Dict[str, Any]] = []
    queue = list(raw_comments)
    while queue:
        c = queue.pop(0)
        if not isinstance(c, dict):
            continue
        all_comments.append(c)
        for reply_key in ("comments", "replies", "topComments"):
            replies = c.get(reply_key)
            if isinstance(replies, list):
                queue.extend(replies)

    return all_comments


def main() -> None:
    json_files = sorted(DATA_DIR.glob("dataset_*.json"))
    if not json_files:
        raise FileNotFoundError(f"Không tìm thấy file dataset_*.json nào trong {DATA_DIR}")

    normalize_teencode = load_teencode_normalizer(TEENCODE_PATH)

    rows: List[Dict[str, Any]] = []
    seen_contents: Set[str] = set()
    post_count = 0
    comment_count = 0
    total_words = 0
    raw_total = 0

    for json_path in json_files:
        with open(json_path, mode="r", encoding="utf-8") as f:
            posts_data = json.load(f)

        file_posts = 0
        file_comments = 0
        for post_idx, post in enumerate(posts_data, start=1):
            if not isinstance(post, dict):
                continue

            # 1. Trích xuất bài đăng chính (post)
            raw_total += 1
            post_content = clean_whitespace(post.get("text") or post.get("message") or "")
            shared = post.get("sharedPost")
            if isinstance(shared, dict):
                shared_text = clean_whitespace(shared.get("text") or shared.get("message") or "")
                if shared_text and shared_text not in post_content:
                    post_content = f"{post_content} {shared_text}".strip() if post_content else shared_text

            post_norm_key = post_content.lower()

            if is_valid_text(post_content, min_words=3) and post_norm_key not in seen_contents:
                seen_contents.add(post_norm_key)
                post_id = str(
                    post.get("legacyId")
                    or post.get("postId")
                    or post.get("id")
                    or f"post_{json_path.stem}_{post_idx}"
                )
                post_time = str(post.get("time") or post.get("date") or "")
                post_likes = parse_likes(
                    post.get("likesCount", post.get("likes", post.get("reactionLikeCount", 0)))
                )
                normalized_post = normalize_teencode(post_content)
                total_words += len(normalized_post.split())

                rows.append(
                    {
                        "id": post_id,
                        "type": "post",
                        "time": post_time,
                        "content": normalized_post,
                        "likes": post_likes,
                        "label": "",
                    }
                )
                post_count += 1
                file_posts += 1

            # 2. Trích xuất toàn bộ bình luận (comments / topComments) của bài đăng
            comments_list = extract_comments_from_post(post)
            raw_total += len(comments_list)
            for c_idx, comment in enumerate(comments_list, start=1):
                c_content = clean_whitespace(comment.get("text") or comment.get("message") or "")
                c_norm_key = c_content.lower()

                if is_valid_text(c_content, min_words=3) and c_norm_key not in seen_contents:
                    seen_contents.add(c_norm_key)
                    c_id = str(
                        comment.get("commentId")
                        or comment.get("id")
                        or f"comment_{json_path.stem}_{post_idx}_{c_idx}"
                    )
                    c_time = str(comment.get("date") or comment.get("time") or post.get("time") or "")
                    c_likes = parse_likes(
                        comment.get("likesCount", comment.get("likes", comment.get("reactionLikeCount", 0)))
                    )
                    normalized_comment = normalize_teencode(c_content)
                    total_words += len(normalized_comment.split())

                    rows.append(
                        {
                            "id": c_id,
                            "type": "comment",
                            "time": c_time,
                            "content": normalized_comment,
                            "likes": c_likes,
                            "label": "",
                        }
                    )
                    comment_count += 1
                    file_comments += 1

        print(f"[FILE] {json_path.name}: +{file_posts} bài đăng, +{file_comments} bình luận hợp lệ")

    # Lưu kết quả ra file CSV với encoding utf-8-sig
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "type", "time", "content", "likes", "label"]

    with open(OUTPUT_CSV_PATH, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    avg_words = total_words / len(rows) if rows else 0.0
    print("=" * 60)
    print("HOÀN TẤT TRÍCH XUẤT & CHUẨN HÓA TẤT CẢ DATASET JSON SANG CSV")
    print("=" * 60)
    print(f"- Số file JSON đã đọc  : {len(json_files)} file")
    print(f"- Tổng bản ghi thô     : {raw_total:,} dòng")
    print(f"- File đầu ra          : {OUTPUT_CSV_PATH}")
    print(f"- Tổng số bài đăng     : {post_count:,}")
    print(f"- Tổng số bình luận    : {comment_count:,}")
    print(f"- Tổng số dòng sạch    : {len(rows):,}")
    print(f"- Độ dài câu TB        : {avg_words:.2f} từ/câu")
    print("=" * 60)


if __name__ == "__main__":
    main()

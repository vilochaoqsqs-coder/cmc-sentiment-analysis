"""
Script thu thập dữ liệu bài viết/bình luận sinh viên bằng Playwright.
Hỗ trợ cả chế độ cào trang web thực tế và chế độ tạo dữ liệu mô phỏng (demo)
để kiểm thử toàn bộ pipeline ngay lập tức.

Cách chạy:
    python crawler/scraper.py --demo
    python crawler/scraper.py --url "https://example.com/forum" --selector ".post-content"
"""

import argparse
import asyncio
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = BASE_DIR / "data" / "crawled_posts.csv"

DEMO_POSTS: List[Dict[str, str]] = [
    {
        "source": "CMC Confessions",
        "topic": "Cơ sở vật chất",
        "content": "Phòng lab tầng 5 mới nâng cấp dàn PC xịn sò quá, code đồ án mượt lun!",
    },
    {
        "source": "Hội Sinh Viên CMC",
        "topic": "Hệ thống CNTT",
        "content": "Cổng dkmh sáng nay hơi lag, mik đợi 15p mới đk đc lớp Tiếng Anh.",
    },
    {
        "source": "Diễn đàn SV",
        "topic": "Học vụ",
        "content": "Lịch học kỳ tới bắt đầu từ tuần đầu tháng 12 nha mọi người.",
    },
    {
        "source": "CMC Confessions",
        "topic": "Giảng viên",
        "content": "Cô dạy môn Trí tuệ nhân tạo cực kỳ tâm huyết, chấm bài có feedback chi tiết cho từng nhóm.",
    },
    {
        "source": "CMC Confessions",
        "topic": "Đời sống SV",
        "content": "Bãi gửi xe giờ cao điểm hơi chật, lấy xe ra mất khá nhiều thời gian.",
    },
]


async def crawl_posts_playwright(
    url: str,
    post_selector: str = "article, .post-content, p",
    max_scrolls: int = 3,
    headless: bool = True,
) -> List[Dict[str, str]]:
    """
    Sử dụng Playwright để mở trình duyệt, cuộn trang và trích xuất nội dung bài viết.
    """
    from playwright.async_api import async_playwright

    results: List[Dict[str, str]] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()
        print(f"[INFO] Đang truy cập: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)

        for i in range(max_scrolls):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1500)
            print(f"[INFO] Đã cuộn trang lần {i + 1}/{max_scrolls}")

        elements = await page.query_selector_all(post_selector)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for idx, el in enumerate(elements, start=1):
            text = (await el.inner_text()).strip()
            if len(text) >= 15:
                results.append(
                    {
                        "id": str(idx),
                        "timestamp": now_str,
                        "source": url,
                        "topic": "Chưa phân loại",
                        "content": " ".join(text.splitlines()),
                    }
                )

        await browser.close()
    return results


def save_to_csv(posts: List[Dict[str, str]], output_path: Path) -> None:
    """Lưu danh sách bài viết thu thập được ra file CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "timestamp", "source", "topic", "content"]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, item in enumerate(posts, start=1):
            writer.writerow(
                {
                    "id": item.get("id", str(idx)),
                    "timestamp": item.get("timestamp", now_str),
                    "source": item.get("source", "Playwright Crawler"),
                    "topic": item.get("topic", "Tổng hợp"),
                    "content": item.get("content", ""),
                }
            )
    print(f"[SUCCESS] Đã lưu {len(posts)} bài viết vào: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Playwright Crawler thu thập bài viết quan điểm sinh viên CMC."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="",
        help="URL trang web cần cào dữ liệu (bỏ trống hoặc dùng --demo để tạo dữ liệu mẫu).",
    )
    parser.add_argument(
        "--selector",
        type=str,
        default="article, .post-content, p",
        help="CSS Selector của phần tử chứa nội dung bài viết.",
    )
    parser.add_argument(
        "--scrolls",
        type=int,
        default=3,
        help="Số lần cuộn trang để tải thêm bài viết.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Đường dẫn file CSV đầu ra.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Chạy chế độ mô phỏng dữ liệu sinh viên mà không cần kết nối mạng.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if args.demo or not args.url:
        print("[INFO] Chạy chế độ DEMO thu thập dữ liệu bài viết sinh viên...")
        save_to_csv(DEMO_POSTS, output_path)
    else:
        posts = asyncio.run(
            crawl_posts_playwright(
                url=args.url,
                post_selector=args.selector,
                max_scrolls=args.scrolls,
            )
        )
        save_to_csv(posts, output_path)


if __name__ == "__main__":
    main()


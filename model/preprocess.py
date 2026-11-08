"""
Module tiền xử lý văn bản tiếng Việt cho bài toán phân tích quan điểm sinh viên:
1. Chuẩn hóa bảng mã Unicode (NFC) và chữ thường.
2. Loại bỏ URL, HTML tags, ký tự lặp bất thường.
3. Chuẩn hóa teencode/từ viết tắt sinh viên từ file `data/teencode.json`.
4. Tách từ tiếng Việt (word segmentation) bằng `underthesea` để đưa vào PhoBERT.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional

try:
    from underthesea import word_tokenize
except ImportError:
    word_tokenize = None

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TEENCODE_PATH = BASE_DIR / "data" / "teencode.json"


class VietnameseTextPreprocessor:
    """Bộ tiền xử lý văn bản tiếng Việt tích hợp từ điển Teencode và Underthesea."""

    def __init__(self, teencode_path: Optional[Path] = None) -> None:
        self.teencode_path = teencode_path or DEFAULT_TEENCODE_PATH
        self.teencode_dict: Dict[str, str] = self._load_teencode(self.teencode_path)

    @staticmethod
    def _load_teencode(path: Path) -> Dict[str, str]:
        if path.exists():
            with open(path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
                return {str(k).strip().lower(): str(v).strip().lower() for k, v in data.items()}
        return {}

    def normalize_teencode(self, text: str) -> str:
        """Thay thế các từ teencode/viết tắt bằng từ tiếng Việt chuẩn."""
        tokens = text.split()
        normalized_tokens = [self.teencode_dict.get(tok, tok) for tok in tokens]
        return " ".join(normalized_tokens)

    def clean_text(self, text: str) -> str:
        """Làm sạch văn bản thô và chuẩn hóa teencode."""
        if not text:
            return ""
        # 1. Chuẩn hóa Unicode dựng sẵn (NFC) và chuyển về chữ thường
        text = unicodedata.normalize("NFC", str(text)).lower().strip()
        # 2. Xóa đường dẫn URL và email
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        # 3. Giảm các ký tự kéo dài liên tiếp (vd: "quáaaaa" -> "quá")
        text = re.sub(r"([a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ])\1{2,}", r"\1", text)
        # 4. Giữ lại chữ cái, chữ số và khoảng trắng
        text = re.sub(
            r"[^0-9a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ\s]",
            " ",
            text,
        )
        text = re.sub(r"\s+", " ", text).strip()
        # 5. Chuẩn hóa teencode
        text = self.normalize_teencode(text)
        return re.sub(r"\s+", " ", text).strip()

    def segment_words(self, text: str) -> str:
        """Tách từ tiếng Việt bằng underthesea (ví dụ: 'sinh viên' -> 'sinh_viên')."""
        if not text:
            return ""
        if word_tokenize is not None:
            return word_tokenize(text, format="text")
        return text

    def preprocess(self, text: str, segment: bool = True) -> str:
        """Thực hiện toàn bộ pipeline tiền xử lý văn bản đầu vào cho PhoBERT."""
        cleaned = self.clean_text(text)
        if segment:
            return self.segment_words(cleaned)
        return cleaned


if __name__ == "__main__":
    preprocessor = VietnameseTextPreprocessor()
    sample = "CSVC trường mik xịn sò thực sự, nhưng cổng dkmh hôm nay hơi lag nha!"
    print("Câu gốc      :", sample)
    print("Sau xử lý    :", preprocessor.preprocess(sample))


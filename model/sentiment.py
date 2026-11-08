"""
Module suy luận cảm xúc (Tiêu cực, Bình thường, Tích cực) sử dụng mô hình PhoBERT
đã fine-tune lưu tại đường dẫn local `model/cmc_phobert_sentiment`.

Pipeline suy luận:
1. Chuẩn hóa văn bản và từ viết tắt sinh viên qua `data/teencode.json`.
2. Tách từ tiếng Việt bằng `underthesea.word_tokenize(..., format='text')`.
3. Đưa văn bản đã tách từ vào tokenizer & model `model/cmc_phobert_sentiment`.
4. Tính xác suất Softmax và trả về nhãn (`Tiêu cực`, `Bình thường`, `Tích cực`)
   kèm phần trăm độ tự tin (confidence score).
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from model.preprocess import VietnameseTextPreprocessor

try:
    from underthesea import word_tokenize
except ImportError:
    word_tokenize = None

DEFAULT_MODEL_DIR = BASE_DIR / "model" / "cmc_phobert_sentiment"
DEFAULT_TEENCODE_PATH = BASE_DIR / "data" / "teencode.json"

ID2LABEL: Dict[int, str] = {
    0: "Tiêu cực",
    1: "Bình thường",
    2: "Tích cực",
}
LABELS: List[str] = ["Tiêu cực", "Bình thường", "Tích cực"]

# Từ khóa dự phòng trong trường hợp môi trường chưa cài đặt torch/transformers
POSITIVE_KEYWORDS = [
    "tốt", "rất tốt", "xuất_sắc", "xịn", "mượt", "nhiệt_tình", "dễ hiểu",
    "tâm_huyết", "hiện_đại", "mát", "yên_tĩnh", "thích", "yêu", "hài_lòng",
    "hợp_lý", "nhanh", "hữu_ích", "tuyệt_vời", "cháy", "đỉnh", "qua môn",
    "thực_chiến", "cơ_hội", "hỗ_trợ", "thân_thiện", "sạch_sẽ", "đẹp",
    "xinh_đẹp", "xinh", "ấn_tượng", "nhiệt_huyết", "tự_hào", "tài_năng",
    "hấp_dẫn", "vui", "cảm_ơn", "ủng_hộ", "hân_hạnh", "đam_mê", "an_toàn",
]

NEGATIVE_KEYWORDS = [
    "chậm", "lỗi", "lỗi hệ_thống", "thất_vọng", "mệt_mỏi", "chập_chờn",
    "kém", "tệ", "đắt", "nóng", "ồn", "chật", "đông", "trượt môn",
    "quá_tải", "bất_tiện", "khó", "ức_chế", "huhu", "không được",
    "không_thể", "mất thời_gian", "dồn_dập", "áp_lực", "than_phiền",
    "vô_duyên", "xàm", "lừa_đảo", "mất", "đánh_rơi", "để quên", "sợ",
]


@dataclass
class SentimentResult:
    raw_text: str
    preprocessed_text: str
    label: str
    confidence: float          # Giá trị từ 0.0 -> 1.0 (ví dụ: 0.9425)
    confidence_percent: float  # Giá trị phần trăm từ 0.0 -> 100.0 (ví dụ: 94.25)
    probabilities: Dict[str, float]
    backend: str


class PhoBERTSentimentAnalyzer:
    """Bộ phân tích cảm xúc bài viết sinh viên CMC từ mô hình fine-tuned local."""

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, Path]] = None,
        teencode_path: Optional[Path] = None,
        use_pretrained_backbone: bool = True,
    ) -> None:
        resolved_path = Path(model_name_or_path) if model_name_or_path else DEFAULT_MODEL_DIR
        self.model_dir = resolved_path if resolved_path.exists() else DEFAULT_MODEL_DIR
        self.model_name_or_path = str(self.model_dir if self.model_dir.exists() else (model_name_or_path or DEFAULT_MODEL_DIR))
        self.preprocessor = VietnameseTextPreprocessor(teencode_path=teencode_path or DEFAULT_TEENCODE_PATH)
        self.tokenizer = None
        self.model = None
        self.id2label: Dict[int, str] = dict(ID2LABEL)
        self.device = "cpu"
        self.backend_status = "Fallback Lexicon Mode"

        if use_pretrained_backbone:
            self._load_local_phobert()

    def _load_local_phobert(self) -> None:
        """Tải trực tiếp tokenizer và mô hình đã fine-tune từ `model/cmc_phobert_sentiment`."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            target_path = str(self.model_dir) if self.model_dir.exists() else self.model_name_or_path

            self.tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=self.model_dir.exists())
            self.model = AutoModelForSequenceClassification.from_pretrained(
                target_path,
                local_files_only=self.model_dir.exists(),
            )
            self.model.to(self.device)
            self.model.eval()

            # Đọc cấu hình id2label trực tiếp từ config.json của mô hình fine-tune
            cfg_id2label = getattr(self.model.config, "id2label", None)
            if isinstance(cfg_id2label, dict) and cfg_id2label:
                self.id2label = {int(k): str(v) for k, v in cfg_id2label.items()}

            self.backend_status = f"Fine-tuned PhoBERT ({Path(target_path).name}) [{self.device.upper()}]"
        except Exception as exc:
            self.tokenizer = None
            self.model = None
            self.backend_status = f"Fallback Mode (Chưa nạp được PhoBERT local: {type(exc).__name__})"

    def preprocess_input(self, text: str) -> str:
        """
        Chuẩn hóa câu đầu vào qua `data/teencode.json` và tách từ tiếng Việt
        bằng `underthesea.word_tokenize(..., format='text')`.
        """
        cleaned = self.preprocessor.clean_text(text)
        if not cleaned:
            return ""
        if word_tokenize is not None:
            return word_tokenize(cleaned, format="text")
        return self.preprocessor.segment_words(cleaned)

    def _fallback_probs(self, segmented_text: str) -> List[float]:
        """Hàm dự phòng tính xác suất khi chạy ở môi trường chưa cài đặt PyTorch."""
        import math

        text_lower = segmented_text.lower()
        pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
        neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)

        if pos_hits > neg_hits:
            logits = [-0.9, 0.1, 1.35 + 0.45 * min(pos_hits, 4)]
        elif neg_hits > pos_hits:
            logits = [1.35 + 0.45 * min(neg_hits, 4), 0.1, -0.9]
        elif pos_hits == 0 and neg_hits == 0:
            logits = [-0.5, 1.4, -0.5]
        else:
            logits = [0.4 * neg_hits, 0.9, 0.4 * pos_hits]

        exp_scores = [math.exp(x) for x in logits]
        total = sum(exp_scores)
        return [s / total for s in exp_scores]

    def predict(self, text: str) -> SentimentResult:
        """
        Dự đoán cảm xúc cho văn bản sinh viên:
        - Chuẩn hóa teencode (`data/teencode.json`) + tách từ (`underthesea`).
        - Suy luận bằng mô hình `model/cmc_phobert_sentiment`.
        - Trả về `SentimentResult` gồm nhãn (`Tiêu cực`, `Bình thường`, `Tích cực`)
          và độ tự tin (`confidence`, `confidence_percent`).
        """
        preprocessed = self.preprocess_input(text)
        if not preprocessed:
            probs = {"Tiêu cực": 0.05, "Bình thường": 0.90, "Tích cực": 0.05}
            return SentimentResult(
                raw_text=text,
                preprocessed_text="",
                label="Bình thường",
                confidence=0.90,
                confidence_percent=90.0,
                probabilities=probs,
                backend=self.backend_status,
            )

        if self.model is not None and self.tokenizer is not None:
            import torch

            inputs = self.tokenizer(
                preprocessed,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                prob_tensor = torch.softmax(outputs.logits.squeeze(0), dim=-1).cpu().tolist()

            probabilities = {
                self.id2label.get(idx, LABELS[idx]): round(float(p), 4)
                for idx, p in enumerate(prob_tensor)
            }
        else:
            prob_list = self._fallback_probs(preprocessed)
            probabilities = {
                label: round(float(p), 4) for label, p in zip(LABELS, prob_list)
            }

        best_label = max(probabilities, key=probabilities.get)
        confidence = round(float(probabilities[best_label]), 4)
        confidence_percent = round(confidence * 100.0, 2)

        return SentimentResult(
            raw_text=text,
            preprocessed_text=preprocessed,
            label=best_label,
            confidence=confidence,
            confidence_percent=confidence_percent,
            probabilities=probabilities,
            backend=self.backend_status,
        )

    def predict_label_and_confidence(self, text: str) -> Tuple[str, float, str]:
        """Trả về tuple `(label, confidence_percent, preprocessed_text)` gọn nhẹ."""
        res = self.predict(text)
        return res.label, res.confidence_percent, res.preprocessed_text


if __name__ == "__main__":
    analyzer = PhoBERTSentimentAnalyzer(model_name_or_path=DEFAULT_MODEL_DIR)
    print("Engine:", analyzer.backend_status)
    test_sentences = [
        "CSVC trường mik xịn sò thực sự, phòng lab máy cấu hình cao!",
        "Web dkmh hôm nay lag quá, ko đk đc môn nào hết.",
        "Cho mik hỏi lịch thi cuối kỳ học phần Toán rời rạc xem ở đâu z mn?",
    ]
    for sentence in test_sentences:
        res = analyzer.predict(sentence)
        print("-" * 60)
        print(f"Câu gốc        : {res.raw_text}")
        print(f"Sau tiền xử lý : {res.preprocessed_text}")
        print(f"Nhãn dự đoán   : {res.label} (Độ tự tin: {res.confidence_percent:.2f}%)")
        print(f"Phân bố xác suất: {res.probabilities}")

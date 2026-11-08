"""
Giao diện Dashboard Streamlit phân tích quan điểm sinh viên Trường Đại học CMC.
Sử dụng mô hình PhoBERT đã fine-tune tại `model/cmc_phobert_sentiment`
và tập dữ liệu thực tế `data/cmc_dataset_labeled.csv`.

Cách chạy:
    streamlit run app.py
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from model.sentiment import DEFAULT_MODEL_DIR, PhoBERTSentimentAnalyzer

BASE_DIR = Path(__file__).resolve().parent
LABELED_DATA_PATH = BASE_DIR / "data" / "cmc_dataset_labeled.csv"

COLOR_MAP = {
    "Tích cực": "#10B981",
    "Bình thường": "#3B82F6",
    "Tiêu cực": "#EF4444",
}

st.set_page_config(
    page_title="CMC Student Sentiment Analysis",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource(show_spinner="Đang nạp mô hình PhoBERT fine-tuned (model/cmc_phobert_sentiment)...")
def load_phobert_analyzer() -> PhoBERTSentimentAnalyzer:
    """
    Chỉ nạp mô hình PhoBERT từ thư mục local `model/cmc_phobert_sentiment`
    đúng 1 lần duy nhất và giữ trong bộ nhớ cache giúp giao diện chạy mượt trên laptop.
    """
    return PhoBERTSentimentAnalyzer(
        model_name_or_path=DEFAULT_MODEL_DIR,
        use_pretrained_backbone=True,
    )


@st.cache_data(show_spinner="Đang tải dữ liệu bài viết và bình luận sinh viên CMC...")
def load_labeled_dataset(csv_path: Path) -> pd.DataFrame:
    """Đọc dữ liệu đã gán nhãn từ `data/cmc_dataset_labeled.csv`."""
    if not csv_path.exists():
        return pd.DataFrame(
            columns=[
                "id", "type", "time", "topic", "content",
                "cleaned_content", "likes", "label", "sentiment", "confidence",
            ]
        )

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "label" not in df.columns and "sentiment" in df.columns:
        df["label"] = df["sentiment"]
    elif "sentiment" not in df.columns and "label" in df.columns:
        df["sentiment"] = df["label"]

    if "likes" in df.columns:
        df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    return df


def render_tab_dashboard(df: pd.DataFrame) -> None:
    """Hiển thị Tab 1: Dashboard Thống kê từ `data/cmc_dataset_labeled.csv`."""
    if df.empty:
        st.error(f"Không tìm thấy dữ liệu tại `{LABELED_DATA_PATH}`.")
        return

    # --- BỘ LỌC DỮ LIỆU ---
    st.markdown("#### 🔍 Bộ lọc dữ liệu")
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.1, 1.3, 1.1, 1.5])

    with f_col1:
        type_options = {
            "Tất cả (Bài đăng & Bình luận)": "all",
            "📝 Chỉ Bài đăng (post)": "post",
            "💬 Chỉ Bình luận (comment)": "comment",
        }
        selected_type_label = st.selectbox("Loại nội dung:", list(type_options.keys()))
        selected_type = type_options[selected_type_label]

    with f_col2:
        all_topics = ["Tất cả"] + sorted(df["topic"].dropna().unique().tolist())
        selected_topic = st.selectbox("Chủ đề (Topic):", all_topics)

    with f_col3:
        selected_label = st.selectbox(
            "Nhãn cảm xúc:",
            ["Tất cả", "Tích cực", "Bình thường", "Tiêu cực"],
        )

    with f_col4:
        keyword = st.text_input("Tìm kiếm từ khóa trong nội dung:", placeholder="Nhập từ khóa (vd: clb, học phí, wifi...)")

    # Áp dụng bộ lọc
    filtered_df = df.copy()
    if selected_type != "all":
        filtered_df = filtered_df[filtered_df["type"] == selected_type]
    if selected_topic != "Tất cả":
        filtered_df = filtered_df[filtered_df["topic"] == selected_topic]
    if selected_label != "Tất cả":
        filtered_df = filtered_df[filtered_df["label"] == selected_label]
    if keyword.strip():
        kw = keyword.strip().lower()
        filtered_df = filtered_df[filtered_df["content"].fillna("").str.lower().str.contains(kw, na=False)]

    # --- THẺ KPI TỔNG QUAN ---
    total_records = len(filtered_df)
    post_cnt = int((filtered_df["type"] == "post").sum())
    comment_cnt = int((filtered_df["type"] == "comment").sum())
    pos_cnt = int((filtered_df["label"] == "Tích cực").sum())
    neu_cnt = int((filtered_df["label"] == "Bình thường").sum())
    neg_cnt = int((filtered_df["label"] == "Tiêu cực").sum())

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric(
        "📊 Tổng bản ghi",
        f"{total_records:,}",
        f"{post_cnt:,} posts | {comment_cnt:,} comments",
    )
    kpi2.metric(
        "😊 Tích cực",
        f"{pos_cnt:,}",
        f"{(pos_cnt / total_records * 100) if total_records else 0:.1f}%",
    )
    kpi3.metric(
        "😐 Bình thường",
        f"{neu_cnt:,}",
        f"{(neu_cnt / total_records * 100) if total_records else 0:.1f}%",
    )
    kpi4.metric(
        "😟 Tiêu cực",
        f"{neg_cnt:,}",
        f"{(neg_cnt / total_records * 100) if total_records else 0:.1f}%",
    )
    kpi5.metric(
        "👍 Tổng lượt thích",
        f"{int(filtered_df['likes'].sum()):,}",
        f"TB: {filtered_df['likes'].mean() if total_records else 0:.1f} likes/bài",
    )

    st.divider()

    # --- BIỂU ĐỒ TRÒN & BIỂU ĐỒ CỘT THEO CHỦ ĐỀ ---
    chart_col1, chart_col2 = st.columns([1, 1.35])

    with chart_col1:
        st.subheader("🥧 Tỷ lệ 3 nhãn cảm xúc")
        if not filtered_df.empty:
            pie_df = (
                filtered_df["label"]
                .value_counts()
                .reindex(["Tích cực", "Bình thường", "Tiêu cực"], fill_value=0)
                .reset_index()
            )
            pie_df.columns = ["label", "count"]
            pie_df = pie_df[pie_df["count"] > 0]
            fig_pie = px.pie(
                pie_df,
                names="label",
                values="count",
                color="label",
                color_discrete_map=COLOR_MAP,
                hole=0.42,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label+value")
            fig_pie.update_layout(margin=dict(t=20, b=20, l=10, r=10), legend_title_text="Nhãn cảm xúc")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Không có dữ liệu phù hợp với điều kiện lọc hiện tại.")

    with chart_col2:
        st.subheader("📊 Phân bố cảm xúc theo Chủ đề (Topic)")
        if not filtered_df.empty:
            topic_df = (
                filtered_df.groupby(["topic", "label"])
                .size()
                .reset_index(name="count")
            )
            fig_bar = px.bar(
                topic_df,
                x="topic",
                y="count",
                color="label",
                barmode="group",
                color_discrete_map=COLOR_MAP,
                category_orders={"label": ["Tích cực", "Bình thường", "Tiêu cực"]},
                labels={"topic": "Chủ đề thảo luận", "count": "Số lượng", "label": "Nhãn cảm xúc"},
                text_auto=True,
            )
            fig_bar.update_layout(margin=dict(t=20, b=20, l=10, r=10))
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Không có dữ liệu phù hợp với điều kiện lọc hiện tại.")

    st.divider()

    # --- BẢNG DANH SÁCH BÀI VIẾT & BÌNH LUẬN THỰC TẾ ---
    st.subheader(f"📋 Danh sách bài viết & bình luận thực tế ({len(filtered_df):,} bản ghi)")
    display_cols = ["id", "type", "time", "topic", "content", "likes", "label", "confidence"]
    available_cols = [c for c in display_cols if c in filtered_df.columns]

    display_df = filtered_df[available_cols].copy()
    if "confidence" in display_df.columns:
        display_df["confidence"] = (display_df["confidence"] * 100).round(2).astype(str) + "%"

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "id": st.column_config.TextColumn("ID"),
            "type": st.column_config.TextColumn("Loại"),
            "time": st.column_config.TextColumn("Thời gian"),
            "topic": st.column_config.TextColumn("Chủ đề"),
            "content": st.column_config.TextColumn("Nội dung bài viết / bình luận", width="large"),
            "likes": st.column_config.NumberColumn("Lượt thích"),
            "label": st.column_config.TextColumn("Nhãn cảm xúc"),
            "confidence": st.column_config.TextColumn("Độ tự tin"),
        },
    )


def render_tab_live_ai(analyzer: PhoBERTSentimentAnalyzer) -> None:
    """Hiển thị Tab 2: Phân tích AI Trực tiếp bằng mô hình `cmc_phobert_sentiment`."""
    st.subheader("🤖 Phân tích Cảm xúc Sinh viên Trực tiếp với `cmc_phobert_sentiment`")
    st.markdown(
        "Nhập bất kỳ câu bài viết hoặc bình luận nào của sinh viên trường Đại học CMC "
        "(hỗ trợ tự động chuẩn hóa từ viết tắt qua `data/teencode.json` và tách từ tiếng Việt bằng `underthesea` trước khi đưa vào PhoBERT):"
    )

    sample_options = [
        "Tự nhập nội dung bất kỳ...",
        "CSVC trường mik xịn sò thực sự, phòng lab máy cấu hình cao học AI mượt lun!",
        "Web dkmh sáng nay lag quá, quay đều từ sáng ko đk đc môn nào hết huhu",
        "Cho mik hỏi lịch thi cuối kỳ học phần Toán rời rạc xem ở đâu z mn?",
        "Event chào tân sv của các clb tối qua cháy quá trời, âm thanh ánh sáng đỉnh của chóp!",
        "Bãi gửi xe giờ cao điểm hơi chật, lấy xe ra mất khá nhiều thời gian.",
    ]
    chosen_sample = st.selectbox("💡 Chọn nhanh câu mẫu để thử nghiệm:", sample_options)

    default_text = (
        "CSVC trường mik xịn sò thực sự, phòng lab máy cấu hình cao học AI mượt lun!"
        if chosen_sample == sample_options[0]
        else chosen_sample
    )

    user_text = st.text_area(
        "✍️ Nhập nội dung bình luận / bài viết của sinh viên:",
        value=default_text,
        height=120,
        placeholder="Gõ câu bình luận chứa teencode (vd: mik, ko, đk, dkmh, csvc, clb, gv, sv)...",
    )

    if st.button("🚀 Dự đoán với PhoBERT (`cmc_phobert_sentiment`)", type="primary"):
        if not user_text.strip():
            st.warning("Vui lòng nhập nội dung văn bản trước khi phân tích.")
            return

        with st.spinner("Đang chuẩn hóa teencode, tách từ Underthesea và suy luận bằng PhoBERT..."):
            result = analyzer.predict(user_text)

        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            st.markdown("##### 1️⃣ Văn bản sau khi chuẩn hóa `teencode.json` & tách từ `underthesea`")
            st.code(result.preprocessed_text, language="text")

            badge_color = COLOR_MAP.get(result.label, "#3B82F6")
            st.markdown(
                f"""
                ##### 2️⃣ Kết quả dự đoán từ `cmc_phobert_sentiment`
                <div style="padding: 16px 20px; border-radius: 10px; background-color: {badge_color}20; border-left: 6px solid {badge_color}; margin-top: 8px;">
                    <div style="font-size: 1.4rem; font-weight: 800; color: {badge_color};">
                        Nhãn cảm xúc: {result.label}
                    </div>
                    <div style="margin-top: 6px; font-size: 1.05rem;">
                        Độ tự tin (Confidence Score): <b>{result.confidence_percent:.2f}%</b>
                    </div>
                    <div style="margin-top: 4px; font-size: 0.85rem; opacity: 0.8;">
                        Engine: {result.backend}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            st.markdown("##### 📊 Phân bố xác suất dự đoán (3 lớp)")
            prob_df = pd.DataFrame(
                {
                    "Nhãn cảm xúc": list(result.probabilities.keys()),
                    "Độ tự tin (%)": [round(v * 100, 2) for v in result.probabilities.values()],
                }
            )
            fig_prob = px.bar(
                prob_df,
                x="Độ tự tin (%)",
                y="Nhãn cảm xúc",
                orientation="h",
                color="Nhãn cảm xúc",
                color_discrete_map=COLOR_MAP,
                range_x=[0, 100],
                text_auto=".2f",
            )
            fig_prob.update_layout(
                showlegend=False,
                height=240,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_prob, use_container_width=True)


def main() -> None:
    st.title("🎓 Hệ thống Phân tích Quan điểm Sinh viên Trường Đại học CMC")
    st.caption(
        "Mô hình: Fine-tuned PhoBERT (`model/cmc_phobert_sentiment`) | "
        "Tiền xử lý: `data/teencode.json` + `underthesea` | "
        "Dữ liệu: `data/cmc_dataset_labeled.csv`"
    )

    # Load model 1 lần duy nhất qua @st.cache_resource
    analyzer = load_phobert_analyzer()
    df = load_labeled_dataset(LABELED_DATA_PATH)

    with st.sidebar:
        st.header("🤖 Thông tin Mô hình")
        st.success(f"**Engine đang chạy:**\n{analyzer.backend_status}")
        st.markdown(
            f"""
            - **Đường dẫn model:** `model/cmc_phobert_sentiment`
            - **Từ điển Teencode:** `data/teencode.json`
            - **Tách từ tiếng Việt:** `underthesea`
            - **Dữ liệu thống kê:** `{LABELED_DATA_PATH.name}` ({len(df):,} dòng)
            - **3 nhãn phân lớp:**
              - 🟢 `Tích cực` (ID: 2)
              - 🔵 `Bình thường` (ID: 1)
              - 🔴 `Tiêu cực` (ID: 0)
            """
        )

    tab1, tab2 = st.tabs(
        [
            "📊 Tab 1: Dashboard Thống kê",
            "🤖 Tab 2: Phân tích AI Trực tiếp",
        ]
    )

    with tab1:
        render_tab_dashboard(df)

    with tab2:
        render_tab_live_ai(analyzer)


if __name__ == "__main__":
    main()

"""
Giao diện Dashboard Streamlit phân tích quan điểm sinh viên CMC (cmc-sentiment-analysis).
Hiển thị thống kê tổng quan, biểu đồ trực quan và ô kiểm thử mô hình trực tiếp.

Cách chạy:
    streamlit run app.py
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from model.sentiment import PhoBERTSentimentAnalyzer

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = BASE_DIR / "data" / "sample_posts.csv"
TEENCODE_PATH = BASE_DIR / "data" / "teencode.json"

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


@st.cache_resource(show_spinner="Đang khởi tạo mô hình PhoBERT & Underthesea...")
def load_analyzer(use_phobert: bool) -> PhoBERTSentimentAnalyzer:
    return PhoBERTSentimentAnalyzer(
        model_name_or_path="vinai/phobert-base-v2",
        use_pretrained_backbone=use_phobert,
    )


@st.cache_data
def load_dataset(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        return df
    return pd.DataFrame(
        columns=["id", "timestamp", "source", "topic", "content", "cleaned_content", "sentiment", "confidence"]
    )


def main() -> None:
    st.title("🎓 Hệ thống Phân tích Quan điểm Sinh viên CMC")
    st.caption(
        "Pipeline: Playwright Crawler → Chuẩn hóa Teencode & Tách từ Underthesea → Suy luận cảm xúc với `vinai/phobert-base-v2`"
    )

    with st.sidebar:
        st.header("⚙️ Cấu hình hệ thống")
        use_phobert = st.toggle(
            "Tải trọng số `vinai/phobert-base-v2` từ HuggingFace",
            value=False,
            help="Bật để nạp trực tiếp mô hình PhoBERT (~540MB) qua thư viện transformers. Tắt để chạy chế độ suy luận nhanh.",
        )
        analyzer = load_analyzer(use_phobert)
        st.info(f"**Trạng thái Engine:**\n{analyzer.backend_status}")

        df = load_dataset(SAMPLE_DATA_PATH)
        topics = ["Tất cả"] + sorted(df["topic"].dropna().unique().tolist()) if not df.empty else ["Tất cả"]
        selected_topic = st.selectbox("Lọc theo chủ đề:", topics)

        sentiments = ["Tất cả", "Tích cực", "Bình thường", "Tiêu cực"]
        selected_sentiment = st.selectbox("Lọc theo cảm xúc:", sentiments)

    # Áp dụng bộ lọc dữ liệu
    filtered_df = df.copy()
    if selected_topic != "Tất cả":
        filtered_df = filtered_df[filtered_df["topic"] == selected_topic]
    if selected_sentiment != "Tất cả":
        filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]

    # --- PHẦN 1: THỐNG KÊ TỔNG QUAN (KPIs) ---
    total_count = len(filtered_df)
    pos_count = int((filtered_df["sentiment"] == "Tích cực").sum())
    neu_count = int((filtered_df["sentiment"] == "Bình thường").sum())
    neg_count = int((filtered_df["sentiment"] == "Tiêu cực").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📝 Tổng số bài viết", f"{total_count}")
    col2.metric("😊 Tích cực", f"{pos_count}", f"{(pos_count / total_count * 100) if total_count else 0:.1f}%")
    col3.metric("😐 Bình thường", f"{neu_count}", f"{(neu_count / total_count * 100) if total_count else 0:.1f}%")
    col4.metric("😟 Tiêu cực", f"{neg_count}", f"{(neg_count / total_count * 100) if total_count else 0:.1f}%")

    st.divider()

    # --- PHẦN 2: BIỂU ĐỒ THỐNG KÊ ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📊 Tỷ lệ phân bố cảm xúc")
        if not filtered_df.empty:
            pie_data = filtered_df["sentiment"].value_counts().reset_index()
            pie_data.columns = ["sentiment", "count"]
            fig_pie = px.pie(
                pie_data,
                names="sentiment",
                values="count",
                color="sentiment",
                color_discrete_map=COLOR_MAP,
                hole=0.42,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Không có dữ liệu phù hợp với bộ lọc.")

    with chart_col2:
        st.subheader("📈 Phân bố cảm xúc theo Chủ đề")
        if not filtered_df.empty:
            topic_data = (
                filtered_df.groupby(["topic", "sentiment"])
                .size()
                .reset_index(name="count")
            )
            fig_bar = px.bar(
                topic_data,
                x="topic",
                y="count",
                color="sentiment",
                barmode="group",
                color_discrete_map=COLOR_MAP,
                labels={"topic": "Chủ đề", "count": "Số bài viết", "sentiment": "Cảm xúc"},
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Không có dữ liệu phù hợp với bộ lọc.")

    st.divider()

    # --- PHẦN 3: Ô TEST TRỰC TIẾP MÔ HÌNH ---
    st.subheader("🧪 Ô Test Trực Tiếp (Live Sentiment Inference)")
    st.markdown(
        "Nhập bình luận hoặc bài viết của sinh viên (có thể chứa **teencode** như `mik`, `ko`, `đk`, `dkmh`, `csvc`, `gv`, `sv`...) để kiểm tra tiền xử lý và dự đoán cảm xúc:"
    )

    default_input = "CSVC trường mik xịn sò thực sự, nhưng cổng dkmh hôm nay hơi lag nha!"
    user_input = st.text_area(
        "Nội dung bài viết / bình luận sinh viên:",
        value=default_input,
        height=100,
    )

    if st.button("🚀 Phân tích cảm xúc", type="primary"):
        if not user_input.strip():
            st.warning("Vui lòng nhập nội dung văn bản cần phân tích.")
        else:
            result = analyzer.predict(user_input)
            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                st.markdown("##### 1. Kết quả tiền xử lý (`teencode.json` + `underthesea`)")
                st.code(result.preprocessed_text, language="text")

                badge_color = COLOR_MAP.get(result.label, "#3B82F6")
                st.markdown(
                    f"""
                    ##### 2. Nhãn dự đoán:
                    <div style="padding: 12px 18px; border-radius: 8px; background-color: {badge_color}22; border-left: 6px solid {badge_color};">
                        <span style="font-size: 1.25rem; font-weight: 700; color: {badge_color};">
                            {result.label}
                        </span>
                        <span style="margin-left: 12px; font-size: 0.95rem;">
                            (Độ tin cậy: <b>{result.confidence:.2%}</b>)
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with res_col2:
                st.markdown("##### Xác suất phân lớp")
                prob_df = pd.DataFrame(
                    {
                        "Nhãn cảm xúc": list(result.probabilities.keys()),
                        "Xác suất": list(result.probabilities.values()),
                    }
                )
                fig_prob = px.bar(
                    prob_df,
                    x="Xác suất",
                    y="Nhãn cảm xúc",
                    orientation="h",
                    color="Nhãn cảm xúc",
                    color_discrete_map=COLOR_MAP,
                    range_x=[0, 1],
                    text_auto=".1%",
                )
                fig_prob.update_layout(showlegend=False, height=220, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_prob, use_container_width=True)

    st.divider()

    # --- PHẦN 4: BẢNG DỮ LIỆU MẪU ---
    with st.expander("📂 Xem dữ liệu bài viết mẫu (`data/sample_posts.csv`)", expanded=True):
        st.dataframe(
            filtered_df[["id", "timestamp", "source", "topic", "content", "cleaned_content", "sentiment", "confidence"]],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()


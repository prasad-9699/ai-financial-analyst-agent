"""
AI Financial Analyst Agent — Main Streamlit Application.

A production-ready AI assistant that analyzes CSV data, answers questions
from PDF documents, generates interactive charts, and provides general
financial insights — all through natural language.
"""

from string import Template

import streamlit as st

from src.config import (
    AppConfig,
    ConfigurationError,
    FileValidationError,
    RAGProcessingError,
    load_config,
    setup_logging,
)
from src.llm import create_llm, invoke_llm_with_retry, chat_with_history
from src.csv_agent import analyze_csv
from src.pdf_rag import process_pdf, query_pdf
from src.charts import detect_chart_type, detect_columns, generate_chart
from src.router import Route, route_question
from src.utils import (
    format_dataframe_info,
    sanitize_user_input,
    validate_csv,
    validate_file_size,
)

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Financial Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════
# INITIALIZATION (before CSS to avoid tokenizer issues)
# ══════════════════════════════════════════════════════

try:
    config = load_config()
except ConfigurationError as e:
    st.error(f"⚙️ **Configuration Error**\n\n{e}")
    st.stop()

logger = setup_logging(config.log_level)


@st.cache_resource
def get_llm():
    """Cached LLM factory."""
    return create_llm(config)


try:
    llm = get_llm()
except Exception as e:
    st.error(f"🔌 **Could not connect to AI model**\n\n{e}")
    st.stop()

# ══════════════════════════════════════════════════════
# THEME STATE
# ══════════════════════════════════════════════════════
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

is_dark = st.session_state.theme == "dark"

# ══════════════════════════════════════════════════════
# THEME DEFINITIONS
# ══════════════════════════════════════════════════════
DARK_THEME = {
    "bg_main": "#0B0F19",
    "bg_sidebar": "#111827",
    "bg_card": "#1E293B",
    "bg_input": "#1E293B",
    "border": "#334155",
    "text_main": "#F8FAFC",
    "text_sec": "#CBD5E1",
    "text_muted": "#94A3B8",
    "accent": "#818CF8",
    "accent_grad": "linear-gradient(135deg, #6366F1, #8B5CF6)",
    "success": "#4ADE80",
    "chat_user": "#1E293B",
    "chat_asst": "#1A2332",
    "chat_user_b": "#6366F1",
    "chat_asst_b": "#4ADE80",
    "input_bdr": "#475569",
    "badge_bg": "rgba(99,102,241,0.15)",
    "badge_bdr": "rgba(99,102,241,0.3)",
    "hero_title": "#E0E7FF",
    "code_color": "#C4B5FD",
    "shadow": "0 4px 24px rgba(0,0,0,0.4)",
}

LIGHT_THEME = {
    "bg_main": "#F8FAFC",
    "bg_sidebar": "#FFFFFF",
    "bg_card": "#FFFFFF",
    "bg_input": "#FFFFFF",
    "border": "#E2E8F0",
    "text_main": "#0F172A",
    "text_sec": "#475569",
    "text_muted": "#94A3B8",
    "accent": "#6366F1",
    "accent_grad": "linear-gradient(135deg, #6366F1, #8B5CF6)",
    "success": "#16A34A",
    "chat_user": "#F1F5F9",
    "chat_asst": "#FFFFFF",
    "chat_user_b": "#6366F1",
    "chat_asst_b": "#16A34A",
    "input_bdr": "#CBD5E1",
    "badge_bg": "rgba(99,102,241,0.08)",
    "badge_bdr": "rgba(99,102,241,0.2)",
    "hero_title": "#1E1B4B",
    "code_color": "#7C3AED",
    "shadow": "0 4px 24px rgba(0,0,0,0.08)",
}

# ══════════════════════════════════════════════════════
# CSS TEMPLATE (uses string.Template with $ — safe with CSS)
# ══════════════════════════════════════════════════════
CSS_TEMPLATE = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* === GLOBAL === */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: $bg_main !important;
}
.stApp { background-color: $bg_main !important; }

/* Force text colors on every element */
.stApp, .stApp p, .stApp span, .stApp div, .stApp label,
.stApp li, .stApp td, .stApp th, .stApp h1, .stApp h2,
.stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp .stMarkdown, .stApp .stMarkdown p,
.stApp .stMarkdown span, .stApp .stMarkdown li,
.stApp .stMarkdown strong, .stApp .stMarkdown em,
.stApp figcaption, .stApp caption {
    color: $text_main !important;
    -webkit-text-fill-color: $text_main !important;
}
.stApp a { color: $accent !important; -webkit-text-fill-color: $accent !important; }
.stApp code { color: $code_color !important; -webkit-text-fill-color: $code_color !important; }
.stCaption, .stCaption span {
    color: $text_muted !important;
    -webkit-text-fill-color: $text_muted !important;
}

/* === HIDE DEFAULTS === */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* === SCROLLBAR === */
::-webkit-scrollbar { width: 7px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: $accent; border-radius: 4px; }

/* === SIDEBAR === */
section[data-testid="stSidebar"] {
    background-color: $bg_sidebar !important;
    border-right: 1px solid $border !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5 {
    color: $text_main !important;
    -webkit-text-fill-color: $text_main !important;
}

/* === HERO === */
.hero-wrap {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-chip {
    display: inline-block;
    background: $accent_grad;
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 6px 18px;
    border-radius: 50px;
    margin-bottom: 1rem;
}
.hero-heading {
    font-size: 2.5rem;
    font-weight: 800;
    color: $hero_title !important;
    -webkit-text-fill-color: $hero_title !important;
    margin: 0 0 0.6rem 0;
    line-height: 1.15;
}
.hero-sub {
    font-size: 1.05rem;
    color: $text_sec !important;
    -webkit-text-fill-color: $text_sec !important;
    max-width: 620px;
    margin: 0 auto;
    line-height: 1.7;
}

/* === BUTTONS === */
.stButton > button {
    background: $accent_grad !important;
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.35) !important;
}

/* === FILE UPLOADER === */
[data-testid="stFileUploader"] {
    border: 2px dashed $border !important;
    border-radius: 12px !important;
    transition: border-color 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: $accent !important;
}
/* Uploader inner text, labels, file names */
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploadDropzone"] span {
    color: $text_main !important;
    -webkit-text-fill-color: $text_main !important;
}
/* Browse files button */
[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploader"] button[kind="secondary"] {
    background-color: $bg_card !important;
    color: $text_main !important;
    -webkit-text-fill-color: $text_main !important;
    border: 1px solid $border !important;
    border-radius: 8px !important;
}
/* Uploaded file info row */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background-color: $bg_card !important;
    border: 1px solid $border !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] small {
    color: $text_sec !important;
    -webkit-text-fill-color: $text_sec !important;
}
/* Delete button on uploaded file */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button {
    color: $text_muted !important;
    -webkit-text-fill-color: $text_muted !important;
    background: transparent !important;
    border: none !important;
}
/* Drag-drop zone */
[data-testid="stFileUploadDropzone"] {
    background-color: $bg_card !important;
    border-radius: 10px !important;
}

/* === CHAT MESSAGES === */
[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    padding: 1rem 1.3rem !important;
    margin-bottom: 0.75rem !important;
    border: 1px solid $border !important;
    animation: msgIn 0.35s ease-out;
}
@keyframes msgIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: $chat_user !important;
    border-left: 3px solid $chat_user_b !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: $chat_asst !important;
    border-left: 3px solid $chat_asst_b !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em,
[data-testid="stChatMessage"] td,
[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] label,
[data-testid="stChatMessage"] .stMarkdown,
[data-testid="stChatMessage"] .stMarkdown p {
    color: $text_main !important;
    -webkit-text-fill-color: $text_main !important;
}

/* === CHAT INPUT === */
[data-testid="stChatInput"] textarea {
    background-color: $bg_input !important;
    border: 1.5px solid $input_bdr !important;
    border-radius: 12px !important;
    color: $text_main !important;
    -webkit-text-fill-color: $text_main !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: $accent !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: $text_muted !important;
    -webkit-text-fill-color: $text_muted !important;
}
[data-testid="stChatInput"] button {
    background: $accent_grad !important;
    border-radius: 10px !important;
}

/* === ALERTS === */
.stAlert { border-radius: 10px !important; }
.stAlert p, .stAlert span, .stAlert div {
    color: $text_main !important;
    -webkit-text-fill-color: $text_main !important;
}

/* === EXPANDER === */
[data-testid="stExpander"] {
    border: 1px solid $border !important;
    border-radius: 10px !important;
    background-color: $bg_card !important;
}
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
    color: $text_main !important;
    -webkit-text-fill-color: $text_main !important;
}

/* === STAT CARDS === */
.stats-row {
    display: flex;
    gap: 0.6rem;
    margin: 0.8rem 0;
}
.stat-box {
    flex: 1;
    background: $bg_card;
    border: 1px solid $border;
    border-radius: 10px;
    padding: 0.8rem 0.5rem;
    text-align: center;
    transition: all 0.2s ease;
}
.stat-box:hover {
    border-color: $accent;
    box-shadow: $shadow;
    transform: translateY(-1px);
}
.stat-num {
    font-size: 1.4rem;
    font-weight: 700;
    color: $accent !important;
    -webkit-text-fill-color: $accent !important;
}
.stat-lbl {
    font-size: 0.65rem;
    font-weight: 600;
    color: $text_muted !important;
    -webkit-text-fill-color: $text_muted !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 0.2rem;
}

/* === SOURCE BADGE === */
.src-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: $badge_bg;
    border: 1px solid $badge_bdr;
    border-radius: 50px;
    padding: 3px 12px;
    font-size: 0.7rem;
    font-weight: 600;
    color: $accent !important;
    -webkit-text-fill-color: $accent !important;
    margin-top: 0.4rem;
}
.src-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.src-dot.csv { background: #6366F1; }
.src-dot.pdf { background: #22C55E; }
.src-dot.chart { background: #EC4899; }
.src-dot.general { background: #F59E0B; }

/* === SIDEBAR CARD === */
.sb-card {
    background: $bg_card;
    border: 1px solid $border;
    border-radius: 12px;
    padding: 1.1rem 1rem;
    margin-bottom: 1rem;
}
.sb-card h3 {
    margin: 0 0 0.3rem 0;
    font-size: 1rem;
    font-weight: 700;
    color: $text_main !important;
    -webkit-text-fill-color: $text_main !important;
}
.sb-card p {
    margin: 0;
    font-size: 0.82rem;
    color: $text_sec !important;
    -webkit-text-fill-color: $text_sec !important;
}

/* === DIVIDER === */
hr {
    border: none !important;
    height: 1px !important;
    background: $border !important;
}

/* === DATAFRAME === */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
}

/* === BLOCKQUOTE === */
blockquote {
    border-left: 3px solid $accent !important;
    background: $bg_card !important;
    padding: 0.8rem 1rem !important;
    border-radius: 0 8px 8px 0 !important;
}
blockquote p {
    color: $text_sec !important;
    -webkit-text-fill-color: $text_sec !important;
}

/* === TOGGLE === */
.theme-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 0.3rem 0;
}
.theme-icon {
    font-size: 1.1rem;
}
</style>
""")

# Inject themed CSS
t = DARK_THEME if is_dark else LIGHT_THEME
st.markdown(CSS_TEMPLATE.substitute(t), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chart_counter" not in st.session_state:
    st.session_state.chart_counter = 0
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "csv_name" not in st.session_state:
    st.session_state.csv_name = None


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════

with st.sidebar:
    # ── Theme Toggle ────────────────────────────
    t_col1, t_col2, t_col3 = st.columns([1, 2, 1])
    with t_col1:
        st.markdown('<div class="theme-icon">☀️</div>', unsafe_allow_html=True)
    with t_col2:
        theme_toggle = st.toggle("Dark Mode", value=is_dark, key="theme_toggle")
        if theme_toggle != is_dark:
            st.session_state.theme = "dark" if theme_toggle else "light"
            st.rerun()
    with t_col3:
        st.markdown('<div class="theme-icon">🌙</div>', unsafe_allow_html=True)

    st.divider()

    # ── Header Card ─────────────────────────────
    st.markdown(
        '<div class="sb-card"><h3>📁 Data Sources</h3>'
        '<p>Upload CSV or PDF files to analyze with AI</p></div>',
        unsafe_allow_html=True,
    )

    # ── CSV Upload ──────────────────────────────
    st.markdown("##### 📄 CSV Data")
    csv_file = st.file_uploader(
        "Upload a CSV file for data analysis & charts",
        type=["csv"],
        key="csv_uploader",
        help="Max size: %d MB" % config.max_csv_size_mb,
    )

    if csv_file is not None:
        try:
            file_bytes = csv_file.getvalue()
            validate_file_size(file_bytes, config.max_csv_size_mb, csv_file.name)
            csv_file.seek(0)
            st.session_state.df = validate_csv(csv_file)
            st.session_state.csv_name = csv_file.name

            df = st.session_state.df
            num_cols = len(df.select_dtypes(include="number").columns)
            st.markdown(
                '<div class="stats-row">'
                '<div class="stat-box"><div class="stat-num">%s</div>'
                '<div class="stat-lbl">Rows</div></div>'
                '<div class="stat-box"><div class="stat-num">%d</div>'
                '<div class="stat-lbl">Columns</div></div>'
                '<div class="stat-box"><div class="stat-num">%d</div>'
                '<div class="stat-lbl">Numeric</div></div>'
                '</div>' % ("{:,}".format(df.shape[0]), df.shape[1], num_cols),
                unsafe_allow_html=True,
            )

            with st.expander("🔍 Preview Data", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

        except FileValidationError as e:
            st.error("❌ %s" % e)
            st.session_state.df = None

    # ── PDF Upload ──────────────────────────────
    st.markdown("##### 📑 PDF Document")
    pdf_file = st.file_uploader(
        "Upload any PDF to search & ask questions",
        type=["pdf"],
        key="pdf_uploader",
        help="Max size: %d MB. Works with any PDF." % config.max_pdf_size_mb,
    )

    if pdf_file is not None:
        if st.session_state.pdf_name != pdf_file.name:
            st.session_state.vectorstore = None
            st.session_state.pdf_name = pdf_file.name

        if st.session_state.vectorstore is None:
            try:
                file_bytes = pdf_file.getvalue()
                validate_file_size(file_bytes, config.max_pdf_size_mb, pdf_file.name)
                with st.spinner("🔄 Processing PDF... (first time may take 1-2 min)"):
                    st.session_state.vectorstore = process_pdf(file_bytes, config)
                st.success("✅ **%s** indexed and ready!" % pdf_file.name)
            except (FileValidationError, RAGProcessingError) as e:
                st.error("❌ %s" % e)
                st.session_state.vectorstore = None
        else:
            st.success("✅ **%s** — ready to query" % pdf_file.name)

    # ── Status ──────────────────────────────────
    st.divider()

    if st.session_state.df is None and st.session_state.vectorstore is None:
        st.markdown(
            "> 👋 **Getting Started**\n>\n"
            "> Upload a **CSV** for data analysis & charts, "
            "or a **PDF** to search its content.\n>\n"
            "> Or just type a question below for general AI insights!"
        )

    # ── Controls ────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chart_counter = 0
            st.rerun()
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            for key in ["messages", "df", "vectorstore", "chart_counter", "pdf_name", "csv_name"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    st.divider()
    st.caption("Powered by Groq · LangChain · FAISS")


# ══════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════

st.markdown(
    '<div class="hero-wrap">'
    '<div class="hero-chip">✨ AI-Powered Analysis</div>'
    '<h1 class="hero-heading">Financial Analyst Agent</h1>'
    '<p class="hero-sub">'
    "Upload your data, ask questions in plain English, and get instant insights, "
    "interactive charts, and answers powered by advanced AI."
    "</p></div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════
# CHAT INTERFACE
# ══════════════════════════════════════════════════════

def _source_badge(route):
    """Return an HTML source badge for the message."""
    labels = {
        Route.CSV: ("CSV Analysis", "csv"),
        Route.PDF: ("PDF Search", "pdf"),
        Route.CHART: ("Chart", "chart"),
        Route.GENERAL: ("General AI", "general"),
    }
    label, css_class = labels.get(route, ("AI", "general"))
    return '<div class="src-tag"><span class="src-dot %s"></span>%s</div>' % (css_class, label)


# Chart template matching theme
_chart_template = "plotly_dark" if is_dark else "plotly_white"

# Display past messages
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message.get("chart") is not None:
            st.plotly_chart(message["chart"], use_container_width=True, key="history_chart_%d" % idx)
        if message.get("content"):
            st.markdown(message["content"])
        if message.get("badge"):
            st.markdown(message["badge"], unsafe_allow_html=True)


# Chat input
if prompt := st.chat_input("Ask anything about your data..."):
    prompt = sanitize_user_input(prompt)

    if not prompt:
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": prompt, "chart": None, "badge": None}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("✨ Analyzing..."):
            route = route_question(
                llm,
                prompt,
                has_csv=st.session_state.df is not None,
                has_pdf=st.session_state.vectorstore is not None,
            )
            badge_html = _source_badge(route)

            try:
                if route == Route.CHART:
                    df = st.session_state.df
                    x_col, y_col = detect_columns(prompt, df)
                    chart_type = detect_chart_type(prompt)
                    fig = generate_chart(chart_type, x_col, y_col, df)
                    fig.update_layout(template=_chart_template)

                    st.session_state.chart_counter += 1
                    chart_key = "new_chart_%d" % st.session_state.chart_counter
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)

                    caption = "📊 %s chart — **%s** by **%s**" % (chart_type.title(), y_col, x_col)
                    st.markdown(caption)
                    st.markdown(badge_html, unsafe_allow_html=True)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": caption, "chart": fig, "badge": badge_html}
                    )

                elif route == Route.CSV:
                    answer = analyze_csv(prompt, st.session_state.df, llm)
                    st.markdown(answer)
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "chart": None, "badge": badge_html}
                    )

                elif route == Route.PDF:
                    answer = query_pdf(
                        prompt, st.session_state.vectorstore, llm, config.retriever_top_k,
                    )
                    st.markdown(answer)
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "chart": None, "badge": badge_html}
                    )

                else:
                    answer = chat_with_history(llm, prompt, st.session_state.messages)
                    st.markdown(answer)
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "chart": None, "badge": badge_html}
                    )

            except Exception as e:
                logger.error("Error handling question: %s", e, exc_info=True)
                error_msg = (
                    "⚠️ **Something went wrong**\n\n%s\n\n"
                    "💡 *Try rephrasing your question or check your internet connection.*"
                ) % str(e)
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "chart": None, "badge": None}
                )
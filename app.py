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
from src.supabase_client import (
    init_supabase,
    signup_user,
    login_user,
    logout_user,
    save_message,
    load_chat_history,
    clear_chat_history,
)
from src.utils import (
    format_dataframe_info,
    get_csv_quick_insights,
    sanitize_user_input,
    validate_csv,
    validate_file_size,
)

st.set_page_config(
    page_title="AI Financial Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Initialize Supabase client
supabase = init_supabase(config) if config.supabase_url and config.supabase_key else None


if "theme" not in st.session_state:
    st.session_state.theme = "light"

is_dark = st.session_state.theme == "dark"


DARK_THEME = {
    "page": "#060A12",
    "sidebar": "#0A0F1A",
    "surface": "#0F1628",
    "surface_2": "#131C30",
    "surface_3": "#182338",
    "border": "rgba(148, 163, 184, 0.10)",
    "border_strong": "rgba(148, 163, 184, 0.22)",
    "text": "#F1F5F9",
    "text_2": "#94A3B8",
    "muted": "#64748B",
    "accent": "#14B8A6",
    "accent_2": "#6366F1",
    "accent_3": "#F59E0B",
    "success": "#22C55E",
    "danger": "#F43F5E",
    "soft": "rgba(20, 184, 166, 0.10)",
    "soft_2": "rgba(99, 102, 241, 0.10)",
    "shadow": "0 20px 60px rgba(0, 0, 0, 0.35)",
    "input": "#0C1222",
    "code": "#67E8F9",
    "gradient": "linear-gradient(135deg, #14B8A6 0%, #6366F1 50%, #8B5CF6 100%)",
    "gradient_2": "linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #A78BFA 100%)",
    "glass": "rgba(15, 22, 40, 0.60)",
    "glass_border": "rgba(148, 163, 184, 0.12)",
}

LIGHT_THEME = {
    "page": "#F1F5F9",
    "sidebar": "#FFFFFF",
    "surface": "#FFFFFF",
    "surface_2": "#F8FAFC",
    "surface_3": "#E2E8F0",
    "border": "rgba(15, 23, 42, 0.08)",
    "border_strong": "rgba(15, 23, 42, 0.14)",
    "text": "#0F172A",
    "text_2": "#475569",
    "muted": "#94A3B8",
    "accent": "#0D9488",
    "accent_2": "#6366F1",
    "accent_3": "#D97706",
    "success": "#16A34A",
    "danger": "#E11D48",
    "soft": "rgba(13, 148, 136, 0.08)",
    "soft_2": "rgba(99, 102, 241, 0.06)",
    "shadow": "0 20px 50px rgba(15, 23, 42, 0.06)",
    "input": "#FFFFFF",
    "code": "#0D9488",
    "gradient": "linear-gradient(135deg, #0D9488 0%, #6366F1 50%, #8B5CF6 100%)",
    "gradient_2": "linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #A78BFA 100%)",
    "glass": "rgba(255, 255, 255, 0.65)",
    "glass_border": "rgba(15, 23, 42, 0.08)",
}


CSS_TEMPLATE = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ═══════ CSS Variables ═══════ */
:root {
    --page: $page;
    --sidebar: $sidebar;
    --surface: $surface;
    --surface-2: $surface_2;
    --surface-3: $surface_3;
    --border: $border;
    --border-strong: $border_strong;
    --text: $text;
    --text-2: $text_2;
    --muted: $muted;
    --accent: $accent;
    --accent-2: $accent_2;
    --accent-3: $accent_3;
    --success: $success;
    --danger: $danger;
    --soft: $soft;
    --soft-2: $soft_2;
    --shadow: $shadow;
    --input: $input;
    --code: $code;
    --gradient: $gradient;
    --gradient-2: $gradient_2;
    --glass: $glass;
    --glass-border: $glass_border;
}

/* ═══════ Keyframe Animations ═══════ */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 16px rgba(20, 184, 166, 0.25); }
    50%      { box-shadow: 0 0 30px rgba(99, 102, 241, 0.35); }
}

@keyframes pulseDot {
    0%, 100% { transform: scale(1); opacity: 1; }
    50%      { transform: scale(1.35); opacity: 0.7; }
}

@keyframes borderGlow {
    0%   { border-color: var(--accent); }
    50%  { border-color: var(--accent-2); }
    100% { border-color: var(--accent); }
}

@keyframes gradientMove {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%      { transform: translateY(-4px); }
}

/* ═══════ Base ═══════ */
html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--page) !important;
    color: var(--text) !important;
}

[data-testid="stAppViewContainer"] {
    background: var(--page) !important;
}

[data-testid="stAppViewBlockContainer"] {
    max-width: 1280px !important;
    padding: 1.5rem 2.5rem 7rem !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.stApp p, .stApp span, .stApp div, .stApp label,
.stApp li, .stApp td, .stApp th, .stApp h1, .stApp h2,
.stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    letter-spacing: -0.01em !important;
}

.stApp a {
    color: var(--accent) !important;
}

.stApp code {
    color: var(--code) !important;
    background: var(--soft) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.15rem 0.4rem !important;
    font-size: 0.85em !important;
}

/* ═══════ Scrollbar ═══════ */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-thumb {
    background: var(--surface-3);
    border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent);
}

/* ═══════ Sidebar ═══════ */
section[data-testid="stSidebar"] {
    background: var(--glass) !important;
    backdrop-filter: blur(24px) saturate(1.3) !important;
    -webkit-backdrop-filter: blur(24px) saturate(1.3) !important;
    border-right: 1px solid var(--glass-border) !important;
    box-shadow: 12px 0 50px rgba(0, 0, 0, 0.06) !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 1.2rem 1rem !important;
}

/* ═══════ Brand ═══════ */
.app-brand {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 0.35rem 0.15rem 1rem;
    animation: fadeInUp 0.5s ease both;
}

.brand-mark {
    width: 44px;
    height: 44px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
    background: var(--gradient);
    background-size: 200% 200%;
    animation: gradientMove 4s ease infinite, pulseGlow 3s ease infinite;
    font-weight: 900;
    font-size: 0.95rem;
    letter-spacing: 0 !important;
}

.brand-title {
    font-weight: 900;
    font-size: 1rem;
    line-height: 1.1;
    letter-spacing: -0.02em !important;
}

.brand-subtitle {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    font-size: 0.72rem;
    margin-top: 0.2rem;
    font-weight: 600;
}

/* ═══════ Sidebar Panels ═══════ */
.sidebar-panel {
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    background: var(--glass);
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    box-shadow: var(--shadow);
    padding: 1rem;
    margin: 0.85rem 0;
    animation: fadeInUp 0.6s ease both;
}

.sidebar-panel h3 {
    margin: 0 0 0.35rem;
    font-size: 0.88rem;
    font-weight: 800;
}

.sidebar-panel p {
    margin: 0;
    color: var(--text-2) !important;
    -webkit-text-fill-color: var(--text-2) !important;
    font-size: 0.76rem;
    line-height: 1.5;
}

/* ═══════ Theme Toggle ═══════ */
.theme-row {
    display: grid;
    grid-template-columns: 34px 1fr 34px;
    gap: 0.7rem;
    align-items: center;
    padding: 0.8rem 0;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
}

.theme-icon {
    width: 34px;
    height: 34px;
    display: grid;
    place-items: center;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--surface-2);
    transition: all 0.25s ease;
}

.theme-icon:hover {
    border-color: var(--accent);
    transform: scale(1.08);
}

/* ═══════ Upload ═══════ */
.upload-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    margin: 1rem 0 0.45rem;
    font-size: 0.78rem;
    font-weight: 800;
}

.upload-label span:last-child {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    font-size: 0.68rem;
    font-weight: 700;
}

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border-strong) !important;
    border-radius: 12px !important;
    padding: 0.35rem !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px var(--soft), var(--shadow);
    animation: borderGlow 2s ease infinite;
}

[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploadDropzone"] {
    background: var(--surface-2) !important;
    border: 0 !important;
    border-radius: 10px !important;
    min-height: 5.5rem !important;
}

[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploader"] button[kind="secondary"] {
    background: var(--surface) !important;
    color: var(--accent) !important;
    -webkit-text-fill-color: var(--accent) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 800 !important;
}

[data-testid="stFileUploaderFile"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ═══════ Stats Row ═══════ */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.45rem;
    margin: 0.8rem 0;
    animation: fadeInUp 0.5s ease both;
}

.stat-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 0.55rem;
    transition: all 0.25s ease;
}

.stat-box:hover {
    border-color: var(--accent);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(20, 184, 166, 0.10);
}

.stat-num {
    font-weight: 900;
    color: var(--accent) !important;
    -webkit-text-fill-color: var(--accent) !important;
    font-size: 1.05rem;
}

.stat-lbl {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    font-size: 0.65rem;
    font-weight: 750;
    margin-top: 0.18rem;
    text-transform: uppercase;
    letter-spacing: 0.04em !important;
}

/* ═══════ Insights Panel ═══════ */
.insights-panel {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem;
    margin: 0.6rem 0;
    animation: fadeInUp 0.6s ease both;
}

.insights-title {
    font-size: 0.72rem;
    font-weight: 850;
    color: var(--accent-2) !important;
    -webkit-text-fill-color: var(--accent-2) !important;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em !important;
}

.insight-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.72rem;
}

.insight-item:last-child {
    border-bottom: none;
}

.insight-col {
    font-weight: 700;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    max-width: 55%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.insight-val {
    font-weight: 800;
    color: var(--accent) !important;
    -webkit-text-fill-color: var(--accent) !important;
    font-variant-numeric: tabular-nums;
    font-size: 0.7rem;
}

/* ═══════ Empty State ═══════ */
.empty-state {
    background: var(--soft);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem;
    margin-top: 0.8rem;
    animation: fadeInUp 0.5s ease both;
}

.empty-title {
    color: var(--accent) !important;
    -webkit-text-fill-color: var(--accent) !important;
    font-weight: 850;
    font-size: 0.82rem;
    margin-bottom: 0.25rem;
}

.empty-state p {
    margin: 0;
    color: var(--text-2) !important;
    -webkit-text-fill-color: var(--text-2) !important;
    font-size: 0.78rem;
    line-height: 1.45;
}

/* ═══════ Hero / Top Bar ═══════ */
.shell-topbar {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1.2rem;
    animation: fadeInUp 0.5s ease both;
}

.page-kicker {
    color: var(--accent) !important;
    -webkit-text-fill-color: var(--accent) !important;
    font-size: 0.7rem;
    font-weight: 850;
    margin-bottom: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.08em !important;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.kicker-line {
    width: 28px;
    height: 2px;
    background: var(--gradient);
    border-radius: 2px;
}

.page-title {
    margin: 0;
    font-size: 2rem;
    line-height: 1.1;
    font-weight: 900;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    letter-spacing: -0.03em !important;
}

.page-title .title-gradient {
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent !important;
    background-clip: text;
    color: transparent !important;
}

.page-subtitle {
    color: var(--text-2) !important;
    -webkit-text-fill-color: var(--text-2) !important;
    font-size: 0.9rem;
    margin-top: 0.4rem;
    max-width: 680px;
    line-height: 1.55;
    font-weight: 450;
}

.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--glass);
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border);
    border-radius: 999px;
    padding: 0.5rem 0.85rem;
    box-shadow: var(--shadow);
    font-size: 0.74rem;
    font-weight: 800;
    white-space: nowrap;
    animation: float 3s ease infinite;
}

.live-dot {
    width: 8px;
    height: 8px;
    background: var(--success);
    border-radius: 50%;
    box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.14);
    animation: pulseDot 2s ease infinite;
}

/* ═══════ Source Cards ═══════ */
.source-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin-bottom: 1.2rem;
}

.source-card {
    background: var(--glass);
    backdrop-filter: blur(16px) saturate(1.2) !important;
    -webkit-backdrop-filter: blur(16px) saturate(1.2) !important;
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 1.1rem;
    box-shadow: var(--shadow);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeInUp 0.6s ease both;
    position: relative;
    overflow: hidden;
}

.source-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--gradient);
    background-size: 200% 100%;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.source-card:hover {
    transform: translateY(-4px);
    border-color: var(--accent);
    box-shadow: 0 24px 60px rgba(20, 184, 166, 0.12);
}

.source-card:hover::before {
    opacity: 1;
    animation: shimmer 2s linear infinite;
}

.source-card:nth-child(2) { animation-delay: 0.1s; }
.source-card:nth-child(3) { animation-delay: 0.2s; }

.source-card-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.7rem;
}

.source-icon {
    width: 38px;
    height: 38px;
    display: grid;
    place-items: center;
    border-radius: 10px;
    font-weight: 900;
    font-size: 0.72rem;
    letter-spacing: 0 !important;
    transition: all 0.3s ease;
}

.source-icon.icon-csv {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
    color: var(--accent-2) !important;
    -webkit-text-fill-color: var(--accent-2) !important;
}

.source-icon.icon-pdf {
    background: linear-gradient(135deg, rgba(20, 184, 166, 0.15) 0%, rgba(13, 148, 136, 0.15) 100%);
    color: var(--accent) !important;
    -webkit-text-fill-color: var(--accent) !important;
}

.source-icon.icon-ai {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.15) 100%);
    color: var(--accent-3) !important;
    -webkit-text-fill-color: var(--accent-3) !important;
}

.source-card:hover .source-icon {
    transform: scale(1.1) rotate(-3deg);
}

.source-status {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    font-size: 0.68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.03em !important;
}

.source-card h3 {
    margin: 0;
    font-size: 0.92rem;
    font-weight: 850;
    letter-spacing: -0.01em !important;
}

.source-card p {
    margin: 0.3rem 0 0;
    color: var(--text-2) !important;
    -webkit-text-fill-color: var(--text-2) !important;
    font-size: 0.76rem;
    line-height: 1.5;
}

/* ═══════ Chat Frame ═══════ */
.chat-frame {
    background: var(--glass);
    backdrop-filter: blur(16px) saturate(1.2) !important;
    -webkit-backdrop-filter: blur(16px) saturate(1.2) !important;
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    box-shadow: var(--shadow);
    padding: 1.1rem;
    margin-bottom: 1rem;
    animation: fadeInUp 0.6s ease both;
}

.chat-frame-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.85rem;
    margin-bottom: 1rem;
}

.chat-title {
    font-weight: 850;
    font-size: 1rem;
    letter-spacing: -0.01em !important;
}

.chat-meta {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    font-size: 0.74rem;
    font-weight: 650;
}

/* ═══════ Chat Messages ═══════ */
[data-testid="stChatMessage"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    box-shadow: none !important;
    padding: 1rem 1.15rem !important;
    margin-bottom: 0.8rem !important;
    animation: fadeInUp 0.4s ease both;
    transition: all 0.2s ease;
    border-left: 3px solid transparent !important;
}

[data-testid="stChatMessage"]:hover {
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06) !important;
}

[data-testid="stChatMessage"] p {
    font-size: 0.94rem !important;
    line-height: 1.7 !important;
}

[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
}

/* ═══════ Source Badge ═══════ */
.src-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: var(--soft);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.35rem 0.7rem;
    color: var(--accent) !important;
    -webkit-text-fill-color: var(--accent) !important;
    font-size: 0.7rem;
    font-weight: 850;
    margin-top: 0.5rem;
    transition: all 0.2s ease;
}

.src-tag:hover {
    transform: translateX(2px);
}

.src-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

.src-dot.csv   { background: var(--accent-2); box-shadow: 0 0 6px rgba(99, 102, 241, 0.4); }
.src-dot.pdf   { background: var(--accent);   box-shadow: 0 0 6px rgba(20, 184, 166, 0.4); }
.src-dot.chart { background: var(--danger);   box-shadow: 0 0 6px rgba(244, 63, 94, 0.4); }
.src-dot.general { background: var(--accent-3); box-shadow: 0 0 6px rgba(245, 158, 11, 0.4); }

/* ═══════ Status Badge ═══════ */
.status-online {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.18);
    border-radius: 999px;
    padding: 0.38rem 0.65rem;
    color: var(--success) !important;
    -webkit-text-fill-color: var(--success) !important;
    font-size: 0.72rem;
    font-weight: 850;
    margin-top: 0.55rem;
    animation: fadeInUp 0.4s ease both;
}

.status-dot {
    width: 7px;
    height: 7px;
    background: var(--success);
    border-radius: 50%;
    display: inline-block;
    animation: pulseDot 2s ease infinite;
}

/* ═══════ Chat Input ═══════ */
[data-testid="stChatInput"] {
    background: linear-gradient(180deg, transparent 0%, var(--page) 36%) !important;
    border-top: 1px solid var(--border) !important;
}

[data-testid="stChatInput"] textarea {
    background: var(--input) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 14px !important;
    min-height: 3.25rem !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    box-shadow: var(--shadow) !important;
    font-size: 0.93rem !important;
    transition: all 0.25s ease !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px var(--soft), var(--shadow) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
}

[data-testid="stChatInput"] button {
    background: var(--gradient) !important;
    border-radius: 12px !important;
    transition: all 0.2s ease;
}

[data-testid="stChatInput"] button:hover {
    transform: scale(1.06);
}

/* ═══════ Buttons ═══════ */
.stButton > button {
    background: var(--gradient) !important;
    background-size: 200% 200% !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
    font-weight: 850 !important;
    min-height: 2.5rem !important;
    font-size: 0.78rem !important;
    box-shadow: 0 10px 24px rgba(99, 102, 241, 0.18) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 0 !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 16px 32px rgba(99, 102, 241, 0.24) !important;
    animation: gradientMove 3s ease infinite;
}

/* ═══════ Expander, DataFrame, Plotly, Alerts ═══════ */
[data-testid="stExpander"],
[data-testid="stDataFrame"],
[data-testid="stPlotlyChart"],
.stAlert {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow) !important;
}

hr {
    border: 0 !important;
    height: 1px !important;
    background: var(--border) !important;
    margin: 1rem 0 !important;
}

/* ═══════ Suggestion Chips ═══════ */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.6rem 0 1rem;
    animation: fadeInUp 0.5s ease both;
}

.chip-row .stButton > button {
    background: var(--surface) !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    min-height: 2rem !important;
    padding: 0 0.85rem !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    box-shadow: none !important;
    transition: all 0.25s ease !important;
}

.chip-row .stButton > button:hover {
    background: var(--soft) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    -webkit-text-fill-color: var(--accent) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(20, 184, 166, 0.10) !important;
}

/* ═══════ Powered-by Footer ═══════ */
.powered-by {
    text-align: center;
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    font-size: 0.68rem;
    font-weight: 650;
    padding-top: 0.3rem;
}

/* ═══════ Responsive ═══════ */
@media (max-width: 900px) {
    [data-testid="stAppViewBlockContainer"] {
        padding: 1rem 1rem 7rem !important;
    }

    .shell-topbar {
        flex-direction: column;
    }

    .source-grid {
        grid-template-columns: 1fr;
    }

    .page-title {
        font-size: 1.5rem;
    }
}

/* ═══════ Form Inputs (Login / Signup / General) ═══════ */
.stTextInput > div > div > input,
.stTextInput input,
[data-testid="stTextInput"] input {
    background: var(--surface) !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 10px !important;
    padding: 0.6rem 0.75rem !important;
    font-size: 0.9rem !important;
    transition: all 0.25s ease !important;
    caret-color: var(--accent) !important;
}

.stTextInput > div > div > input:focus,
.stTextInput input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--soft), var(--shadow) !important;
    outline: none !important;
}

.stTextInput > div > div > input::placeholder,
.stTextInput input::placeholder,
[data-testid="stTextInput"] input::placeholder {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    opacity: 1 !important;
}

/* Labels */
.stTextInput > label,
[data-testid="stTextInput"] label,
.stSelectbox > label,
.stFileUploader > label {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    font-weight: 700 !important;
    font-size: 0.84rem !important;
}

/* Select boxes */
.stSelectbox > div > div,
[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 10px !important;
}

/* Toggle / Switch */
.stToggle label span,
[data-testid="stToggle"] label span {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}

/* Tab labels */
.stTabs [data-baseweb="tab"] {
    color: var(--text-2) !important;
    -webkit-text-fill-color: var(--text-2) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}

/* Form submit buttons */
.stFormSubmitButton > button,
[data-testid="stFormSubmitButton"] > button {
    background: var(--gradient) !important;
    background-size: 200% 200% !important;
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 850 !important;
    font-size: 0.85rem !important;
    min-height: 2.6rem !important;
    box-shadow: 0 10px 24px rgba(99, 102, 241, 0.18) !important;
    transition: all 0.3s ease !important;
}

.stFormSubmitButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 16px 32px rgba(99, 102, 241, 0.24) !important;
}

/* Download button */
.stDownloadButton > button {
    background: var(--surface) !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}

.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    -webkit-text-fill-color: var(--accent) !important;
    background: var(--soft) !important;
}

/* Password toggle eye icon */
.stTextInput button[kind="icon"],
[data-testid="stTextInput"] button {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
}

/* Expander */
[data-testid="stExpander"] summary span {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    font-weight: 700 !important;
}

/* Alert/Error/Success messages */
.stAlert p,
[data-testid="stAlert"] p {
    color: inherit !important;
    -webkit-text-fill-color: inherit !important;
}

</style>
""")



t = DARK_THEME if is_dark else LIGHT_THEME
st.markdown(CSS_TEMPLATE.substitute(t), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# AUTHENTICATION GATE
# ══════════════════════════════════════════════════════

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    # Auth-page styles
    st.markdown(
        '<style>'
        '.auth-brand{text-align:center;margin-bottom:1.5rem}'
        '.auth-brand-icon{width:60px;height:60px;display:inline-grid;place-items:center;'
        'border-radius:16px;background:var(--gradient);background-size:200% 200%;'
        'animation:gradientMove 4s ease infinite,pulseGlow 3s ease infinite;'
        'color:#fff!important;-webkit-text-fill-color:#fff!important;'
        'font-weight:900;font-size:1.4rem;margin-bottom:.8rem;'
        'box-shadow:0 16px 40px rgba(99,102,241,.22)}'
        '.auth-title{font-size:1.7rem;font-weight:900;margin:0;line-height:1.15;'
        'letter-spacing:-0.03em}'
        '.auth-subtitle{color:var(--text-2)!important;-webkit-text-fill-color:var(--text-2)!important;'
        'font-size:.88rem;margin-top:.4rem;font-weight:450}'
        '.stTabs [data-baseweb="tab-list"]{gap:0;justify-content:center}'
        '.stTabs [data-baseweb="tab"]{font-weight:700!important;font-size:.85rem!important}'
        '</style>',
        unsafe_allow_html=True,
    )

    _auth_l, _auth_c, _auth_r = st.columns([1, 2, 1])
    with _auth_c:
        st.markdown(
            '<div class="auth-brand">'
            '<div class="auth-brand-icon">FA</div>'
            '<div class="auth-title">Financial Analyst</div>'
            '<div class="auth-subtitle">Sign in to access your AI workspace</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        login_tab, signup_tab = st.tabs(["🔑 Log In", "✨ Sign Up"])

        with login_tab:
            with st.form("login_form"):
                login_email = st.text_input("Email", placeholder="you@example.com", key="login_email")
                login_password = st.text_input("Password", type="password", placeholder="Your password", key="login_password")
                login_submitted = st.form_submit_button("Sign In", use_container_width=True)

            if login_submitted:
                if not login_email or not login_password:
                    st.error("Please enter both email and password.")
                else:
                    with st.spinner("Signing in..."):
                        result = login_user(supabase, login_email, login_password)
                    if result["success"]:
                        st.session_state.user = result["user"]
                        st.rerun()
                    else:
                        st.error(result["error"])

        with signup_tab:
            with st.form("signup_form"):
                signup_email = st.text_input("Email", placeholder="you@example.com", key="signup_email")
                signup_password = st.text_input("Password", type="password", placeholder="Min 6 characters", key="signup_password")
                signup_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="signup_confirm")
                signup_submitted = st.form_submit_button("Create Account", use_container_width=True)

            if signup_submitted:
                if not signup_email or not signup_password:
                    st.error("Please fill in all fields.")
                elif signup_password != signup_confirm:
                    st.error("Passwords do not match.")
                elif len(signup_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    with st.spinner("Creating account..."):
                        result = signup_user(supabase, signup_email, signup_password)
                    if result["success"]:
                        st.success(result["message"])
                    else:
                        st.error(result["error"])

    st.stop()


# ══════════════════════════════════════════════════════
# SESSION STATE (authenticated user)
# ══════════════════════════════════════════════════════

user_id = st.session_state.user.id

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history(supabase, user_id)
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
if "voice_input" not in st.session_state:
    st.session_state.voice_input = ""


with st.sidebar:
    st.markdown(
        '<div class="app-brand">'
        '<div class="brand-mark">FA</div>'
        '<div><div class="brand-title">Financial Analyst</div>'
        '<div class="brand-subtitle">AI-powered workspace</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── User Info + Logout ───────────────────────
    _user_email = st.session_state.user.email or "User"
    _u_col1, _u_col2 = st.columns([3, 1])
    with _u_col1:
        st.markdown(
            '<div style="font-size:0.78rem;font-weight:700;color:var(--text-2,#4B5563);'
            'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:0.4rem 0;">'
            '👤 %s</div>' % _user_email,
            unsafe_allow_html=True,
        )
    with _u_col2:
        if st.button("↩", key="logout_btn", help="Sign out"):
            logout_user(supabase)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.markdown('<div class="theme-row">', unsafe_allow_html=True)
    t_col1, t_col2, t_col3 = st.columns([1, 2, 1])
    with t_col1:
        st.markdown('<div class="theme-icon">☀️</div>', unsafe_allow_html=True)
    with t_col2:
        theme_toggle = st.toggle("Dark", value=is_dark, key="theme_toggle")
        if theme_toggle != is_dark:
            st.session_state.theme = "dark" if theme_toggle else "light"
            st.rerun()
    with t_col3:
        st.markdown('<div class="theme-icon">🌙</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-panel">'
        '<h3>📂 Data Sources</h3>'
        '<p>Upload a CSV for analytics or a PDF for retrieval-based document answers.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="upload-label"><strong>CSV dataset</strong><span>Analysis + charts</span></div>',
        unsafe_allow_html=True,
    )
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
            null_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100) if df.shape[0] > 0 else 0
            st.markdown(
                '<div class="stats-row">'
                '<div class="stat-box"><div class="stat-num">%s</div><div class="stat-lbl">Rows</div></div>'
                '<div class="stat-box"><div class="stat-num">%d</div><div class="stat-lbl">Columns</div></div>'
                '<div class="stat-box"><div class="stat-num">%d</div><div class="stat-lbl">Numeric</div></div>'
                '</div>' % ("{:,}".format(df.shape[0]), df.shape[1], num_cols),
                unsafe_allow_html=True,
            )

            # ── Auto CSV Insights ────────────────────────
            insights = get_csv_quick_insights(df)
            if insights:
                items_html = ""
                for ins in insights:
                    items_html += (
                        '<div class="insight-item">'
                        '<span class="insight-col">%s</span>'
                        '<span class="insight-val">μ %s</span>'
                        '</div>' % (ins["col"], "{:,.0f}".format(ins["mean"]))
                    )
                st.markdown(
                    '<div class="insights-panel">'
                    '<div class="insights-title">⚡ Quick Insights</div>'
                    '%s'
                    '</div>' % items_html,
                    unsafe_allow_html=True,
                )

            with st.expander("Preview dataset", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

        except FileValidationError as e:
            st.error("❌ %s" % e)
            st.session_state.df = None

    st.markdown(
        '<div class="upload-label"><strong>PDF document</strong><span>Search + Q&A</span></div>',
        unsafe_allow_html=True,
    )
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
                with st.spinner("Indexing PDF..."):
                    st.session_state.vectorstore = process_pdf(file_bytes, config)
                st.markdown(
                    '<div class="status-online"><span class="status-dot"></span>'
                    '<span>%s ready</span></div>' % pdf_file.name,
                    unsafe_allow_html=True,
                )
            except (FileValidationError, RAGProcessingError) as e:
                st.error("❌ %s" % e)
                st.session_state.vectorstore = None
        else:
            st.markdown(
                '<div class="status-online"><span class="status-dot"></span>'
                '<span>%s ready</span></div>' % pdf_file.name,
                unsafe_allow_html=True,
            )

    if st.session_state.df is None and st.session_state.vectorstore is None:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-title">No source attached</div>'
            '<p>You can still ask general finance questions, or upload files for grounded answers.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear", use_container_width=True):
            clear_chat_history(supabase, user_id)
            st.session_state.messages = []
            st.session_state.chart_counter = 0
            st.rerun()
    with col2:
        if st.button("Reset", use_container_width=True):
            clear_chat_history(supabase, user_id)
            for key in ["messages", "df", "vectorstore", "chart_counter", "pdf_name", "csv_name"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # ── Export Chat ──────────────────────────────
    if st.session_state.get("messages"):
        st.divider()
        chat_export = "# Financial Analyst — Chat Export\n\n"
        for msg in st.session_state.messages:
            role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
            content = msg.get("content", "")
            if content:
                chat_export += f"### {role}\n{content}\n\n---\n\n"
        st.download_button(
            "📥 Export Chat",
            data=chat_export,
            file_name="financial_analyst_chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()
    st.markdown(
        '<div class="powered-by">Groq · LangChain · FAISS · Supabase</div>',
        unsafe_allow_html=True,
    )


csv_status = "Connected" if st.session_state.df is not None else "Not attached"
pdf_status = "Indexed" if st.session_state.vectorstore is not None else "Not attached"
message_count = len(st.session_state.messages)

st.markdown(
    '<div class="shell-topbar">'
    '<div>'
    '<div class="page-kicker"><span class="kicker-line"></span>AI FINANCIAL ANALYSIS CONSOLE</div>'
    '<h1 class="page-title">Ask, analyze, and <span class="title-gradient">visualize</span> financial data.</h1>'
    '<div class="page-subtitle">A focused workspace for CSV analysis, PDF search, chart generation, and general financial reasoning — powered by AI.</div>'
    '</div>'
    '<div class="live-pill"><span class="live-dot"></span>Model online</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="source-grid">'
    '<div class="source-card">'
    '<div class="source-card-top"><div class="source-icon icon-csv">CSV</div><div class="source-status">%s</div></div>'
    '<h3>Dataset Analysis</h3><p>Ask questions about rows, columns, trends, summaries, and comparisons.</p>'
    '</div>'
    '<div class="source-card">'
    '<div class="source-card-top"><div class="source-icon icon-pdf">PDF</div><div class="source-status">%s</div></div>'
    '<h3>Document Intelligence</h3><p>Search uploaded PDFs and answer from retrieved document context.</p>'
    '</div>'
    '<div class="source-card">'
    '<div class="source-card-top"><div class="source-icon icon-ai">AI</div><div class="source-status">%d messages</div></div>'
    '<h3>Analyst Chat</h3><p>Use plain English to request insights, explanations, and charts.</p>'
    '</div>'
    '</div>' % (csv_status, pdf_status, message_count),
    unsafe_allow_html=True,
)

# ── Suggestion Chips ──────────────────────────────
if st.session_state.df is not None and message_count == 0:
    st.markdown('<div class="chip-row">', unsafe_allow_html=True)
    chip_cols = st.columns(4)
    _chip_prompts = [
        "📊 Show total revenue",
        "📈 Plot monthly trend",
        "📋 Describe the dataset",
        "🏆 Top 5 by profit",
    ]
    for i, label in enumerate(_chip_prompts):
        with chip_cols[i]:
            if st.button(label, key=f"chip_{i}"):
                st.session_state.voice_input = label.split(" ", 1)[1] if " " in label else label
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="chat-frame">'
    '<div class="chat-frame-header">'
    '<div><div class="chat-title">💬 Conversation</div><div class="chat-meta">Answers are routed automatically to CSV, PDF, chart, or general reasoning.</div></div>'
    '<div class="chat-meta">%d total</div>'
    '</div>'
    '</div>' % message_count,
    unsafe_allow_html=True,
)


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


_chart_template = "plotly_dark" if is_dark else "plotly_white"

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message.get("chart") is not None:
            st.plotly_chart(message["chart"], use_container_width=True, key="history_chart_%d" % idx)
        if message.get("content"):
            st.markdown(message["content"])
        if message.get("badge"):
            st.markdown(message["badge"], unsafe_allow_html=True)

# ======================================================
# VOICE INPUT (optional — works if Streamlit >= 1.33)
# ======================================================

_voice_prompt = ""

try:
    from groq import Groq as GroqClient

    audio_bytes = st.audio_input(
        "🎙️ Record a voice question",
        key="voice_recorder",
        help="Click the mic, speak your question, then click stop.",
    )

    if audio_bytes is not None:
        # Use hash to detect new recordings
        audio_data = audio_bytes.read()
        audio_hash = hash(audio_data)

        if audio_hash != st.session_state.get("_last_audio_hash"):
            st.session_state["_last_audio_hash"] = audio_hash
            with st.spinner("🎤 Transcribing your voice..."):
                try:
                    groq_client = GroqClient(api_key=config.groq_api_key)
                    transcription = groq_client.audio.transcriptions.create(
                        file=("recording.wav", audio_data),
                        model="whisper-large-v3",
                        response_format="text",
                        language="en",
                        temperature=0.0,
                    )
                    voice_text = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
                    if voice_text:
                        st.session_state.voice_input = voice_text
                        st.rerun()
                except Exception as e:
                    logger.error("Voice transcription error: %s", e, exc_info=True)
                    st.warning("Could not transcribe audio: %s" % str(e))
except Exception:
    # st.audio_input not available in this Streamlit version — skip silently
    pass

# Process voice input if available
if st.session_state.get("voice_input"):
    _voice_prompt = st.session_state.voice_input
    st.session_state.voice_input = ""


# ======================================================
# CHAT INPUT
# ======================================================

if prompt := (_voice_prompt or st.chat_input("Ask about your data — sum, average, trends, charts, or any question...")):
    prompt = sanitize_user_input(prompt)

    if not prompt:
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": prompt, "chart": None, "badge": None}
    )
    if supabase:
        save_message(supabase, user_id, "user", prompt)

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
                    fig.update_layout(
                        template=_chart_template,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin={"l": 24, "r": 24, "t": 48, "b": 32},
                    )

                    st.session_state.chart_counter += 1
                    chart_key = "new_chart_%d" % st.session_state.chart_counter
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)

                    caption = "📊 %s chart — **%s** by **%s**" % (chart_type.title(), y_col, x_col)
                    st.markdown(caption)
                    st.markdown(badge_html, unsafe_allow_html=True)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": caption, "chart": fig, "badge": badge_html}
                    )
                    if supabase:
                        save_message(supabase, user_id, "assistant", caption, badge_html, "chart")

                elif route == Route.CSV:
                    answer = analyze_csv(prompt, st.session_state.df, llm)
                    st.markdown(answer)
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "chart": None, "badge": badge_html}
                    )
                    if supabase:
                        save_message(supabase, user_id, "assistant", answer, badge_html, "csv")

                elif route == Route.PDF:
                    answer = query_pdf(
                        prompt, st.session_state.vectorstore, llm, config.retriever_top_k,
                    )
                    st.markdown(answer)
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "chart": None, "badge": badge_html}
                    )
                    if supabase:
                        save_message(supabase, user_id, "assistant", answer, badge_html, "pdf")

                else:
                    answer = chat_with_history(llm, prompt, st.session_state.messages)
                    st.markdown(answer)
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "chart": None, "badge": badge_html}
                    )
                    if supabase:
                        save_message(supabase, user_id, "assistant", answer, badge_html, "general")

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
                if supabase:
                    save_message(supabase, user_id, "assistant", error_msg)
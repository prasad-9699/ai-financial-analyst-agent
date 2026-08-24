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
    "page": "#080A0F",
    "sidebar": "#0D111A",
    "surface": "#111827",
    "surface_2": "#151D2B",
    "surface_3": "#1B2433",
    "border": "rgba(148, 163, 184, 0.16)",
    "border_strong": "rgba(148, 163, 184, 0.28)",
    "text": "#F3F4F6",
    "text_2": "#B6C0D1",
    "muted": "#7C879A",
    "accent": "#14B8A6",
    "accent_2": "#2563EB",
    "accent_3": "#F59E0B",
    "success": "#22C55E",
    "danger": "#F43F5E",
    "soft": "rgba(20, 184, 166, 0.12)",
    "soft_2": "rgba(37, 99, 235, 0.12)",
    "shadow": "0 18px 50px rgba(0, 0, 0, 0.28)",
    "input": "#0F172A",
    "code": "#67E8F9",
    "gradient": "linear-gradient(135deg, #14B8A6 0%, #2563EB 100%)",
}

LIGHT_THEME = {
    "page": "#F4F7FB",
    "sidebar": "#FFFFFF",
    "surface": "#FFFFFF",
    "surface_2": "#F8FAFC",
    "surface_3": "#EEF3F9",
    "border": "rgba(15, 23, 42, 0.10)",
    "border_strong": "rgba(15, 23, 42, 0.16)",
    "text": "#111827",
    "text_2": "#4B5563",
    "muted": "#8792A5",
    "accent": "#0F766E",
    "accent_2": "#2563EB",
    "accent_3": "#B45309",
    "success": "#16A34A",
    "danger": "#E11D48",
    "soft": "rgba(15, 118, 110, 0.10)",
    "soft_2": "rgba(37, 99, 235, 0.08)",
    "shadow": "0 18px 44px rgba(15, 23, 42, 0.08)",
    "input": "#FFFFFF",
    "code": "#0F766E",
    "gradient": "linear-gradient(135deg, #0F766E 0%, #2563EB 100%)",
}


CSS_TEMPLATE = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

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
}

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--page) !important;
    color: var(--text) !important;
}

[data-testid="stAppViewContainer"] {
    background: var(--page) !important;
}

[data-testid="stAppViewBlockContainer"] {
    max-width: 1240px !important;
    padding: 1.25rem 2rem 7rem !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.stApp p, .stApp span, .stApp div, .stApp label,
.stApp li, .stApp td, .stApp th, .stApp h1, .stApp h2,
.stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    letter-spacing: 0 !important;
}

.stApp a {
    color: var(--accent) !important;
}

.stApp code {
    color: var(--code) !important;
    background: var(--soft) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.12rem 0.35rem !important;
}

section[data-testid="stSidebar"] {
    background: var(--sidebar) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 12px 0 40px rgba(15, 23, 42, 0.05) !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 1.1rem 1rem !important;
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-thumb {
    background: var(--surface-3);
    border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent);
}

.app-brand {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 0.35rem 0.15rem 1rem;
}

.brand-mark {
    width: 42px;
    height: 42px;
    display: grid;
    place-items: center;
    border-radius: 10px;
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
    background: var(--gradient);
    font-weight: 900;
    box-shadow: 0 12px 28px rgba(37, 99, 235, 0.18);
}

.brand-title {
    font-weight: 850;
    font-size: 0.98rem;
    line-height: 1.1;
}

.brand-subtitle {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    font-size: 0.74rem;
    margin-top: 0.2rem;
}

.sidebar-panel {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    box-shadow: var(--shadow);
    padding: 1rem;
    margin: 0.85rem 0;
}

.sidebar-panel h3 {
    margin: 0 0 0.35rem;
    font-size: 0.92rem;
    font-weight: 800;
}

.sidebar-panel p {
    margin: 0;
    color: var(--text-2) !important;
    -webkit-text-fill-color: var(--text-2) !important;
    font-size: 0.78rem;
    line-height: 1.45;
}

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
    border-radius: 9px;
    border: 1px solid var(--border);
    background: var(--surface-2);
}

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
    border: 1px dashed var(--border-strong) !important;
    border-radius: 10px !important;
    padding: 0.35rem !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--soft);
}

[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploadDropzone"] {
    background: var(--surface-2) !important;
    border: 0 !important;
    border-radius: 9px !important;
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

.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.45rem;
    margin: 0.8rem 0;
}

.stat-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 0.75rem 0.55rem;
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
    font-size: 0.66rem;
    font-weight: 750;
    margin-top: 0.18rem;
}

.empty-state {
    background: var(--soft);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.85rem;
    margin-top: 0.8rem;
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

.shell-topbar {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1rem;
}

.page-kicker {
    color: var(--accent) !important;
    -webkit-text-fill-color: var(--accent) !important;
    font-size: 0.73rem;
    font-weight: 850;
    margin-bottom: 0.25rem;
}

.page-title {
    margin: 0;
    font-size: 1.85rem;
    line-height: 1.12;
    font-weight: 900;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}

.page-subtitle {
    color: var(--text-2) !important;
    -webkit-text-fill-color: var(--text-2) !important;
    font-size: 0.92rem;
    margin-top: 0.35rem;
    max-width: 680px;
    line-height: 1.5;
}

.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.48rem 0.7rem;
    box-shadow: var(--shadow);
    font-size: 0.76rem;
    font-weight: 800;
    white-space: nowrap;
}

.live-dot {
    width: 8px;
    height: 8px;
    background: var(--success);
    border-radius: 50%;
    box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.14);
}

.source-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.8rem;
    margin-bottom: 1.1rem;
}

.source-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    box-shadow: var(--shadow);
}

.source-card-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.65rem;
}

.source-icon {
    width: 34px;
    height: 34px;
    display: grid;
    place-items: center;
    border-radius: 9px;
    background: var(--soft);
    color: var(--accent) !important;
    -webkit-text-fill-color: var(--accent) !important;
    font-weight: 900;
}

.source-status {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    font-size: 0.7rem;
    font-weight: 800;
}

.source-card h3 {
    margin: 0;
    font-size: 0.95rem;
    font-weight: 850;
}

.source-card p {
    margin: 0.3rem 0 0;
    color: var(--text-2) !important;
    -webkit-text-fill-color: var(--text-2) !important;
    font-size: 0.78rem;
    line-height: 1.45;
}

.chat-frame {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: var(--shadow);
    padding: 1rem;
    margin-bottom: 1rem;
}

.chat-frame-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.8rem;
    margin-bottom: 0.95rem;
}

.chat-title {
    font-weight: 850;
    font-size: 0.98rem;
}

.chat-meta {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    font-size: 0.76rem;
    font-weight: 750;
}

[data-testid="stChatMessage"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    padding: 1rem 1.1rem !important;
    margin-bottom: 0.75rem !important;
}

[data-testid="stChatMessage"] p {
    font-size: 0.96rem !important;
    line-height: 1.65 !important;
}

[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
}

.src-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: var(--soft);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.35rem 0.65rem;
    color: var(--accent) !important;
    -webkit-text-fill-color: var(--accent) !important;
    font-size: 0.7rem;
    font-weight: 850;
    margin-top: 0.5rem;
}

.src-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

.src-dot.csv { background: var(--accent-2); }
.src-dot.pdf { background: var(--accent); }
.src-dot.chart { background: var(--danger); }
.src-dot.general { background: var(--accent-3); }

.status-online {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(34, 197, 94, 0.10);
    border: 1px solid rgba(34, 197, 94, 0.22);
    border-radius: 999px;
    padding: 0.35rem 0.6rem;
    color: var(--success) !important;
    -webkit-text-fill-color: var(--success) !important;
    font-size: 0.72rem;
    font-weight: 850;
    margin-top: 0.55rem;
}

.status-dot {
    width: 7px;
    height: 7px;
    background: var(--success);
    border-radius: 50%;
    display: inline-block;
}

[data-testid="stChatInput"] {
    background: linear-gradient(180deg, transparent 0%, var(--page) 36%) !important;
    border-top: 1px solid var(--border) !important;
}

[data-testid="stChatInput"] textarea {
    background: var(--input) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 12px !important;
    min-height: 3.25rem !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    box-shadow: var(--shadow) !important;
    font-size: 0.94rem !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--soft), var(--shadow) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
}

[data-testid="stChatInput"] button {
    background: var(--gradient) !important;
    border-radius: 10px !important;
}

.stButton > button {
    background: var(--gradient) !important;
    border: none !important;
    border-radius: 9px !important;
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
    font-weight: 850 !important;
    min-height: 2.45rem !important;
    font-size: 0.78rem !important;
    box-shadow: 0 12px 24px rgba(37, 99, 235, 0.16) !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 30px rgba(37, 99, 235, 0.22) !important;
}

[data-testid="stExpander"],
[data-testid="stDataFrame"],
[data-testid="stPlotlyChart"],
.stAlert {
    border-radius: 10px !important;
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

.powered-by {
    text-align: center;
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    font-size: 0.7rem;
    font-weight: 650;
}

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
        font-size: 1.45rem;
    }
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
        '.auth-brand-icon{width:56px;height:56px;display:inline-grid;place-items:center;'
        'border-radius:14px;background:var(--gradient);color:#fff!important;'
        '-webkit-text-fill-color:#fff!important;font-weight:900;font-size:1.3rem;'
        'margin-bottom:.75rem;box-shadow:0 12px 28px rgba(37,99,235,.18)}'
        '.auth-title{font-size:1.6rem;font-weight:900;margin:0;line-height:1.2}'
        '.auth-subtitle{color:var(--text-2)!important;-webkit-text-fill-color:var(--text-2)!important;'
        'font-size:.88rem;margin-top:.35rem}'
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
        '<div class="brand-subtitle">Production AI workspace</div></div>'
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
        '<h3>Data Sources</h3>'
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
            st.markdown(
                '<div class="stats-row">'
                '<div class="stat-box"><div class="stat-num">%s</div><div class="stat-lbl">Rows</div></div>'
                '<div class="stat-box"><div class="stat-num">%d</div><div class="stat-lbl">Columns</div></div>'
                '<div class="stat-box"><div class="stat-num">%d</div><div class="stat-lbl">Numeric</div></div>'
                '</div>' % ("{:,}".format(df.shape[0]), df.shape[1], num_cols),
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
    '<div class="page-kicker">AI FINANCIAL ANALYSIS CONSOLE</div>'
    '<h1 class="page-title">Ask, analyze, and visualize financial data.</h1>'
    '<div class="page-subtitle">A focused workspace for CSV analysis, PDF search, chart generation, and general financial reasoning.</div>'
    '</div>'
    '<div class="live-pill"><span class="live-dot"></span>Model online</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="source-grid">'
    '<div class="source-card">'
    '<div class="source-card-top"><div class="source-icon">CSV</div><div class="source-status">%s</div></div>'
    '<h3>Dataset Analysis</h3><p>Ask questions about rows, columns, trends, summaries, and comparisons.</p>'
    '</div>'
    '<div class="source-card">'
    '<div class="source-card-top"><div class="source-icon">PDF</div><div class="source-status">%s</div></div>'
    '<h3>Document Intelligence</h3><p>Search uploaded PDFs and answer from retrieved document context.</p>'
    '</div>'
    '<div class="source-card">'
    '<div class="source-card-top"><div class="source-icon">AI</div><div class="source-status">%d messages</div></div>'
    '<h3>Analyst Chat</h3><p>Use plain English to request insights, explanations, and charts.</p>'
    '</div>'
    '</div>' % (csv_status, pdf_status, message_count),
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="chat-frame">'
    '<div class="chat-frame-header">'
    '<div><div class="chat-title">Conversation</div><div class="chat-meta">Answers are routed automatically to CSV, PDF, chart, or general reasoning.</div></div>'
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
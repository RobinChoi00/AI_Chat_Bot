import streamlit as st
import pandas as pd
import sqlite3
import os
import json
from datetime import datetime, timedelta
import extra_streamlit_components as stx
from dotenv import load_dotenv

# 1. Page Configuration
st.set_page_config(
    page_title="Titan AI - Admin Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

cookie_manager = stx.CookieManager(key="admin_cookie_manager")

load_dotenv(override=True)

try:
    creds_str = os.environ.get("ADMIN_CREDENTIALS", "{}")
    VALID_CREDENTIALS = json.loads(creds_str)
except json.JSONDecodeError:
    VALID_CREDENTIALS = {}
    st.error("🚨 [Security Configuration Error] Invalid credentials format in .env file.")

# ==========================================
# Global Custom CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ─── Base ────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
}

/* ─── Hide default Streamlit chrome ───── */
#MainMenu, footer, header {visibility: hidden;}
div[data-testid="stDecoration"] {display: none;}

/* ─── Login Card ─────────────────────── */
.login-wrapper {
    display: flex; justify-content: center; align-items: center;
    min-height: 80vh;
}
.login-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 48px 40px 36px;
    max-width: 400px; width: 100%;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.login-card h2 {
    color: #ffffff; text-align: center; margin-bottom: 8px;
    font-weight: 700; font-size: 1.6rem;
}
.login-card p {
    color: rgba(255,255,255,0.5); text-align: center;
    margin-bottom: 28px; font-size: 0.9rem;
}

/* ─── Text input styling ─────────────── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6C63FF !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.25) !important;
}
.stTextInput label {
    color: rgba(255,255,255,0.7) !important;
    font-weight: 500 !important;
}

/* ─── Primary button ─────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF, #4ECDC4) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(108,99,255,0.4) !important;
}

/* ─── KPI metric cards ───────────────── */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    transition: transform 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(108,99,255,0.3);
}
div[data-testid="stMetric"] label {
    color: rgba(255,255,255,0.6) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
}

/* ─── Section headers ────────────────── */
h1, h2, h3 {
    color: #ffffff !important;
}
h1 { font-weight: 700 !important; }
p, span, label { color: rgba(255,255,255,0.75); }

/* ─── Tabs ───────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: rgba(255,255,255,0.5) !important;
    font-weight: 500;
    padding: 10px 20px;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: rgba(108,99,255,0.25) !important;
    color: #ffffff !important;
    font-weight: 600;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ─── Dataframe styling ──────────────── */
div[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    overflow: hidden;
}

/* ─── Selectbox ──────────────────────── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}
.stSelectbox label {
    color: rgba(255,255,255,0.7) !important;
    font-weight: 500 !important;
}

/* ─── Date input ─────────────────────── */
.stDateInput > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
}
.stDateInput label {
    color: rgba(255,255,255,0.7) !important;
    font-weight: 500 !important;
}

/* ─── Expander ───────────────────────── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.8) !important;
    font-weight: 500 !important;
}

/* ─── Divider ────────────────────────── */
hr {
    border-color: rgba(255,255,255,0.08) !important;
}

/* ─── Scrollbar ──────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.15);
    border-radius: 3px;
}

/* ─── Brand badge colors ─────────────── */
.brand-titan { color: #4ECDC4; font-weight: 600; }
.brand-osakiusa { color: #FF6B6B; font-weight: 600; }
.brand-osakimassage { color: #FFD93D; font-weight: 600; }
.brand-unknown { color: rgba(255,255,255,0.4); font-weight: 500; }

/* ─── Chat bubble styling ────────────── */
.chat-bubble-user {
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 14px 14px 4px 14px;
    padding: 14px 18px;
    margin-bottom: 8px;
    color: #e0e0ff;
    font-size: 0.92rem;
    line-height: 1.5;
}
.chat-bubble-bot {
    background: rgba(78,205,196,0.1);
    border: 1px solid rgba(78,205,196,0.2);
    border-radius: 14px 14px 14px 4px;
    padding: 14px 18px;
    margin-bottom: 16px;
    color: #d0f0ed;
    font-size: 0.92rem;
    line-height: 1.5;
    max-height: 300px;
    overflow-y: auto;
}
.chat-label {
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.chat-label-user { color: rgba(108,99,255,0.7); }
.chat-label-bot { color: rgba(78,205,196,0.7); }
.chat-time {
    font-size: 0.73rem;
    color: rgba(255,255,255,0.3);
    margin-bottom: 12px;
}

/* ─── Logout button override ─────────── */
.logout-btn button {
    background: rgba(255,75,75,0.15) !important;
    color: #ff6b6b !important;
    border: 1px solid rgba(255,75,75,0.25) !important;
    font-size: 0.82rem !important;
    padding: 8px 16px !important;
}
.logout-btn button:hover {
    background: rgba(255,75,75,0.25) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ─── Stat card with icon ────────────── */
.stat-icon {
    font-size: 2rem;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# Login logic (Enter key + cookie persistence)
# ==========================================
def check_login():
    if cookie_manager.get(cookie="admin_auth_token") == "authenticated":
        st.session_state["logged_in"] = True
        return True

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        st.markdown('<div class="login-wrapper">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1.2, 1])
        with col2:
            st.markdown("""
            <div class="login-card">
                <h2>🔒 Titan AI</h2>
                <p>Admin Dashboard Login</p>
            </div>
            """, unsafe_allow_html=True)

            with st.form("login_form", clear_on_submit=False):
                input_id = st.text_input("User ID", placeholder="Enter your ID")
                input_pw = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)

                if submitted:
                    if input_id in VALID_CREDENTIALS and VALID_CREDENTIALS[input_id] == input_pw:
                        st.session_state["logged_in"] = True
                        cookie_manager.set("admin_auth_token", "authenticated", max_age=86400)
                        st.rerun()
                    else:
                        st.error("Invalid User ID or Password.")

        st.markdown('</div>', unsafe_allow_html=True)
        return False
    return True


if not check_login():
    st.stop()


# ==========================================
# Logout
# ==========================================
header_left, header_right = st.columns([8, 2])
with header_left:
    st.markdown("## 📊 Titan AI — Intelligence Dashboard")
    st.caption("Real-time AI Chatbot Monitoring for Management & Marketing Teams")
with header_right:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            cookie_manager.delete("admin_auth_token")
            st.session_state["logged_in"] = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# Database Connection & Data Loading
# ==========================================
DB_PATH = os.path.join(os.getcwd(), "db_data", "chat_history.db")

DOMAIN_LABELS = {
    "All Sites": None,
    "Titan Chair": "titanchair",
    "Osaki USA": "osakiusa",
    "Osaki Massage Chair": "osakimassagechair",
}

BRAND_COLORS = {
    "Titan Chair": "#4ECDC4",
    "Osaki USA": "#FF6B6B",
    "Osaki Massage Chair": "#FFD93D",
    "Unknown": "rgba(255,255,255,0.4)",
}


@st.cache_data(ttl=60)
def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM chat_logs ORDER BY created_at DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = df['created_at'].dt.date
        if 'domain' not in df.columns:
            df['domain'] = 'unknown'
        df['brand'] = df['domain'].apply(_resolve_brand_label)
    return df


def _resolve_brand_label(domain: str) -> str:
    d = (domain or "").lower()
    if "titanchair" in d:
        return "Titan Chair"
    if "osakiusa" in d or "osaki-usa" in d:
        return "Osaki USA"
    if "osakimassage" in d:
        return "Osaki Massage Chair"
    return "Unknown"


# ==========================================
# Load data
# ==========================================
df = load_data()

if df.empty:
    st.warning("No chat records found in the database yet. Start a conversation in the chatbot first!")
    st.stop()


# ==========================================
# Top Controls: Filter + Refresh
# ==========================================
st.markdown("---")

ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
with ctrl1:
    selected_label = st.selectbox("🌐 Filter by Brand", list(DOMAIN_LABELS.keys()))
with ctrl2:
    date_range = st.date_input(
        "📅 Date Range",
        value=(df['date'].min(), df['date'].max()),
        min_value=df['date'].min(),
        max_value=df['date'].max(),
    )
with ctrl3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

keyword = DOMAIN_LABELS[selected_label]
if keyword:
    filtered = df[df['domain'].str.contains(keyword, case=False, na=False)]
else:
    filtered = df

if isinstance(date_range, tuple) and len(date_range) == 2:
    filtered = filtered[
        (filtered['date'] >= date_range[0]) & (filtered['date'] <= date_range[1])
    ]


# ==========================================
# KPI Metrics
# ==========================================
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

total_chats = len(filtered)
unique_users = filtered['session_id'].nunique() if not filtered.empty else 0
today_chats = len(filtered[filtered['date'] == datetime.today().date()]) if not filtered.empty else 0

yesterday = datetime.today().date() - timedelta(days=1)
yesterday_chats = len(filtered[filtered['date'] == yesterday]) if not filtered.empty else 0
delta_today = today_chats - yesterday_chats

with col1:
    col1.metric("💬 Total Conversations", f"{total_chats:,}")
with col2:
    col2.metric("👥 Unique Sessions", f"{unique_users:,}")
with col3:
    col3.metric("🔥 Today's Chats", f"{today_chats:,}", delta=f"{delta_today:+d} vs yesterday")
with col4:
    if keyword is None and 'brand' in df.columns and not df.empty:
        brand_counts = filtered['brand'].value_counts()
        col4.metric("🏷️ Top Brand", brand_counts.index[0] if len(brand_counts) > 0 else "N/A")
    else:
        col4.metric("🏷️ Current Filter", selected_label)


# ==========================================
# Main Content Tabs
# ==========================================
st.markdown("---")

tab_overview, tab_by_brand, tab_logs, tab_conversations = st.tabs([
    "📈 Overview",
    "🏢 By Brand",
    "📋 Chat Logs",
    "💬 Conversations"
])


# ─── Tab 1: Overview ─────────────────────
with tab_overview:
    st.markdown("### Daily Chat Traffic")
    daily_counts = filtered.groupby('date').size().reset_index(name='counts')
    st.bar_chart(data=daily_counts, x='date', y='counts', use_container_width=True)

    if keyword is None and 'brand' in filtered.columns and not filtered.empty:
        st.markdown("### Brand Distribution")
        brand_daily = filtered.groupby(['date', 'brand']).size().reset_index(name='counts')
        brand_pivot = brand_daily.pivot(index='date', columns='brand', values='counts').fillna(0)
        st.area_chart(brand_pivot, use_container_width=True)


# ─── Tab 2: By Brand ─────────────────────
with tab_by_brand:
    if 'brand' not in filtered.columns or filtered.empty:
        st.info("No brand data available.")
    else:
        brand_list = sorted(filtered['brand'].unique())
        for brand_name in brand_list:
            brand_color = BRAND_COLORS.get(brand_name, "#ffffff")
            brand_df = filtered[filtered['brand'] == brand_name]

            with st.expander(f"**{brand_name}** — {len(brand_df):,} conversations", expanded=True):
                b1, b2, b3 = st.columns(3)
                b1.metric("Conversations", f"{len(brand_df):,}")
                b2.metric("Sessions", f"{brand_df['session_id'].nunique():,}")
                brand_today = len(brand_df[brand_df['date'] == datetime.today().date()])
                b3.metric("Today", f"{brand_today:,}")

                brand_daily = brand_df.groupby('date').size().reset_index(name='counts')
                st.bar_chart(data=brand_daily, x='date', y='counts', use_container_width=True)

                recent_logs = brand_df.head(5)[['created_at', 'session_id', 'user_query', 'bot_response']]
                st.markdown("**Recent Conversations**")
                st.dataframe(recent_logs, use_container_width=True, hide_index=True)

                brand_export_cols = ['created_at', 'session_id', 'user_query', 'bot_response']
                brand_export_df = brand_df[[c for c in brand_export_cols if c in brand_df.columns]]
                brand_csv = brand_export_df.to_csv(index=False).encode('utf-8-sig')
                brand_tag = brand_name.replace(" ", "_").lower()
                st.download_button(
                    label=f"⬇️ Download {brand_name} CSV",
                    data=brand_csv,
                    file_name=f"chat_logs_{brand_tag}_{datetime.today().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key=f"dl_{brand_tag}",
                )


# ─── Tab 3: Chat Logs (table view) ───────
with tab_logs:
    st.markdown("### 🕵️ Raw Chat Logs")

    log_brand_filter = st.selectbox(
        "Filter logs by brand",
        ["All"] + sorted(filtered['brand'].unique().tolist()) if not filtered.empty else ["All"],
        key="log_brand_filter"
    )

    if log_brand_filter != "All":
        log_filtered = filtered[filtered['brand'] == log_brand_filter]
    else:
        log_filtered = filtered

    search_query = st.text_input("🔍 Search in messages", placeholder="Type keyword to search...", key="search_logs")
    if search_query:
        mask = (
            log_filtered['user_query'].str.contains(search_query, case=False, na=False)
            | log_filtered['bot_response'].str.contains(search_query, case=False, na=False)
        )
        log_filtered = log_filtered[mask]

    dl_col1, dl_col2 = st.columns([3, 1])
    with dl_col1:
        st.caption(f"Showing {len(log_filtered):,} records")
    with dl_col2:
        display_cols = ['created_at', 'brand', 'session_id', 'user_query', 'bot_response']
        export_df = log_filtered[[c for c in display_cols if c in log_filtered.columns]]
        csv_data = export_df.to_csv(index=False).encode('utf-8-sig')
        brand_tag = log_brand_filter.replace(" ", "_").lower() if log_brand_filter != "All" else "all_brands"
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name=f"chat_logs_{brand_tag}_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    display_df = export_df
    st.dataframe(display_df, use_container_width=True, height=500, hide_index=True)


# ─── Tab 4: Conversation view (chat bubbles) ─
with tab_conversations:
    st.markdown("### 💬 Conversation Viewer")

    conv_brand_filter = st.selectbox(
        "Filter by brand",
        ["All"] + sorted(filtered['brand'].unique().tolist()) if not filtered.empty else ["All"],
        key="conv_brand_filter"
    )

    if conv_brand_filter != "All":
        conv_filtered = filtered[filtered['brand'] == conv_brand_filter]
    else:
        conv_filtered = filtered

    session_ids = conv_filtered['session_id'].unique().tolist()

    if not session_ids:
        st.info("No conversations to display.")
    else:
        selected_session = st.selectbox(
            "Select Session",
            session_ids,
            format_func=lambda sid: f"Session: {sid[:16]}..." if len(str(sid)) > 16 else f"Session: {sid}",
            key="session_select"
        )

        if selected_session:
            session_data = conv_filtered[conv_filtered['session_id'] == selected_session].sort_values('created_at')
            brand_name = session_data['brand'].iloc[0] if 'brand' in session_data.columns else "Unknown"
            brand_color = BRAND_COLORS.get(brand_name, "#ffffff")

            st.markdown(
                f'<span style="display:inline-block;background:{brand_color}22;color:{brand_color};'
                f'padding:4px 12px;border-radius:20px;font-size:0.82rem;font-weight:600;'
                f'border:1px solid {brand_color}44;margin-bottom:16px;">{brand_name}</span>',
                unsafe_allow_html=True
            )
            st.caption(f"{len(session_data)} messages · Started {session_data['created_at'].iloc[0].strftime('%Y-%m-%d %H:%M')}")

            for _, row in session_data.iterrows():
                ts = row['created_at'].strftime('%H:%M:%S') if pd.notna(row['created_at']) else ""

                user_q = str(row.get('user_query', '')).replace('<', '&lt;').replace('>', '&gt;')
                bot_r = str(row.get('bot_response', '')).replace('<', '&lt;').replace('>', '&gt;')

                st.markdown(f"""
                <div class="chat-label chat-label-user">👤 Customer</div>
                <div class="chat-bubble-user">{user_q}</div>
                <div class="chat-label chat-label-bot">🤖 AI Agent</div>
                <div class="chat-bubble-bot">{bot_r}</div>
                <div class="chat-time">{ts}</div>
                """, unsafe_allow_html=True)

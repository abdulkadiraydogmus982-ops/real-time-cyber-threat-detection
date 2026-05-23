import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import sqlite3
import plotly.graph_objects as go
import os
from datetime import datetime

# ── 1. VERİTABANI ──────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('siber_olaylar.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS saldirilar
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  zaman TEXT, tur TEXT, guven REAL, gercek_etiket TEXT)''')
    conn.commit()
    conn.close()

def save_to_db(tur, guven, gercek):
    try:
        conn = sqlite3.connect('siber_olaylar.db')
        c = conn.cursor()
        zaman = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO saldirilar (zaman, tur, guven, gercek_etiket) VALUES (?,?,?,?)",
                  (zaman, tur, guven, gercek))
        conn.commit()
        conn.close()
    except:
        pass

@st.cache_data
def load_rag_knowledge():
    if os.path.exists('cybersecurity_attacks.csv'):
        return pd.read_csv('cybersecurity_attacks.csv')
    return None

# ── 2. RAG ─────────────────────────────────────────────────────────────────────
def get_rag_insight(attack_label, df_ref):
    if df_ref is None:
        return "🔍 Bilgi bankası (CSV) bağlı değil."
    match = df_ref[df_ref['Attack Type'].str.contains(attack_label, case=False, na=False)]
    if not match.empty:
        v = match.sample(1).iloc[0]
        return (f"<span style='color:#00ff88'>▶ RAG:</span> {v['Attack Signature']} imzası saptandı."
                f"&nbsp;&nbsp;<span style='color:#ffcc00'>⚑ Öneri:</span> {v['Action Taken']}")
    return "<span style='color:#ff6600'>⚠ Yeni nesil tehdit — veritabanında eşleşme yok.</span>"

# ── 3. FEATURES PARSER ─────────────────────────────────────────────────────────
def parse_features(f_raw):
    if isinstance(f_raw, (list, tuple)):
        if len(f_raw) > 0 and isinstance(f_raw[0], (list, tuple, np.ndarray)):
            return list(f_raw[0])
        return list(f_raw)
    elif isinstance(f_raw, np.ndarray):
        return f_raw.flatten().tolist()
    elif isinstance(f_raw, (pd.Index, pd.Series)):
        return f_raw.tolist()
    elif isinstance(f_raw, str):
        return None
    else:
        try:
            return list(f_raw)
        except Exception:
            return None

# ── 4. VARLIK YÜKLEME ──────────────────────────────────────────────────────────
@st.cache_resource
def load_all_assets():
    init_db()
    try:
        m_raw = joblib.load('Siber_Kalkan_Final_V3.joblib')
        model = m_raw[0] if isinstance(m_raw, (list, tuple)) else m_raw

        le_raw = joblib.load('final_v3_le.joblib')
        le = le_raw[0] if isinstance(le_raw, (list, tuple)) else le_raw

        f_raw    = joblib.load('final_v3_features.joblib')
        features = parse_features(f_raw)

        if not features:
            st.error("⚠️ 'final_v3_features.joblib' geçersiz. Eğitimde: joblib.dump(list(X_train.columns), ...)")
            return None, None, None

        return model, le, features
    except FileNotFoundError as e:
        st.error(f"⚠️ Dosya bulunamadı: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"⚠️ Yükleme hatası: {e}")
        return None, None, None

# ── 5. SAYFA AYARI & GLOBAL CSS ────────────────────────────────────────────────
st.set_page_config(page_title="Siber Zeka V16", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace !important;
    background-color: #020c02 !important;
    color: #00ff41 !important;
}
.stApp { background-color: #020c02 !important; }

/* Başlık */
.soc-header {
    background: linear-gradient(90deg, #001a00 0%, #003300 50%, #001a00 100%);
    border: 1px solid #00ff41;
    border-left: 4px solid #00ff41;
    padding: 18px 28px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.soc-header::before {
    content: '';
    position: absolute; top: 0; left: -100%;
    width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,255,65,0.06), transparent);
    animation: scan 3s linear infinite;
}
@keyframes scan { to { left: 150%; } }
.soc-title {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.6rem; font-weight: 900;
    color: #00ff41; letter-spacing: 4px;
    text-shadow: 0 0 20px #00ff41, 0 0 40px #00aa2a;
    margin: 0;
}
.soc-sub {
    font-size: 0.7rem; color: #006622;
    letter-spacing: 6px; margin-top: 4px;
}

/* Metrik kartları */
.metric-card {
    background: #001a00;
    border: 1px solid #005500;
    border-top: 2px solid #00ff41;
    padding: 14px 18px;
    text-align: center;
    position: relative;
}
.metric-label {
    font-size: 0.65rem; color: #006622;
    letter-spacing: 3px; text-transform: uppercase;
}
.metric-value {
    font-family: 'Orbitron', monospace !important;
    font-size: 2rem; font-weight: 700;
    color: #00ff41;
    text-shadow: 0 0 12px #00ff41;
}
.metric-value.danger { color: #ff3333; text-shadow: 0 0 12px #ff3333; }
.metric-value.warn   { color: #ffcc00; text-shadow: 0 0 12px #ffcc00; }

/* Panel */
.panel {
    background: #010d01;
    border: 1px solid #003300;
    border-top: 2px solid #00aa2a;
    padding: 0;
    margin-bottom: 12px;
}
.panel-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem; letter-spacing: 4px;
    color: #006622; background: #001500;
    padding: 6px 14px; border-bottom: 1px solid #003300;
}

/* Log */
.log-container {
    background: #000d00;
    border: 1px solid #002200;
    padding: 10px;
    height: 320px;
    overflow-y: auto;
    font-size: 0.72rem;
    line-height: 1.8;
}
.log-benign  { color: #006622; }
.log-threat  { color: #ff4444; }
.log-ts      { color: #004d00; }

/* RAG */
.rag-box {
    background: #000d00;
    border: 1px solid #003300;
    border-left: 3px solid #00ff41;
    padding: 10px 16px;
    font-size: 0.78rem;
    min-height: 44px;
    margin-top: 8px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #010a01 !important;
    border-right: 1px solid #003300 !important;
}
section[data-testid="stSidebar"] * { color: #00cc33 !important; }

/* Butonlar */
.stButton > button {
    background: #001a00 !important;
    border: 1px solid #00ff41 !important;
    color: #00ff41 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    padding: 8px 18px !important;
    width: 100% !important;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #003300 !important;
    box-shadow: 0 0 12px #00ff41 !important;
}

/* Slider, selectbox */
.stSlider > div > div { background: #003300 !important; }
div[data-baseweb="select"] > div {
    background: #001500 !important;
    border-color: #005500 !important;
}

/* İlerleme çubuğu */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #003300, #00ff41) !important;
}

/* Divider */
hr { border-color: #003300 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #001000; }
::-webkit-scrollbar-thumb { background: #005500; border-radius: 2px; }

/* Uploader */
[data-testid="stFileUploader"] {
    background: #001500 !important;
    border: 1px dashed #005500 !important;
    border-radius: 0 !important;
}

/* Info/warning kutuları */
.stAlert { border-radius: 0 !important; background: #001500 !important; }
</style>
""", unsafe_allow_html=True)

# ── 6. VARLIK YÜKLEMESİ ────────────────────────────────────────────────────────
model, le, features = load_all_assets()
df_ref = load_rag_knowledge()

# ── 7. SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:12px 0 8px;'>
        <div style='font-family:Orbitron,monospace; font-size:1rem;
                    color:#00ff41; letter-spacing:4px;
                    text-shadow:0 0 10px #00ff41;'>SOC</div>
        <div style='font-size:0.6rem; color:#004d00; letter-spacing:6px;'>COMMAND CENTER</div>
    </div>
    <hr style='border-color:#003300; margin:8px 0 16px;'>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂  VERİ SETİ (.parquet)", type=['parquet'])

    st.markdown("<div style='font-size:0.65rem; letter-spacing:3px; color:#005500; margin:16px 0 4px;'>OKUMA LİMİTİ</div>", unsafe_allow_html=True)
    limit_mode = st.radio(
        "Mod",
        ["Tümünü Oku", "Satır Sayısı Gir", "Yüzde Belirle"],
        label_visibility="collapsed"
    )

    satir_limiti = None
    if limit_mode == "Satır Sayısı Gir":
        satir_limiti = st.number_input(
            "Kaç satır?", min_value=1, max_value=1_000_000,
            value=1000, step=100, format="%d"
        )
    elif limit_mode == "Yüzde Belirle":
        yuzde = st.slider("Veri yüzdesi (%)", min_value=1, max_value=100, value=20, step=1)
        st.caption(f"Dosya yüklendikten sonra hesaplanır.")

    st.markdown("<div style='font-size:0.65rem; letter-spacing:3px; color:#005500; margin:16px 0 4px;'>İŞLEM HIZI</div>", unsafe_allow_html=True)
    hiz = st.select_slider(
        "Hız",
        options=[1.0, 0.5, 0.25, 0.1, 0.05, 0.01],
        value=0.1,
        format_func=lambda x: {1.0:"▌ Çok Yavaş", 0.5:"▌▌ Yavaş",
                                0.25:"▌▌▌ Orta", 0.1:"▌▌▌▌ Hızlı",
                                0.05:"▌▌▌▌▌ Çok Hızlı", 0.01:"⚡ Maksimum"}[x],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#003300; margin:16px 0;'>", unsafe_allow_html=True)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("▶  BAŞLAT"):
            st.session_state.active = True
    with col_b2:
        if st.button("■  DURDUR"):
            st.session_state.active = False

    st.markdown("<hr style='border-color:#003300; margin:16px 0;'>", unsafe_allow_html=True)

    # Durum göstergesi
    durum = "● AKTİF" if st.session_state.get("active") else "○ BEKLEMEDE"
    renk  = "#00ff41" if st.session_state.get("active") else "#444"
    st.markdown(f"<div style='font-family:Orbitron,monospace; font-size:0.7rem; color:{renk}; text-align:center; letter-spacing:3px;'>{durum}</div>", unsafe_allow_html=True)

    # Tanılama
    with st.expander("🔧 Tanılama"):
        if features:
            st.success(f"✅ {len(features)} özellik yüklendi")
            st.write(features[:8])
        else:
            st.error("❌ Features yüklenemedi")
        st.write(f"Model: {'✅' if model else '❌'}")
        st.write(f"RAG CSV: {'✅' if df_ref is not None else '❌'}")

# Session state
if 'active' not in st.session_state:
    st.session_state.active = False

# ── 8. BAŞLIK ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class='soc-header'>
    <div class='soc-title'>🛰 SİBER ZEKA V16</div>
    <div class='soc-sub'>RAG INTEGRATED · REAL-TIME SOC PLATFORM · AI-POWERED THREAT DETECTION</div>
</div>
""", unsafe_allow_html=True)

# ── 9. ANA DÖNGÜ ───────────────────────────────────────────────────────────────
if model and features and uploaded_file:
    df_full = pd.read_parquet(uploaded_file)

    # Sütun eşleştirme
    mapping = {
        'Total Fwd Packets':       'Tot Fwd Pkts',
        'Total Backward Packets':  'Tot Bwd Pkts',
        'Fwd Packet Length Max':   'Fwd Pkt Len Max',
        'Fwd Packet Length Std':   'Fwd Pkt Len Std',
        'Bwd Packet Length Std':   'Bwd Pkt Len Std',
        'Flow IAT Mean':           'Flow Iat Mean',
        'Flow IAT Max':            'Flow Iat Max',
        'Fwd PSH Flags':           'PSH Flag Cnt',
        'Init Win bytes forward':  'Init Fwd Win Byts',
        'Init Win bytes backward': 'Init Bwd Win Byts',
        'Fwd Segment Size Avg':    'Fwd Seg Size Min',
    }
    df_full = df_full.rename(columns=mapping)

    # Satır limiti uygula
    total_rows = len(df_full)
    if limit_mode == "Satır Sayısı Gir" and satir_limiti:
        df = df_full.head(int(satir_limiti))
    elif limit_mode == "Yüzde Belirle":
        n = max(1, int(total_rows * yuzde / 100))
        df = df_full.head(n)
    else:
        df = df_full

    islenecek = len(df)

    # Bilgi bandı
    st.markdown(
        f"<div style='background:#001500; border:1px solid #003300; padding:8px 14px; "
        f"font-size:0.72rem; color:#006622; margin-bottom:14px;'>"
        f"📊 Toplam: <b style='color:#00cc33'>{total_rows:,}</b> satır &nbsp;|&nbsp; "
        f"İşlenecek: <b style='color:#00ff41'>{islenecek:,}</b> satır &nbsp;|&nbsp; "
        f"Hız: <b style='color:#00ff41'>{hiz}s/paket</b>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Metrik kartları
    mc1, mc2, mc3, mc4 = st.columns(4)
    ph_pkt   = mc1.empty()
    ph_thr   = mc2.empty()
    ph_acc   = mc3.empty()
    ph_prog  = mc4.empty()

    def render_metric(ph, label, value, cls=""):
        ph.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value {cls}'>{value}</div>"
            f"</div>", unsafe_allow_html=True
        )

    render_metric(ph_pkt,  "İŞLENEN",  "—")
    render_metric(ph_thr,  "TEHDİT",   "—", "danger")
    render_metric(ph_acc,  "DOĞRULUK", "—", "warn")
    render_metric(ph_prog, "İLERLEME", "—")

    # İlerleme çubuğu
    progress_bar = st.progress(0)

    # RAG kutusu
    rag_placeholder = st.empty()
    rag_placeholder.markdown("<div class='rag-box'>⬛ RAG motoru bekliyor...</div>", unsafe_allow_html=True)

    # Alt panel: harita + log
    col_map, col_log = st.columns([3, 2])
    with col_map:
        st.markdown("<div class='panel-title'>◈ KÜRESEL TEHDİT HARİTASI</div>", unsafe_allow_html=True)
        map_placeholder = st.empty()
    with col_log:
        st.markdown("<div class='panel-title'>◈ CANLI OLAY AKIŞI</div>", unsafe_allow_html=True)
        log_placeholder = st.empty()

    # ── Analiz döngüsü ────────────────────────────────────────────────────────
    if st.session_state.active:
        threats, correct, logs = 0, 0, []
        map_pts = pd.DataFrame(columns=['lat', 'lon'])

        for i in range(islenecek):
            if not st.session_state.active:
                break

            row     = df.iloc[[i]]
            X_input = (row.reindex(columns=features)
                          .fillna(0)
                          .apply(pd.to_numeric, errors='coerce')
                          .fillna(0))
            X_mat   = X_input.values.reshape(1, -1)

            probs   = model.predict_proba(X_mat)
            p_idx   = int(np.argmax(probs))
            conf    = float(np.max(probs)) * 100
            actual  = str(row['Label'].values[0]) if 'Label' in row.columns else "N/A"

            try:
                label = le.inverse_transform([p_idx])[0] if hasattr(le, 'inverse_transform') else f"Sınıf_{p_idx}"
            except Exception:
                label = f"Sınıf_{p_idx}"

            is_threat = label.lower() != "benign"
            is_right  = label.lower() in actual.lower() or actual.lower() in label.lower()

            if is_threat:
                threats += 1
                insight = get_rag_insight(label, df_ref)
                rag_placeholder.markdown(
                    f"<div class='rag-box'>{insight}</div>",
                    unsafe_allow_html=True
                )
                save_to_db(label, conf, actual)
                map_pts = pd.concat([map_pts, pd.DataFrame({
                    'lat': [np.random.uniform(-50, 65)],
                    'lon': [np.random.uniform(-120, 140)]
                })], ignore_index=True)

            if is_right:
                correct += 1

            # Metrikler
            acc_val  = (correct / (i + 1)) * 100
            prog_val = (i + 1) / islenecek * 100
            render_metric(ph_pkt,  "İŞLENEN",  f"{i+1:,}")
            render_metric(ph_thr,  "TEHDİT",   str(threats), "danger")
            render_metric(ph_acc,  "DOĞRULUK", f"%{acc_val:.1f}", "warn")
            render_metric(ph_prog, "İLERLEME", f"%{prog_val:.0f}")
            progress_bar.progress((i + 1) / islenecek)

            # Harita
            if i % 20 == 0 and not map_pts.empty:
                fig = go.Figure()
                fig.add_trace(go.Scattergeo(
                    lat=map_pts['lat'], lon=map_pts['lon'],
                    mode='markers',
                    marker=dict(
                        size=8, color='#ff3333',
                        symbol='circle',
                        line=dict(color='#ff8888', width=1)
                    )
                ))
                fig.update_layout(
                    geo=dict(
                        projection_type='natural earth',
                        showland=True, landcolor="#0a1a0a",
                        showocean=True, oceancolor="#020c02",
                        showframe=False,
                        showcountries=True, countrycolor="#003300",
                        bgcolor="rgba(0,0,0,0)"
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=340
                )
                map_placeholder.plotly_chart(fig, use_container_width=True, key=f"map_{i}")

            # Log
            ts        = datetime.now().strftime("%H:%M:%S")
            log_class = "log-threat" if is_threat else "log-benign"
            icon      = "⚠" if is_threat else "✓"
            logs.insert(0,
                f"<span class='log-ts'>[{ts}]</span> "
                f"<span class='{log_class}'>{icon} #{i+1:04d} {label}</span>"
                f"<span style='color:#004400'> → {actual}</span>"
                f"<span style='color:#003300'> {conf:.0f}%</span>"
            )
            log_placeholder.markdown(
                f"<div class='log-container'>{'<br>'.join(logs[:30])}</div>",
                unsafe_allow_html=True
            )

            time.sleep(hiz)

        # Tamamlandı
        if st.session_state.active:
            st.session_state.active = False
            st.markdown(
                f"<div style='background:#001a00; border:1px solid #00ff41; padding:12px 18px; "
                f"font-family:Orbitron,monospace; color:#00ff41; letter-spacing:2px; text-align:center;'>"
                f"✅ ANALİZ TAMAMLANDI — {islenecek:,} paket işlendi · "
                f"{threats} tehdit · %{(correct/islenecek*100):.1f} doğruluk"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        # Harita boş görünüm
        fig = go.Figure()
        fig.update_layout(
            geo=dict(
                projection_type='natural earth',
                showland=True, landcolor="#0a1a0a",
                showocean=True, oceancolor="#020c02",
                showcountries=True, countrycolor="#003300",
                bgcolor="rgba(0,0,0,0)", showframe=False
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0), height=340
        )
        map_placeholder.plotly_chart(fig, use_container_width=True, key="map_idle")
        log_placeholder.markdown(
            "<div class='log-container'><span style='color:#004400'>▌ Sistem hazır. BAŞLAT butonuna basın...</span></div>",
            unsafe_allow_html=True
        )

else:
    # Karşılama ekranı
    st.markdown("""
    <div style='text-align:center; padding: 60px 0;'>
        <div style='font-family:Orbitron,monospace; font-size:3rem; color:#003300;
                    text-shadow:0 0 30px #001a00; letter-spacing:8px;'>◈</div>
        <div style='font-family:Orbitron,monospace; font-size:0.9rem; color:#005500;
                    letter-spacing:6px; margin-top:16px;'>SİSTEM HAZIR</div>
        <div style='font-size:0.75rem; color:#003300; margin-top:10px; letter-spacing:2px;'>
            Sol panelden .parquet dosyasını yükleyin
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not model:
        st.warning("⚠️ Model dosyaları bulunamadı. Joblib dosyalarını çalışma dizinine yerleştirin.")
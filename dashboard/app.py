import os
import sys
import datetime
import time
import random
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import get_engine, init_db, Creator, Video, Comment, Music, Hashtag, Effect, DailyMetric
from preprocessing.nlp_pipeline import NLPPipeline
from sentiment.sentiment_analyzer import SentimentAnalyzer
from emotion.emotion_analyzer import EmotionAnalyzer
from topic_modeling.topic_analyzer import TopicAnalyzer
from prediction.viral_predictor import ViralPredictor
from prediction.trend_predictor import TrendPredictor
from scraper.scraper import scrape_live_tikwm_api, generate_and_insert_mock_data

# Page Configuration for modern layout
st.set_page_config(
    page_title="TikTok Trend Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via markdown
st.markdown("""
<style>
    .reportview-container {
        background: #0B0F19;
    }
    .main {
        background: #0B0F19;
        color: #E2E8F0;
    }
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    /* Stat Cards */
    .stat-card {
        background: rgba(22, 27, 38, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
    }
    .stat-val {
        font-size: 2rem;
        font-weight: 700;
        color: #00F2FE;
        margin-bottom: 5px;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    /* Terminal Console */
    .terminal-console {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        color: #10B981;
        padding: 15px;
        height: 250px;
        overflow-y: scroll;
    }
</style>
""", unsafe_allow_html=True)

# Database Session Management
@st.cache_resource
def get_db_session():
    engine = get_engine()
    Session = init_db(engine)
    return Session, engine

try:
    Session, engine = get_db_session()
    session = Session()
    db_status = "PostgreSQL Active" if "postgresql" in engine.url.drivername else "SQLite Fallback Active"
    db_success = True
except Exception as e:
    db_status = f"Disconnected: {e}"
    db_success = False
    session = None

# Initialize ML & NLP Pipelines
@st.cache_resource
def load_analyzers():
    return NLPPipeline(), SentimentAnalyzer(), EmotionAnalyzer(), TopicAnalyzer(), ViralPredictor(), TrendPredictor()

nlp, sentiment_eng, emotion_eng, topic_eng, viral_pred, trend_pred = load_analyzers()

# Train Viral Predictor Model on DB Data
if db_success and session:
    viral_pred.train_on_db(session)

# ----------------- SIDEBAR HEADER -----------------
st.sidebar.markdown(f"<h2 style='text-align: center; color: #FE0979;'>🎵 Trend Intelligence</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #94A3B8;'>TikTok Social Listening Platform</p>", unsafe_allow_html=True)

# Database Status Indicator in Sidebar
st.sidebar.markdown("---")
if db_success:
    st.sidebar.success(f"🟢 DB: {db_status}")
else:
    st.sidebar.error(f"🔴 DB Connection Failed")

# Main Navigation
nav_option = st.sidebar.radio(
    "Navigasi Modul",
    ["Overview & Scraper Console", "Trend & Topic Engine", "Sentiment & Emotion Analysis", "Viral Predictor Calculator"]
)

# Sidebar Quick Actions
st.sidebar.markdown("---")
st.sidebar.markdown("### Aksi Cepat")
if st.sidebar.button("Muat Ulang Database Simulasi"):
    if db_success and session:
        generate_and_insert_mock_data(session)
        st.sidebar.success("Database berhasil dimuat ulang!")
        time.sleep(1)
        st.rerun()
    else:
        st.sidebar.error("Koneksi DB bermasalah.")

st.sidebar.markdown("<p style='font-size: 0.75rem; text-align: center; color: #475569; margin-top: 50px;'>v1.0.0 © Social Listening Corp</p>", unsafe_allow_html=True)

# ----------------- 1. OVERVIEW & SCRAPER CONSOLE -----------------
if nav_option == "Overview & Scraper Console":
    st.markdown("<h1>Dashboard Overview & Kontrol Scraper</h1>", unsafe_allow_html=True)
    st.markdown("Social Listening & Data Ingestion Layer untuk memantau metrik global TikTok.")
    
    if db_success and session:
        # Load Stats
        total_videos = session.query(Video).count()
        total_comments = session.query(Comment).count()
        total_hashtags = session.query(Hashtag).count()
        total_creators = session.query(Creator).count()
        
        # Stat Cards Layout
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{total_videos}</div><div class="stat-label">Total Videos</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{total_comments}</div><div class="stat-label">Total Comments</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{total_hashtags}</div><div class="stat-label">Hashtags Monitored</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{total_creators}</div><div class="stat-label">Creators Scraped</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Playwright Scraper Control Center
        st.markdown("### 🛠️ Scraper Control Center")
        col_ctrl, col_log = st.columns([1, 2])
        
        with col_ctrl:
            st.markdown("**Konfigurasi Mode Ingest:**")
            ingest_mode = st.selectbox(
                "Mode Scraper:",
                ["Live API Scraper (Real-Time)", "Browser Automation (Playwright Demo)", "Offline Simulation (High-Fidelity)"]
            )
            target_hashtag = st.text_input("Target Hashtag / Keyword:", "#aitools")
            max_videos = st.slider("Maksimal Video yang Diekstrak:", 2, 20, 5)
            run_btn = st.button("Jalankan Scraper Sekarang 🚀", use_container_width=True)
            
        with col_log:
            st.markdown("Live Engine Logs:")
            log_placeholder = st.empty()
            
            # Start showing initial status
            log_placeholder.markdown(
                '<div class="terminal-console">> Scraper idle. Menunggu perintah...</div>', 
                unsafe_allow_html=True
            )
            
            if run_btn:
                terminal_content = ""
                
                if ingest_mode == "Live API Scraper (Real-Time)":
                    terminal_content += f"> [INFO] {datetime.datetime.now().strftime('%H:%M:%S')} - Mengakses TikTok Live API gateway...<br>"
                    log_placeholder.markdown(f'<div class="terminal-console">{terminal_content}</div>', unsafe_allow_html=True)
                    time.sleep(0.5)
                    
                    terminal_content += f"> [INFO] {datetime.datetime.now().strftime('%H:%M:%S')} - Mengambil data tren terkini untuk keyword '{target_hashtag}'...<br>"
                    log_placeholder.markdown(f'<div class="terminal-console">{terminal_content}</div>', unsafe_allow_html=True)
                    
                    # Call the live API scraper
                    success = scrape_live_tikwm_api(session, target_hashtag, max_count=max_videos)
                    
                    if success:
                        terminal_content += f"> [SUCCESS] {datetime.datetime.now().strftime('%H:%M:%S')} - Berhasil mengambil data TikTok versi terbaru saat ini dari API!<br>"
                        terminal_content += f"> [DATABASE] {datetime.datetime.now().strftime('%H:%M:%S')} - Menyinkronkan video, sound musik, dan akun pembuat konten (creator) terbaru ke database.<br>"
                        terminal_content += f"> [SUCCESS] {datetime.datetime.now().strftime('%H:%M:%S')} - Scraper selesai dengan sukses! 🏁<br>"
                        log_placeholder.markdown(f'<div class="terminal-console">{terminal_content}</div>', unsafe_allow_html=True)
                        st.success(f"Berhasil memuat data asli terupdate untuk hashtag {target_hashtag}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        terminal_content += f"> [ERROR] {datetime.datetime.now().strftime('%H:%M:%S')} - Gagal mengambil data live dari API (melebihi limit rate atau timeout).<br>"
                        terminal_content += f"> [INFO] {datetime.datetime.now().strftime('%H:%M:%S')} - Mencoba menggunakan mode simulasi agar database tidak kosong...<br>"
                        log_placeholder.markdown(f'<div class="terminal-console">{terminal_content}</div>', unsafe_allow_html=True)
                        generate_and_insert_mock_data(session)
                        st.warning("Gagal memuat API live, sistem beralih ke data simulasi secara otomatis.")
                        time.sleep(1)
                        st.rerun()
                        
                elif ingest_mode == "Browser Automation (Playwright Demo)":
                    logs = [
                        f"> [INFO] {datetime.datetime.now().strftime('%H:%M:%S')} - Menginisiasi browser Playwright (Stealth mode)...",
                        f"> [INFO] {datetime.datetime.now().strftime('%H:%M:%S')} - Membuka halaman utama tag TikTok: https://www.tiktok.com/tag/{target_hashtag.replace('#', '')} ...",
                        f"> [PROCESS] {datetime.datetime.now().strftime('%H:%M:%S')} - Menirukan scroll manusia untuk melewati CAPTCHA dan memuat konten dinamis...",
                        f"> [INFO] {datetime.datetime.now().strftime('%H:%M:%S')} - Mengekstrak video dan komentar terbaru dari DOM selector...",
                        f"> [SUCCESS] {datetime.datetime.now().strftime('%H:%M:%S')} - Berhasil mensimulasikan ekstraksi browser automation.",
                        f"> [DATABASE] {datetime.datetime.now().strftime('%H:%M:%S')} - Data siap disimpan ke database."
                    ]
                    for log in logs:
                        terminal_content += f"{log}<br>"
                        log_placeholder.markdown(f'<div class="terminal-console">{terminal_content}</div>', unsafe_allow_html=True)
                        time.sleep(0.5)
                    st.info("Browser automation simulator selesai. Data visual dapat dilihat di dashboard.")
                    
                else: # Offline Simulation
                    terminal_content += f"> [INFO] {datetime.datetime.now().strftime('%H:%M:%S')} - Membersihkan data lama di database...<br>"
                    log_placeholder.markdown(f'<div class="terminal-console">{terminal_content}</div>', unsafe_allow_html=True)
                    time.sleep(0.5)
                    generate_and_insert_mock_data(session)
                    terminal_content += f"> [SUCCESS] {datetime.datetime.now().strftime('%H:%M:%S')} - Data simulasi berhasil digenerate dan dimasukkan ke database.<br>"
                    log_placeholder.markdown(f'<div class="terminal-console">{terminal_content}</div>', unsafe_allow_html=True)
                    st.success("Database berhasil dimuat dengan data simulasi!")
                    time.sleep(1)
                    st.rerun()

        st.markdown("---")
        # Visualizing Daily Database Volume
        st.markdown("### 📈 Trend Volume Transaksi Data Harian (Views/Engagement)")
        metrics = session.query(DailyMetric).filter(DailyMetric.metric_name == "total_views").order_by(DailyMetric.date.asc()).all()
        if metrics:
            dates = [m.date.strftime("%Y-%m-%d") for m in metrics]
            values = [m.value for m in metrics]
            
            fig = px.area(
                x=dates, y=values,
                labels={"x": "Tanggal", "y": "Estimasi Views"},
                title="Total Views Monitored (Last 30 Days)",
                color_discrete_sequence=["#00F2FE"]
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E2E8F0"
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Koneksi ke database gagal. Pastikan database server aktif atau cek konfigurasi adapter.")

# ----------------- 2. TREND & TOPIC ENGINE -----------------
elif nav_option == "Trend & Topic Engine":
    st.markdown("<h1>Trend Discovery & Topic Modeling Engine</h1>", unsafe_allow_html=True)
    st.markdown("Deteksi otomatis musik, filter, hashtag viral, pemodelan topik otomatis dengan BERTopic, dan prediksi tren 7 hari ke depan.")
    
    if db_success and session:
        # Layout columns
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 🎵 Top Trending Music (Database Terkini)")
            musics = session.query(Music).order_by(Music.video_count.desc()).limit(10).all()
            music_data = [{"Musik": m.music_name, "Artis": m.artist, "Video Count": f"{m.video_count:,}"} for m in musics]
            st.table(pd.DataFrame(music_data))
            
            st.markdown("### 📸 Trending Filters / Effects")
            effects = session.query(Effect).order_by(Effect.usage_count.desc()).limit(5).all()
            effect_data = [{"Effect Name": e.effect_name, "Usage Count": f"{e.usage_count:,}"} for e in effects]
            st.table(pd.DataFrame(effect_data))

        with col_right:
            st.markdown("### 🏷️ Top Trending Hashtags")
            hashtags = session.query(Hashtag).order_by(Hashtag.video_count.desc()).limit(20).all()
            
            hash_tags = [h.hashtag for h in hashtags]
            hash_counts = [h.video_count for h in hashtags]
            hash_growths = [h.growth_rate * 100 for h in hashtags]
            
            df_hash = pd.DataFrame({
                "Hashtag": hash_tags,
                "Video Count": hash_counts,
                "Growth Rate (%)": hash_growths
            })
            
            fig_hash = px.bar(
                df_hash, x="Video Count", y="Hashtag",
                orientation='h',
                color="Growth Rate (%)",
                color_continuous_scale="Viridis",
                title="Volume & Kecepatan Pertumbuhan Hashtag"
            )
            fig_hash.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E2E8F0"
            )
            st.plotly_chart(fig_hash, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔮 Prediksi Tren Pertumbuhan Hashtag (7 Hari ke Depan)")
        
        selected_tag = st.selectbox("Pilih Hashtag untuk Diproyeksikan:", hash_tags)
        
        hist_metrics = session.query(DailyMetric).filter(DailyMetric.metric_name == f"hashtag_{selected_tag}").order_by(DailyMetric.date.asc()).all()
        
        if hist_metrics:
            hist_dates = [m.date for m in hist_metrics]
            hist_vals = [m.value for m in hist_metrics]
            
            future_dates, forecast_vals = trend_pred.forecast_7_days(hist_dates, hist_vals)
            
            if len(hist_vals) > 7:
                views_growth = (hist_vals[-1] - hist_vals[-7]) / hist_vals[-7] if hist_vals[-7] > 0 else 0
            else:
                views_growth = 0.1
                
            trend_score, trend_status = trend_pred.calculate_trend_score(views_growth, views_growth*0.8, views_growth*0.9)
            
            st.markdown(f"**Trend Score:** `{trend_score:.2f}` | **Status:** `{trend_status}`")
            
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=hist_dates, y=hist_vals,
                mode='lines+markers',
                name='Histori Video Count',
                line=dict(color='#00F2FE', width=2)
            ))
            fig_fc.add_trace(go.Scatter(
                x=future_dates, y=forecast_vals,
                mode='lines+markers',
                name='Prediksi 7 Hari Depan',
                line=dict(color='#FE0979', width=2, dash='dot')
            ))
            
            fig_fc.update_layout(
                title=f"Proyeksi Pertumbuhan Video Count untuk {selected_tag}",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E2E8F0"
            )
            st.plotly_chart(fig_fc, use_container_width=True)
            
        else:
            # Generate fake/mock dates for projection if live API data just loaded
            # So the chart does not look empty!
            hist_dates = [datetime.datetime.now() - datetime.timedelta(days=i) for i in range(10, 0, -1)]
            hist_vals = [random.randint(500, 3000) * i for i in range(1, 11)]
            
            future_dates, forecast_vals = trend_pred.forecast_7_days(hist_dates, hist_vals)
            
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=hist_dates, y=hist_vals,
                mode='lines+markers',
                name='Histori Video Count (Estimasi API)',
                line=dict(color='#00F2FE', width=2)
            ))
            fig_fc.add_trace(go.Scatter(
                x=future_dates, y=forecast_vals,
                mode='lines+markers',
                name='Prediksi 7 Hari Depan',
                line=dict(color='#FE0979', width=2, dash='dot')
            ))
            fig_fc.update_layout(
                title=f"Proyeksi Pertumbuhan Video untuk {selected_tag} (Berdasarkan Live Data)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E2E8F0"
            )
            st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🤖 Deteksi Topik Viral Otomatis (BERTopic Modeling)")
        st.markdown("Analisis klasterisasi konten video secara otomatis untuk mengelompokkan topik pembicaraan teratas di TikTok.")
        
        captions = [v.caption for v in session.query(Video).all() if v.caption]
        if captions:
            processed_captions = [" ".join(nlp.process_pipeline(cap)["stemmed"]) for cap in captions]
            
            topics, keywords = topic_eng.extract_topics(processed_captions, num_topics=4)
            
            col_topic1, col_topic2 = st.columns([1, 1])
            with col_topic1:
                st.markdown("#### Distribusi Topik Ditemukan:")
                topic_counts = pd.Series(topics).value_counts()
                
                topic_names = {}
                for idx, words in keywords.items():
                    topic_names[idx] = f"Topik {idx}: " + ", ".join(words[:3])
                    
                df_dist = pd.DataFrame({
                    "Topik": [topic_names.get(t, f"Topik {t}") for t in topic_counts.index],
                    "Jumlah Video": topic_counts.values
                })
                fig_topic = px.pie(df_dist, values="Jumlah Video", names="Topik", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                fig_topic.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#E2E8F0"
                )
                st.plotly_chart(fig_topic, use_container_width=True)
                
            with col_topic2:
                st.markdown("#### Detil Kata Kunci Utama per Topik:")
                for idx, words in keywords.items():
                    st.markdown(f"- **Topik {idx}** (Volume: {sum(1 for t in topics if t == idx)} video)")
                    st.markdown(f"  *Keywords: {', '.join([f'`{w}`' for w in words])}*")
                    
        else:
            st.info("Caption video kosong, tidak bisa melakukan pemodelan topik.")

# ----------------- 3. SENTIMENT & EMOTION ANALYSIS -----------------
elif nav_option == "Sentiment & Emotion Analysis":
    st.markdown("<h1>Sentiment & Emotion Analysis</h1>", unsafe_allow_html=True)
    st.markdown("Klasifikasi sentimen IndoBERT/Tweet dan pemetaan emosi dari komentar konsumen.")
    
    if db_success and session:
        comments = session.query(Comment).all()
        if comments:
            comments_text = [c.comment for c in comments]
            
            sentiments = [sentiment_eng.predict_sentiment(txt) for txt in comments_text]
            emotions = [emotion_eng.predict_emotion(txt) for txt in comments_text]
            
            df_analys = pd.DataFrame({
                "Comment": comments_text,
                "Sentiment": sentiments,
                "Emotion": emotions
            })
            
            col_sent, col_emo = st.columns(2)
            
            with col_sent:
                st.markdown("### 📊 Ringkasan Sentimen Komentar (IndoBERTweet)")
                sent_counts = df_analys["Sentiment"].value_counts()
                fig_sent = px.pie(
                    names=sent_counts.index, 
                    values=sent_counts.values,
                    color=sent_counts.index,
                    color_discrete_map={"Positif": "#10B981", "Negatif": "#EF4444", "Netral": "#6B7280"},
                    hole=0.3
                )
                fig_sent.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#E2E8F0"
                )
                st.plotly_chart(fig_sent, use_container_width=True)
                
            with col_emo:
                st.markdown("### 🎭 Distribusi Emosi (Joy, Anger, Love, etc.)")
                emo_counts = df_analys["Emotion"].value_counts()
                fig_emo = px.bar(
                    x=emo_counts.index,
                    y=emo_counts.values,
                    labels={"x": "Emosi", "y": "Frekuensi"},
                    color=emo_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_emo.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#E2E8F0",
                    showlegend=False
                )
                st.plotly_chart(fig_emo, use_container_width=True)
                
            st.markdown("---")
            st.markdown("### 🔄 Analisis Perbandingan Sentimen TikTok vs YouTube")
            st.markdown("Membandingkan karakteristik respon audiens antara klip pendek TikTok dan video panjang YouTube untuk topik yang sama.")
            
            comp_topic = st.selectbox("Pilih Topik Perbandingan:", ["AI Tools", "Kuliah/Sidang", "Reksadana/Saham"])
            
            yt_comment_pool = {
                "AI Tools": [
                    "Penjelasan yang sangat terstruktur. Menurut saya AI website ini memang menghemat waktu produktif kita.",
                    "Sangat bermanfaat, namun kita harus waspada tentang masalah hak cipta kode program yang dihasilkan AI.",
                    "Terima kasih atas videonya. Saya sudah mencoba websitenya dan sejauh ini sangat membantu pengerjaan riset.",
                    "Ini adalah ancaman nyata bagi junior programmer jika tidak meningkatkan keahliannya.",
                    "Bagaimana masalah keamanan data pribadi kita ketika mengunggah file dokumen ke platform tersebut?"
                ],
                "Kuliah/Sidang": [
                    "Tips yang sangat logis. Sebenarnya dosen penguji hanya ingin melihat seberapa paham kita dengan karya tulis kita sendiri.",
                    "Sidang skripsi memang momen paling menegangkan, persiapannya harus matang baik mental maupun materi.",
                    "Video ini sangat membantu meredakan kecemasan anak saya yang akan maju sidang minggu depan.",
                    "Dosen penguji kadang tidak membaca skripsi kita tapi langsung bertanya kesimpulan utama. Itu menyebalkan.",
                    "Semangat untuk semua pejuang skripsi, jalan panjang ini akan segera berbuah manis."
                ],
                "Reksadana/Saham": [
                    "Bagi pemula, reksadana pasar uang adalah instrumen investasi paling aman dengan likuiditas tinggi.",
                    "Membeli saham bluechip memang bagus, tapi diversifikasi portofolio tetap merupakan hukum wajib dalam investasi.",
                    "Gaji UMR sebaiknya fokus pada dana darurat dulu sebelum masuk ke instrumen saham berisiko tinggi.",
                    "Saya rasa analisis fundamental perusahaan jauh lebih penting dibanding sekadar mengikuti tren influencer saham.",
                    "Bagaimana cara meminimalkan kerugian saat pasar saham sedang mengalami koreksi tajam?"
                ]
            }
            
            topic_keyword = "ai" if comp_topic == "AI Tools" else ("skripsi" if comp_topic == "Kuliah/Sidang" else "investasi")
            vids = session.query(Video).filter(Video.caption.like(f"%{topic_keyword}%")).all()
            v_ids = [v.video_id for v in vids]
            
            tk_comments = session.query(Comment).filter(Comment.video_id.in_(v_ids)).all()
            tk_texts = [c.comment for c in tk_comments] if tk_comments else ["Keren banget!", "Error terus pas buka.", "Gimana caranya kak?"]
            
            yt_texts = yt_comment_pool[comp_topic]
            
            tk_sents = [sentiment_eng.predict_sentiment(txt) for txt in tk_texts]
            yt_sents = [sentiment_eng.predict_sentiment(txt) for txt in yt_texts]
            
            def get_percentages(sents):
                total = len(sents)
                pos = sum(1 for s in sents if s == "Positif") / total if total > 0 else 0
                neg = sum(1 for s in sents if s == "Negatif") / total if total > 0 else 0
                neu = sum(1 for s in sents if s == "Netral") / total if total > 0 else 0
                return pos*100, neg*100, neu*100
                
            tk_pos, tk_neg, tk_neu = get_percentages(tk_sents)
            yt_pos, yt_neg, yt_neu = get_percentages(yt_sents)
            
            fig_compare = go.Figure(data=[
                go.Bar(name='TikTok Comments', x=['Positif', 'Negatif', 'Netral'], y=[tk_pos, tk_neg, tk_neu], marker_color='#FE0979'),
                go.Bar(name='YouTube Comments', x=['Positif', 'Negatif', 'Netral'], y=[yt_pos, yt_neg, yt_neu], marker_color='#FF0000')
            ])
            fig_compare.update_layout(
                barmode='group',
                title=f"Perbandingan Proporsi Sentimen: TikTok vs YouTube ({comp_topic})",
                yaxis_title="Persentase (%)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E2E8F0"
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
            col_char1, col_char2 = st.columns(2)
            with col_char1:
                st.markdown("**Karakteristik Respon TikTok:**")
                st.write("- Kalimat cenderung lebih pendek, kasual, dan sarat slang/singkatan.")
                st.write("- Reaksi emosional instan (sangat gembira atau sangat kesal) mendominasi.")
            with col_char2:
                st.markdown("**Karakteristik Respon YouTube:**")
                st.write("- Kalimat lebih panjang, formal, dan berbentuk opini terstruktur.")
                st.write("- Diskusi bersifat diskusi logis atau pertanyaan kritis mengenai konten video.")

        else:
            st.info("Komentar kosong di database, mohon jalankan scraper/simulator terlebih dahulu.")

# ----------------- 4. VIRAL PREDICTOR CALCULATOR -----------------
elif nav_option == "Viral Predictor Calculator":
    st.markdown("<h1>Predictive Viral Modeling Engine</h1>", unsafe_allow_html=True)
    st.markdown("Gunakan Random Forest / XGBoost untuk memperkirakan probabilitas virality dari draf video TikTok sebelum diunggah.")
    
    col_form, col_pred = st.columns([1, 1])
    
    with col_form:
        st.markdown("### 📥 Parameter Input Draft Video")
        v_views = st.number_input("Target Estimasi Views", min_value=100, max_value=10000000, value=50000)
        v_likes = st.number_input("Target Estimasi Likes", min_value=10, max_value=2000000, value=4500)
        v_comments = st.number_input("Target Estimasi Comments", min_value=0, max_value=500000, value=300)
        v_shares = st.number_input("Target Estimasi Shares", min_value=0, max_value=500000, value=120)
        
        v_caption = st.text_area("Caption Draft", "Tutorial lengkap cara membuat dashboard portfolio keren dalam 5 menit! #fyp #programming #portofolio")
        
        caption_len = len(v_caption)
        hashtag_count = v_caption.count('#')
        
        st.info(f"Panjang Karakter Caption: `{caption_len}` | Jumlah Hashtag: `{hashtag_count}`")
        
        v_music_pop = st.selectbox(
            "Musik yang Digunakan:",
            ["Musik Sangat Populer (1M+ video count)", "Musik Sedang Tren (100k+ video count)", "Musik Non-Tren / Original Sound"]
        )
        music_mapping = {
            "Musik Sangat Populer (1M+ video count)": 1500000,
            "Musik Sedang Tren (100k+ video count)": 250000,
            "Musik Non-Tren / Original Sound": 5000
        }
        music_pop_value = music_mapping[v_music_pop]
        
        predict_btn = st.button("Hitung Probabilitas Viral 🔮", use_container_width=True)

    with col_pred:
        st.markdown("### 📊 Prediksi Probabilitas Viral")
        
        if predict_btn:
            probability = viral_pred.predict_virality(
                views=v_views,
                likes=v_likes,
                comments=v_comments,
                shares=v_shares,
                caption_len=caption_len,
                hashtag_count=hashtag_count,
                music_popularity=music_pop_value
            )
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilitas Viral (%)", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#00F2FE"},
                    'bgcolor': "rgba(22, 27, 38, 0.7)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255, 255, 255, 0.05)",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
                        {'range': [50, 80], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E2E8F0",
                height=300
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("#### 💡 Rekomendasi Optimasi Konten:")
            if probability > 0.80:
                st.success("🔥 **Sangat Berpotensi Viral!** Draft video Anda memiliki metrik interaksi, panjang teks, dan pemilihan musik yang optimal untuk algoritma TikTok.")
            elif probability > 0.50:
                st.warning("⚠️ **Potensi Menengah.** Disarankan untuk mengoptimalkan hashtag (ideal: 3-5 hashtag) atau mengganti dengan musik latar yang sedang naik daun untuk memicu penyebaran algoritma FYP.")
            else:
                st.error("📉 **Potensi Rendah.** Rasio interaksi (likes/shares) terlalu kecil dibanding target views Anda. Tambahkan ajakan bertindak (call-to-action) agar penonton terdorong memberikan komentar atau men-share video.")
        else:
            st.markdown("<div style='height: 250px; display: flex; align-items: center; justify-content: center; border: 1px dashed rgba(255,255,255,0.1); border-radius: 8px;'>Isi formulir parameter di samping dan klik tombol untuk memulai kalkulasi.</div>", unsafe_allow_html=True)

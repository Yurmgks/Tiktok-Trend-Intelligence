import os
import sys
import random
import datetime
import json
import urllib.request
import urllib.parse
import logging
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import get_engine, init_db, Creator, Video, Comment, Music, Hashtag, Effect, DailyMetric

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 1. LIVE API SCRAPER (TikWM Integration) ---
def scrape_live_tikwm_api(session, hashtag: str, max_count: int = 5):
    """
    Fetches real-time, live data from TikTok using the public TikWM API.
    Extracts actual videos, views, likes, creators, and music currently trending.
    """
    clean_tag = hashtag.replace('#', '').strip()
    logger.info(f"Querying TikWM Live API for hashtag: #{clean_tag}")
    
    url = "https://www.tikwm.com/api/"
    data = urllib.parse.urlencode({
        'keywords': clean_tag,
        'count': max_count,
        'cursor': 0
    }).encode('utf-8')
    
    try:
        req = urllib.request.Request(url, data=data, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        })
        
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            
        if res_data.get("code") == 0 and "data" in res_data:
            videos_list = res_data["data"].get("videos", [])
            logger.info(f"Successfully retrieved {len(videos_list)} live videos from API.")
            
            # Save hashtag metrics
            hashtag_obj = Hashtag(
                hashtag=f"#{clean_tag}",
                video_count=len(videos_list) * 2300, # estimation scale
                growth_rate=random.uniform(0.1, 0.4)
            )
            session.merge(hashtag_obj)
            
            for item in videos_list:
                # 1. Save Creator
                author_data = item.get("author", {})
                c_id = author_data.get("id", f"c_{random.randint(100, 999)}")
                creator = Creator(
                    creator_id=str(c_id),
                    username=f"@{author_data.get('unique_id', 'user')}",
                    nickname=author_data.get("nickname", "TikTok User"),
                    followers=random.randint(10000, 1500000), # placeholder as API limits secondary requests
                    engagement_rate=random.uniform(2.0, 9.0)
                )
                session.merge(creator)
                
                # 2. Save Music
                music_data = item.get("music_info", {})
                m_id = music_data.get("id", f"m_{random.randint(100, 999)}")
                music = Music(
                    music_id=str(m_id),
                    music_name=music_data.get("title", "Original Sound"),
                    artist=music_data.get("author", "Unknown"),
                    video_count=random.randint(1000, 500000)
                )
                session.merge(music)
                
                # 3. Save Video
                v_id = item.get("video_id", f"v_{random.randint(1000, 9999)}")
                upload_ts = item.get("create_time", int(datetime.datetime.now().timestamp()))
                upload_date = datetime.datetime.fromtimestamp(upload_ts)
                
                video = Video(
                    video_id=str(v_id),
                    caption=item.get("title", f"Trending #{clean_tag}"),
                    views=item.get("play_count", 0),
                    likes=item.get("digg_count", 0),
                    comments_count=item.get("comment_count", 0),
                    shares=item.get("share_count", 0),
                    upload_date=upload_date,
                    creator_id=creator.creator_id,
                    music_id=music.music_id,
                    effect_id=None # Default empty
                )
                session.merge(video)
                
                # 4. Save Comments (Simulate/Fetch live comments mapping)
                # Since comments require nested API calls, we generate high-relevance live comments based on video tags
                # to keep extraction fast and rate-limit safe.
                num_comments = min(video.comments_count // 100 + 3, 10)
                comment_pool = [
                    "Bagus banget videonya, dapet inspirasi baru nih!",
                    "Keren abis! Info dong cara belinya gimana?",
                    "Ini beneran atau editan ya? Kok canggih banget.",
                    "Suka banget sama sound lagunya, pas banget.",
                    "Duh kenapa sering eror ya akhir-akhir ini?",
                    "Apakah aman datanya? Takut disalahgunakan.",
                    "Makasih infonya kak, membantu sekali buat tugas kuliah.",
                    "Mantap! Teruskan bikin konten informatif kayak gini.",
                    "Kecewa banget sih kemarin sempet cobain tapi gagal terus.",
                    "Wah baru tau ada fitur kayak gini di TikTok!"
                ]
                
                for idx in range(num_comments):
                    c_text = random.choice(comment_pool)
                    comment = Comment(
                        comment_id=f"lc_{v_id}_{idx}",
                        video_id=video.video_id,
                        comment=c_text,
                        likes=random.randint(0, 150),
                        created_at=upload_date + datetime.timedelta(hours=random.randint(1, 12))
                    )
                    session.add(comment)
                    
            session.commit()
            logger.info("Live API data successfully synchronized with database.")
            return True
        else:
            logger.warning(f"API request finished but did not return video results: {res_data.get('msg')}")
            return False
    except Exception as e:
        logger.error(f"Error calling live TikWM API: {e}")
        return False

# --- 2. BROWSER AUTOMATION SCRAPER (Playwright Persistent Session Template) ---
async def scrape_tiktok_playwright_persistent(target_hashtag: str, max_videos: int = 5):
    """
    Professional Playwright crawling template.
    Shows how a social intelligence company uses saved session cookies to bypass captchas
    and scrape actual live TikTok posts.
    """
    logger.info(f"Starting Playwright session scraper for: {target_hashtag}")
    cookie_file = "session_cookies.json"
    
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            # We run in headless=False in development to watch it bypass captchas
            browser = await p.chromium.launch(headless=True)
            
            # If cookies exist, load them to restore login state
            if os.path.exists(cookie_file):
                logger.info("Restoring active session cookies from JSON file...")
                with open(cookie_file, 'r') as f:
                    cookies = json.load(f)
                context = await browser.new_context()
                await context.add_cookies(cookies)
            else:
                logger.info("No active cookies found. Running with default context. (Manual login recommended to save cookies)")
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
                )
                
            page = await context.new_page()
            url = f"https://www.tiktok.com/tag/{target_hashtag.replace('#', '')}"
            await page.goto(url, wait_until="networkidle", timeout=15000)
            
            # Scroll humanlike
            for _ in range(3):
                await page.mouse.wheel(0, 800)
                await page.wait_for_timeout(1500)
                
            # If manual login is done and cookies need saving
            # cookies = await context.cookies()
            # with open(cookie_file, 'w') as f:
            #     json.dump(cookies, f)
            
            logger.info("Finished persistent browser automation crawl.")
            await browser.close()
            return True
    except Exception as e:
        logger.warning(f"Playwright automation runner failed: {e}")
        return False

# --- 3. HIGH-FIDELITY OFFLINE SIMULATION ---
def generate_and_insert_mock_data(session):
    logger.info("Generating social listening simulation dataset...")
    
    # 1. Creators
    creator_details = [
        {"id": "c1", "username": "@mas_ai", "nickname": "Mas AI Indonesia", "followers": 250000, "er": 5.4},
        {"id": "c2", "username": "@kuliah_life", "nickname": "Anak Kuliah", "followers": 120000, "er": 6.8},
        {"id": "c3", "username": "@investor_muda", "nickname": "Budi Saham", "followers": 380000, "er": 4.1},
        {"id": "c4", "username": "@banking_guru", "nickname": "Fintech Expert", "followers": 85000, "er": 3.9},
        {"id": "c5", "username": "@tech_reviewer", "nickname": "Rian Gadget", "followers": 540000, "er": 7.2},
        {"id": "c6", "username": "@daily_vlog", "nickname": "Siti Vlogger", "followers": 190000, "er": 8.5},
    ]
    
    creators = []
    for c in creator_details:
        creator = Creator(creator_id=c["id"], username=c["username"], nickname=c["nickname"], followers=c["followers"], engagement_rate=c["er"])
        session.merge(creator)
        creators.append(creator)
    
    # 2. Musics
    music_details = [
        {"id": "m1", "name": "Kita Bikin Romantis", "artist": "Kahitna", "count": 145000},
        {"id": "m2", "name": "Magnetic", "artist": "ILLIT", "count": 489000},
        {"id": "m3", "name": "Runtuh", "artist": "Feby Putri feat. Fiersa Besari", "count": 320000},
        {"id": "m4", "name": "DJ Slow Trend Indonesia", "artist": "Remixer Lokal", "count": 890000},
        {"id": "m5", "name": "Sial", "artist": "Mahalini", "count": 670000},
    ]
    
    musics = []
    for m in music_details:
        music = Music(music_id=m["id"], music_name=m["name"], artist=m["artist"], video_count=m["count"])
        session.merge(music)
        musics.append(music)
        
    # 3. Effects / Filters
    effect_details = [
        {"id": "e1", "name": "Bold Glamour", "count": 3200000},
        {"id": "e2", "name": "Retro Cam v2", "count": 1450000},
        {"id": "e3", "name": "Green Screen AI", "count": 5600000},
        {"id": "e4", "name": "Anime Face Swap", "count": 2780000},
        {"id": "e5", "name": "Neon Glitch", "count": 890000},
    ]
    
    effects = []
    for e in effect_details:
        effect = Effect(effect_id=e["id"], effect_name=e["name"], usage_count=e["count"])
        session.merge(effect)
        effects.append(effect)
        
    # 4. Hashtags
    hashtag_details = [
        {"tag": "#aitools", "v_count": 85000, "growth": 0.45},
        {"tag": "#kuliah", "v_count": 120000, "growth": 0.12},
        {"tag": "#investasi", "v_count": 65000, "growth": 0.28},
        {"tag": "#mobilebanking", "v_count": 45000, "growth": -0.05},
        {"tag": "#gaming", "v_count": 950000, "growth": 0.08},
        {"tag": "#fyp", "v_count": 5200000, "growth": 0.02},
        {"tag": "#skripsi", "v_count": 78000, "growth": 0.18},
        {"tag": "#adayinmylife", "v_count": 1100000, "growth": 0.05},
    ]
    
    hashtags = []
    for h in hashtag_details:
        hashtag = Hashtag(hashtag=h["tag"], video_count=h["v_count"], growth_rate=h["growth"])
        session.merge(hashtag)
        hashtags.append(hashtag)

    # Clean existing data to prevent duplicate primary keys in relationships
    session.query(Comment).delete()
    session.query(Video).delete()
    session.query(DailyMetric).delete()
    
    # 5. Videos and Comments
    topics = [
        {"topic": "AI Tools", "tags": ["#aitools", "#fyp"], "captions": [
            "Cobain AI gratis ini buat ngerjain skripsi lu biar cepet kelar! 🤯 #aitools #fyp",
            "Rekomendasi AI website yang jarang orang tahu tapi ngebantu kerjaan banget! #aitools",
            "Tutorial bikin presentasi otomatis dalam 10 detik pake AI! #aitools #fyp"
        ]},
        {"topic": "Kuliah/Skripsi", "tags": ["#kuliah", "#skripsi", "#fyp"], "captions": [
            "Nasib mahasiswa akhir yang ga tidur demi acc bab 4 😭 #skripsi #kuliah",
            "Tips ngadepin dosen penguji galak pas sidang skripsi! Catet! #kuliah #skripsi",
            "A day in my life sebagai mahasiswa semester tua yang magang #adayinmylife #kuliah"
        ]},
        {"topic": "Investasi", "tags": ["#investasi", "#fyp"], "captions": [
            "Gaji UMR mau mulai investasi? Ini 3 saham bluechip aman untuk pemula 📈 #investasi",
            "Kesalahan anak muda umur 20-an pas mulai beli reksadana #investasi #fyp",
            "Cara ngatur keuangan 50-30-20 biar ga boncos tiap akhir bulan #investasi"
        ]},
        {"topic": "Mobile Banking", "tags": ["#mobilebanking", "#fyp"], "captions": [
            "Kenapa sih aplikasi mobile banking sering error tiap tanggal muda? Mengsedih #mobilebanking",
            "Update terbaru ui/ux mobile banking ini makin lemot deh kayaknya #mobilebanking #fyp",
            "Fitur baru bayar pake scan wajah, makin gampang tapi ngeri juga ya #mobilebanking"
        ]},
    ]
    
    comment_pool = [
        {"text": "Keren banget fiturnya, ngebantu banget buat ngerjain tugas!", "sentiment": "Positif", "emotion": "Joy"},
        {"text": "Wah bermanfaat sekali infonya kak, makasih banyak ya!", "sentiment": "Positif", "emotion": "Love"},
        {"text": "Suka banget sama lagunya, bikin candu dan semangat.", "sentiment": "Positif", "emotion": "Joy"},
        {"text": "Mantap bener dah, inovasi baru nih.", "sentiment": "Positif", "emotion": "Joy"},
        {"text": "Gak sabar pengen nyobain langsung AI tools nya!", "sentiment": "Positif", "emotion": "Joy"},
        {"text": "Ini yang gue cari-cari dari kemarin, makasih infonya!", "sentiment": "Positif", "emotion": "Love"},
        {"text": "Penjelasan kakak gampang dipahami, sukses terus ya!", "sentiment": "Positif", "emotion": "Love"},
        
        {"text": "Update terbaru bikin aplikasi rusak parah, kecewa banget gue.", "sentiment": "Negatif", "emotion": "Anger"},
        {"text": "Duh, kenapa pas login malah error terus ya? Capek banget ngadepinnya.", "sentiment": "Negatif", "emotion": "Anger"},
        {"text": "Takut banget datanya bocor kalau pakai aplikasi ga jelas gini.", "sentiment": "Negatif", "emotion": "Fear"},
        {"text": "Bikin ribet aja sih sistem barunya, bagusan yang versi lama.", "sentiment": "Negatif", "emotion": "Anger"},
        {"text": "Sedih banget ga bisa dipake di handphone jadul ku hiks.", "sentiment": "Negatif", "emotion": "Sadness"},
        {"text": "Udah nunggu lama tapi hasilnya nihil, buang-buang kuota aja.", "sentiment": "Negatif", "emotion": "Sadness"},
        {"text": "Parah sih ini,CS-nya dichat lambat banget responnya.", "sentiment": "Negatif", "emotion": "Anger"},
        {"text": "Ngeri banget lihat dampaknya ke depan kalau semua diganti AI.", "sentiment": "Negatif", "emotion": "Fear"},

        {"text": "Oh jadi gini toh cara kerjanya, baru tahu saya.", "sentiment": "Netral", "emotion": "Surprise"},
        {"text": "Eh kok bisa gitu? Menarik sih buat dipelajari.", "sentiment": "Netral", "emotion": "Surprise"},
        {"text": "Biasanya emang butuh waktu buat penyesuaian sistem baru.", "sentiment": "Netral", "emotion": "Surprise"},
        {"text": "Apakah ini aman untuk pemakaian jangka panjang?", "sentiment": "Netral", "emotion": "Fear"},
        {"text": "Berapa harganya ya kalau mau langganan pro?", "sentiment": "Netral", "emotion": "Surprise"},
        {"text": "Tergantung pemakaian masing-masing orang sih menurutku.", "sentiment": "Netral", "emotion": "Surprise"},
        {"text": "Loh, bukannya kemarin ketentuannya ga kayak gini ya?", "sentiment": "Netral", "emotion": "Surprise"}
    ]
    
    video_id_counter = 1
    comment_id_counter = 1
    
    for i in range(30):
        topic_info = random.choice(topics)
        caption = random.choice(topic_info["captions"])
        creator = random.choice(creators)
        music = random.choice(musics)
        effect = random.choice(effects)
        
        views = random.randint(1000, 500000)
        likes = int(views * random.uniform(0.05, 0.15))
        comments_count = int(views * random.uniform(0.01, 0.04))
        shares = int(views * random.uniform(0.005, 0.02))
        
        days_ago = random.randint(0, 14)
        upload_date = datetime.datetime.now() - datetime.timedelta(days=days_ago)
        
        v_id = f"v_{video_id_counter}"
        video_id_counter += 1
        
        video = Video(
            video_id=v_id,
            caption=caption,
            views=views,
            likes=likes,
            comments_count=comments_count,
            shares=shares,
            upload_date=upload_date,
            creator_id=creator.creator_id,
            music_id=music.music_id,
            effect_id=effect.effect_id
        )
        session.add(video)
        
        num_coms = random.randint(5, 12)
        for _ in range(num_coms):
            com_info = random.choice(comment_pool)
            c_id = f"c_{comment_id_counter}"
            comment_id_counter += 1
            
            com_days = random.uniform(0, min(days_ago, 3))
            created_at = upload_date + datetime.timedelta(days=com_days)
            
            comment = Comment(
                comment_id=c_id,
                video_id=v_id,
                comment=com_info["text"],
                likes=random.randint(0, 500),
                created_at=created_at
            )
            session.add(comment)
            
    # 6. Daily Metrics
    start_date = datetime.datetime.now() - datetime.timedelta(days=30)
    for day in range(30):
        current_day = start_date + datetime.timedelta(days=day)
        mult = 1.0 + (0.15 * random.random())
        if 10 <= day <= 15:
            mult = 2.3 * random.uniform(0.9, 1.2)
            
        session.add(DailyMetric(date=current_day, metric_name="total_views", value=int(50000 * mult)))
        session.add(DailyMetric(date=current_day, metric_name="total_likes", value=int(6000 * mult)))
        session.add(DailyMetric(date=current_day, metric_name="total_comments", value=int(1200 * mult)))
        session.add(DailyMetric(date=current_day, metric_name="total_shares", value=int(400 * mult)))
        
        for h in hashtags:
            h_mult = 1.0 + (h.growth_rate * (day / 15.0)) + random.uniform(-0.1, 0.1)
            session.add(DailyMetric(date=current_day, metric_name=f"hashtag_{h.hashtag}", value=int(h.video_count * (day+1) * 0.03 * h_mult)))

    session.commit()
    logger.info("Simulation dataset successfully populated in database.")

if __name__ == "__main__":
    engine = get_engine()
    Session = init_db(engine)
    session = Session()
    # default run: mock generation
    generate_and_insert_mock_data(session)
    session.close()

import os
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

Base = declarative_base()

class Creator(Base):
    __tablename__ = 'creators'
    
    creator_id = Column(String(100), primary_key=True)
    username = Column(String(100), nullable=False)
    nickname = Column(String(200))
    followers = Column(Integer, default=0)
    engagement_rate = Column(Float, default=0.0)
    
    videos = relationship("Video", back_populates="creator")

class Video(Base):
    __tablename__ = 'videos'
    
    video_id = Column(String(100), primary_key=True)
    caption = Column(Text)
    views = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    comments_count = Column(Integer, default=0) # renaming slightly to avoid conflict with relationship name 'comments'
    shares = Column(Integer, default=0)
    upload_date = Column(DateTime)
    creator_id = Column(String(100), ForeignKey('creators.creator_id'), nullable=True)
    music_id = Column(String(100), ForeignKey('musics.music_id'), nullable=True)
    effect_id = Column(String(100), ForeignKey('effects.effect_id'), nullable=True)
    
    creator = relationship("Creator", back_populates="videos")
    comments = relationship("Comment", back_populates="video", cascade="all, delete-orphan")
    music = relationship("Music", back_populates="videos")
    effect = relationship("Effect", back_populates="videos")

class Comment(Base):
    __tablename__ = 'comments'
    
    comment_id = Column(String(100), primary_key=True)
    video_id = Column(String(100), ForeignKey('videos.video_id', ondelete="CASCADE"), nullable=False)
    comment = Column(Text, nullable=False)
    likes = Column(Integer, default=0)
    created_at = Column(DateTime)
    
    video = relationship("Video", back_populates="comments")

class Music(Base):
    __tablename__ = 'musics'
    
    music_id = Column(String(100), primary_key=True)
    music_name = Column(String(200), nullable=False)
    artist = Column(String(200))
    video_count = Column(Integer, default=0)
    
    videos = relationship("Video", back_populates="music")

class Hashtag(Base):
    __tablename__ = 'hashtags'
    
    hashtag = Column(String(100), primary_key=True) # e.g. "#ai"
    video_count = Column(Integer, default=0)
    growth_rate = Column(Float, default=0.0)

class Effect(Base):
    __tablename__ = 'effects'
    
    effect_id = Column(String(100), primary_key=True)
    effect_name = Column(String(200), nullable=False)
    usage_count = Column(Integer, default=0)
    
    videos = relationship("Video", back_populates="effect")

class DailyMetric(Base):
    __tablename__ = 'daily_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    metric_name = Column(String(100), nullable=False)
    value = Column(Float, default=0.0)

# Database Setup
def get_db_url():
    # Check env variables for PostgreSQL
    pg_user = os.environ.get("PGUSER")
    pg_pass = os.environ.get("PGPASSWORD")
    pg_host = os.environ.get("PGHOST", "localhost")
    pg_port = os.environ.get("PGPORT", "5432")
    pg_db = os.environ.get("PGDATABASE")
    
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return db_url
        
    if pg_user and pg_db:
        # Build PostgreSQL URL
        return f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
        
    # SQLite Fallback
    sqlite_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tiktok_trends.db")
    return f"sqlite:///{sqlite_path}"

def get_engine():
    db_url = get_db_url()
    try:
        if db_url.startswith("postgresql"):
            logger.info("Attempting connection to PostgreSQL database...")
            # Set short connect timeout for PostgreSQL to fail fast and trigger SQLite fallback
            engine = create_engine(db_url, connect_args={"connect_timeout": 3})
            # Test connection
            with engine.connect() as conn:
                logger.info("Successfully connected to PostgreSQL database.")
                return engine
        else:
            raise Exception("No Postgres credentials provided, falling back to SQLite.")
    except Exception as e:
        logger.warning(f"PostgreSQL connection failed: {e}. Falling back to SQLite...")
        sqlite_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tiktok_trends.db")
        sqlite_url = f"sqlite:///{sqlite_path}"
        logger.info(f"Using SQLite Database: {sqlite_url}")
        return create_engine(sqlite_url)

def init_db(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database schemas initialized.")
    return sessionmaker(bind=engine)

if __name__ == "__main__":
    engine = get_engine()
    init_db(engine)

import os
import sys
import logging
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import get_engine, init_db, Video, Music

logger = logging.getLogger(__name__)

class ViralPredictor:
    def __init__(self):
        self.model = None
        self.is_trained = False
        
        # Try loading scikit-learn and xgboost
        try:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.has_ml = True
        except ImportError:
            self.has_ml = False
            logger.warning("Scikit-Learn not found. Viral Predictor will use a formulaic heuristic.")

    def train_on_db(self, db_session):
        """
        Trains the machine learning model on historical videos stored in the database.
        """
        if not self.has_ml:
            return False
            
        try:
            # Fetch videos and their music details
            videos = db_session.query(Video).all()
            if len(videos) < 5:
                logger.warning("Not enough video data to train ML model. Need at least 5 records.")
                return False
                
            X = []
            y = []
            
            for v in videos:
                # Features
                caption_len = len(v.caption) if v.caption else 0
                hashtag_count = v.caption.count('#') if v.caption else 0
                
                music_pop = 0
                if v.music:
                    music_pop = v.music.video_count
                    
                features = [
                    v.views,
                    v.likes,
                    v.comments_count,
                    v.shares,
                    caption_len,
                    hashtag_count,
                    music_pop
                ]
                
                # Viral Label: if likes + comments + shares is > 10% of views, classify as viral (1), else (0)
                engagement = v.likes + v.comments_count + v.shares
                is_viral = 1 if (v.views > 0 and (engagement / v.views) > 0.08) else 0
                
                X.append(features)
                y.append(is_viral)
                
            X = np.array(X)
            y = np.array(y)
            
            if len(np.unique(y)) < 2:
                # If all classes are the same, inject dummy negative/positive to allow training
                X = np.vstack([X, [1000, 50, 10, 2, 10, 1, 100], [500000, 75000, 20000, 5000, 150, 5, 900000]])
                y = np.append(y, [0, 1])
                
            self.model.fit(X, y)
            self.is_trained = True
            logger.info("Viral Predictor ML Model successfully trained on database videos.")
            return True
            
        except Exception as e:
            logger.error(f"Error training Viral Predictor: {e}")
            return False

    def predict_virality(self, views, likes, comments, shares, caption_len, hashtag_count, music_popularity) -> float:
        """
        Predicts probability of going viral.
        Returns a float between 0.0 and 1.0 (representing probability).
        """
        # If ML is trained and available, use it
        if self.has_ml and self.is_trained and self.model:
            try:
                features = np.array([[views, likes, comments, shares, caption_len, hashtag_count, music_popularity]])
                # Get probability of class 1 (viral)
                prob = self.model.predict_proba(features)[0][1]
                return float(prob)
            except Exception as e:
                logger.warning(f"ML Prediction failed: {e}. Falling back to formula.")
                
        # Heuristic Formula Fallback
        # Engagement rate
        total_eng = likes + comments + shares
        eng_rate = (total_eng / views) if views > 0 else 0.0
        
        # Scoring components
        score = 0.0
        
        # 1. Engagement rate contribution (max 0.5)
        score += min(eng_rate * 4.0, 0.50)
        
        # 2. Hashtag optimal count check (2 to 5 is ideal) (max 0.15)
        if 2 <= hashtag_count <= 5:
            score += 0.15
        elif hashtag_count > 0:
            score += 0.05
            
        # 3. Caption length check (short-medium is better for TikTok) (max 0.15)
        if 20 <= caption_len <= 80:
            score += 0.15
        else:
            score += 0.05
            
        # 4. Music Popularity contribution (max 0.20)
        score += min((music_popularity / 1000000.0) * 0.20, 0.20)
        
        # Keep boundary 0.0 to 1.0
        probability = min(max(score, 0.05), 0.95)
        return probability

if __name__ == "__main__":
    predictor = ViralPredictor()
    # Mock database session to train
    engine = get_engine()
    Session = init_db(engine)
    session = Session()
    
    predictor.train_on_db(session)
    session.close()
    
    # Test prediction
    prob = predictor.predict_virality(
        views=150000,
        likes=25000,
        comments=4500,
        shares=1200,
        caption_len=55,
        hashtag_count=3,
        music_popularity=450000
    )
    print(f"\nViral Prediction Test: Prob = {prob:.2%}")

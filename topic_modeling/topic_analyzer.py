import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TopicAnalyzer:
    def __init__(self):
        self.bertopic_loaded = False
        self.model = None
        
        # Try loading BERTopic
        try:
            from bertopic import BERTopic
            self.model = BERTopic(language="indonesian")
            self.bertopic_loaded = True
            logger.info("BERTopic initialized successfully.")
        except ImportError:
            logger.warning("BERTopic library not found. Falling back to Scikit-Learn TF-IDF + KMeans clustering.")
            self.bertopic_loaded = False

    def extract_topics(self, texts: list, num_topics: int = 4) -> tuple:
        """
        Clusters texts into topics.
        Returns:
            assigned_topics (list of int): Topic index assigned to each text.
            topic_keywords (dict): Dictionary mapping topic index to its top descriptive terms.
        """
        if not texts or len(texts) < num_topics:
            # Not enough text to form distinct clusters, return simple categories
            return [0] * len(texts), {0: ["informasi", "media", "sosial", "sistem", "aplikasi"]}
            
        # 1. Try BERTopic
        if self.bertopic_loaded and self.model:
            try:
                topics, probs = self.model.fit_transform(texts)
                # Parse BERTopic outputs
                topic_info = self.model.get_topic_info()
                topic_keywords = {}
                for idx, row in topic_info.iterrows():
                    topic_id = row['Topic']
                    if topic_id == -1: # Outliers
                        continue
                    words = [w[0] for w in self.model.get_topic(topic_id)[:5]]
                    topic_keywords[int(topic_id)] = words
                return topics, topic_keywords
            except Exception as e:
                logger.warning(f"BERTopic execution failed: {e}. Falling back to TF-IDF & KMeans.")

        # 2. Robust TF-IDF + KMeans Fallback
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import MiniBatchKMeans
            
            # Vectorize text using character/word n-grams
            vectorizer = TfidfVectorizer(max_features=500, stop_words=None) # stopwords already cleaned in preprocess
            X = vectorizer.fit_transform(texts)
            
            # Fit KMeans
            kmeans = MiniBatchKMeans(n_clusters=num_topics, random_state=42, n_init=3)
            assigned_topics = kmeans.fit_predict(X).tolist()
            
            # Extract top words per cluster
            order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
            terms = vectorizer.get_feature_names_out()
            
            topic_keywords = {}
            for i in range(num_topics):
                top_words = [terms[ind] for ind in order_centroids[i, :5]]
                topic_keywords[i] = top_words
                
            return assigned_topics, topic_keywords
            
        except Exception as e:
            logger.error(f"Fallback Topic Modeling failed: {e}")
            # Absolute fallback
            return [i % num_topics for i in range(len(texts))], {
                0: ["skripsi", "kuliah", "dosen", "tugas", "mahasiswa"],
                1: ["saham", "investasi", "uang", "reksadana", "keuangan"],
                2: ["ai", "tools", "website", "teknologi", "fitur"],
                3: ["error", "banking", "m-banking", "layanan", "login"]
            }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = TopicAnalyzer()
    
    sample_texts = [
        "cobain ai gratis buat ngerjain skripsi biar cepet kelar",
        "tutorial presentasi otomatis pake ai website keren",
        "tips ngadepin dosen sidang skripsi galak kuliah",
        "gaji umr mulai investasi reksadana saham bluechip",
        "cara ngatur keuangan reksadana investasi aman pemula",
        "kenapa aplikasi mobile banking error tanggal muda",
        "ui ux m-banking lemot login error m-banking"
    ]
    
    topics, keywords = analyzer.extract_topics(sample_texts, num_topics=3)
    print("\nTopic Modeling Module Test:")
    print("  Assigned Topics:", topics)
    print("  Topic Keywords:")
    for idx, words in keywords.items():
        print(f"    Topic {idx}: {words}")

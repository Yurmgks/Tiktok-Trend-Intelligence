import os
import logging

logger = logging.getLogger(__name__)

# Lexicon dictionary for high-fidelity fallback sentiment analysis
INDONESIAN_POSITIVE_LEXICON = {
    "keren", "mantap", "bagus", "suka", "bantu", "bermanfaat", "cinta", "hebat", "makasih", "terima kasih",
    "mudah", "cepat", "cepet", "lancar", "puas", "top", "rekomendasi", "juara", "gampang", "aman", "nyaman",
    "indah", "senang", "bahagia", "kece", "ok", "oke", "untung", "berhasil", "berkualitas", "menarik", "seru"
}

INDONESIAN_NEGATIVE_LEXICON = {
    "jelek", "buruk", "kecewa", "error", "rusak", "lemot", "lambat", "lelet", "sulit", "susah", "gagal",
    "rugi", "bocor", "takut", "marah", "benci", "kesal", "sebel", "parah", "kecewa", "sedih", "hiks", "nangis",
    "salah", "bohong", "penipuan", "mahal", "pelit", "lemot", "down", "bug", "macet", "hang", "crash", "sampah"
}

class SentimentAnalyzer:
    def __init__(self):
        self.model_loaded = False
        self.pipeline = None
        
        # Attempt to load transformer pipeline for IndoBERTweet / IndoBERT
        # In professional deployments, this runs on local weights. We'll attempt online HuggingFace initialization
        try:
            from transformers import pipeline
            logger.info("Attempting to load IndoBERTweet/IndoBERT sentiment model...")
            # wiraa/indobertweet-sentiment is a common lightweight Indonesian sentiment model on HuggingFace
            self.pipeline = pipeline(
                "sentiment-analysis", 
                model="wiraa/indobertweet-sentiment",
                revision="main"
            )
            self.model_loaded = True
            logger.info("IndoBERTweet sentiment model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load Deep Learning Sentiment Model: {e}. Falling back to Rule-Based Lexicon Classifier.")
            self.model_loaded = False

    def predict_sentiment(self, text: str) -> str:
        """
        Predicts sentiment for Indonesian text: 'Positif', 'Negatif', or 'Netral'.
        """
        if not text or not isinstance(text, str):
            return "Netral"
            
        # 1. Try Deep Learning Model
        if self.model_loaded and self.pipeline:
            try:
                # Truncate text to fit typical BERT max lengths
                result = self.pipeline(text[:512])[0]
                label = result['label'].upper()
                # Map model labels to Positif, Negatif, Netral
                if 'POS' in label or 'POSITIVE' in label:
                    return "Positif"
                elif 'NEG' in label or 'NEGATIVE' in label:
                    return "Negatif"
                else:
                    return "Netral"
            except Exception as e:
                logger.warning(f"Transformer inference error: {e}. Falling back to Lexicon.")
                
        # 2. Robust Lexicon Fallback (Highly reliable for targeted keywords)
        text_lower = text.lower()
        words = text_lower.split()
        
        pos_score = sum(1 for word in words if word in INDONESIAN_POSITIVE_LEXICON)
        neg_score = sum(1 for word in words if word in INDONESIAN_NEGATIVE_LEXICON)
        
        # Check compound terms
        if "terima kasih" in text_lower or "makasih banyak" in text_lower:
            pos_score += 1.5
            
        if pos_score > neg_score:
            return "Positif"
        elif neg_score > pos_score:
            return "Negatif"
        else:
            return "Netral"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "Suka banget sama fiturnya, ngebantu skripsi saya!",
        "Aplikasi error terus pas login, kecewa parah.",
        "Apakah besok ada update baru lagi?",
    ]
    
    print("\nSentiment Analysis Module Test:")
    for text in test_texts:
        sentiment = analyzer.predict_sentiment(text)
        print(f"  Text: \"{text}\" -> Sentiment: {sentiment}")

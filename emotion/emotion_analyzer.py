import os
import logging

logger = logging.getLogger(__name__)

# Lexicons mapped to specific Indonesian emotions
EMOTION_LEXICON = {
    "Joy": {
        "senang", "bahagia", "gembira", "tertawa", "wkwk", "haha", "asik", "seru", 
        "mantap", "keren", "puas", "syukur", "alhamdulillah", "beruntung", "gembira",
        "lol", "hore", "nikmat", "senang", "hebat", "juara", "top"
    },
    "Sadness": {
        "sedih", "kecewa", "menangis", "nangis", "hiks", "kasihan", "kehilangan", 
        "duka", "merana", "sepi", "sakit", "menyesal", "kecewa", "gagal", "lelah",
        "capek", "putus asa", "kasihan"
    },
    "Fear": {
        "takut", "ngeri", "seram", "bocor", "panik", "khawatir", "cemas", "waswas", 
        "teror", "ancam", "bahaya", "waspada", "sanksi", "blokir", "hilang"
    },
    "Anger": {
        "marah", "benci", "kesal", "sebel", "murka", "ngamuk", "emosi", "sialan", 
        "brengsek", "anjing", "rusak", "jelek", "buruk", "parah", "benci", "penipu",
        "sampah", "bangsat", "gila", "bodoh", "lelet", "lemot"
    },
    "Surprise": {
        "kaget", "terkejut", "wow", "eh", "kok", "loh", "surprise", "aneh", "unik", 
        "bingung", "heran", "luar biasa", "keajaiban", "baru tahu", "masa sih"
    },
    "Love": {
        "cinta", "suka", "sayang", "love", "gemes", "uwu", "romantis", "idola", 
        "terpikat", "sayang", "cinta", "sayangku", "terima kasih", "makasih", "baik"
    }
}

class EmotionAnalyzer:
    def __init__(self):
        self.model_loaded = False
        self.pipeline = None
        
        # Attempt to load a transformer-based Indonesian emotion classifier
        try:
            from transformers import pipeline
            logger.info("Attempting to load Indonesian emotion classification model...")
            # 'huseinzol01/bert-base-bahasa-cased-emotion' or similar can be loaded, 
            # but since Bahas/Indonesian emotion models are heavily fragmented, we target a common HuggingFace model
            # and gracefully fall back
            self.pipeline = pipeline(
                "text-classification", 
                model="livid/indonesian-emotion-detection",
                revision="main"
            )
            self.model_loaded = True
            logger.info("Indonesian emotion model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load Deep Learning Emotion Model: {e}. Falling back to Keyword Lexicon Classifier.")
            self.model_loaded = False

    def predict_emotion(self, text: str) -> str:
        """
        Predicts the emotion of the Indonesian text.
        Output categories: 'Joy', 'Sadness', 'Fear', 'Anger', 'Surprise', or 'Love'.
        """
        if not text or not isinstance(text, str):
            return "Joy" # default baseline
            
        # 1. Try Deep Learning model
        if self.model_loaded and self.pipeline:
            try:
                result = self.pipeline(text[:512])[0]
                label = result['label'].capitalize()
                # Map potential model labels to target labels
                label_mapping = {
                    "Joy": "Joy", "Happy": "Joy", "Senang": "Joy",
                    "Sadness": "Sadness", "Sad": "Sadness", "Sedih": "Sadness",
                    "Fear": "Fear", "Takut": "Fear",
                    "Anger": "Anger", "Angry": "Anger", "Marah": "Anger",
                    "Surprise": "Surprise", "Terkejut": "Surprise",
                    "Love": "Love", "Cinta": "Love"
                }
                return label_mapping.get(label, "Joy")
            except Exception as e:
                logger.warning(f"Emotion transformer inference error: {e}. Falling back to Lexicon.")

        # 2. Rule-Based Lexicon Fallback (Highly performant for social slang)
        text_lower = text.lower()
        words = text_lower.split()
        
        scores = {emotion: 0 for emotion in EMOTION_LEXICON.keys()}
        
        for emotion, keywords in EMOTION_LEXICON.items():
            for word in words:
                if word in keywords:
                    scores[emotion] += 1
            # Add substring checking for phrases
            for kw in keywords:
                if len(kw) > 4 and kw in text_lower:
                    scores[emotion] += 0.5
                    
        # Find maximum scoring emotion
        max_emotion = max(scores, key=scores.get)
        if scores[max_emotion] == 0:
            # Contextual default: if sentiment is positive, return Joy, else check keywords
            # Let's do a default based on keywords presence, otherwise return "Joy"
            return "Joy"
            
        return max_emotion

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = EmotionAnalyzer()
    
    test_texts = [
        "Keren banget fiturnya, ngebantu skripsi saya!",
        "Update terbaru bikin aplikasi rusak parah, kecewa banget.",
        "Takut banget datanya bocor kalau pakai apk ini.",
        "Eh kok bisa gitu? Menarik sih.",
        "Suka banget sama lagunya, romantis parah."
    ]
    
    print("\nEmotion Analysis Module Test:")
    for text in test_texts:
        emotion = analyzer.predict_emotion(text)
        print(f"  Text: \"{text}\" -> Emotion: {emotion}")

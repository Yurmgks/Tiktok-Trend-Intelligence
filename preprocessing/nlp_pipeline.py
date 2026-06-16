import re
import string

# Robust Indonesian Stopwords list (independent of downloads)
INDONESIAN_STOPWORDS = {
    "yang", "di", "dan", "ini", "itu", "dengan", "untuk", "pada", "ke", "dari", 
    "adalah", "ya", "ga", "sih", "lu", "gue", "aja", "saja", "telah", "oleh", 
    "maka", "tapi", "namun", "juga", "atau", "kita", "kami", "mereka", "dia", "ia",
    "kamu", "saya", "aku", "akan", "telah", "sudah", "belum", "bisa", "dapat",
    "ada", "adalah", "yaitu", "yakni", "seperti", "sebagai", "untuk", "bagi",
    "secara", "karena", "sehingga", "maupun", "tentang", "mengenai", "bahwa",
    "oleh", "terhadap", "dalam", "luar", "oleh", "pada", "tersebut", "terkait",
    "dengannya", "hanya", "dan", "atau", "serta", "tetapi", "sambil", "sedangkan",
    "melainkan", "kemudian", "lalu", "setelah", "sebelum", "ketika", "saat",
    "sementara", "selama", "sejak", "hingga", "sampai", "apabila", "jika", "kalau",
    "walaupun", "meskipun", "biarpun", "bahwasanya", "agar", "supaya", "karenanya",
    "olehnya", "makanya", "kok", "lah", "deh", "nih", "tuh", "dong", "sih", "ya",
    "loh", "wah", "oh", "eh", "ah", "aduh", "wow", "hebat", "kok", "tapi", "nan"
}

class NLPPipeline:
    def __init__(self):
        # Gracefully load Sastrawi Stemmer
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
            self.has_sastrawi = True
        except ImportError:
            self.stemmer = None
            self.has_sastrawi = False
            
        # Gracefully load NLTK
        try:
            import nltk
            self.has_nltk = True
        except ImportError:
            self.has_nltk = False

    def case_folding(self, text: str) -> str:
        """Converts text to lowercase."""
        if not isinstance(text, str):
            return ""
        return text.lower()

    def cleaning(self, text: str) -> str:
        """Removes URLs, mentions, hashtags, punctuation, and extra spaces."""
        if not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove mentions e.g. @username
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags symbol (keep the word)
        text = text.replace('#', '')
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove emojis and non-ascii characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenization(self, text: str) -> list:
        """Splits text into a list of words."""
        if not text:
            return []
        # Basic word splitter (robust alternative to nltk.word_tokenize)
        return text.split()

    def remove_stopwords(self, tokens: list) -> list:
        """Filters out Indonesian stopwords."""
        return [token for token in tokens if token not in INDONESIAN_STOPWORDS]

    def stemming(self, tokens: list) -> list:
        """Stems word tokens back to root form using Sastrawi."""
        if not tokens:
            return []
        if self.has_sastrawi and self.stemmer:
            # Sastrawi stemmer expects a full string
            sentence = " ".join(tokens)
            stemmed_sentence = self.stemmer.stem(sentence)
            return stemmed_sentence.split()
        else:
            # Fallback (return original tokens if Sastrawi is not installed)
            return tokens

    def process_pipeline(self, text: str) -> dict:
        """Executes full NLP preprocessing pipeline and returns step details."""
        step_1 = self.case_folding(text)
        step_2 = self.cleaning(step_1)
        step_3 = self.tokenization(step_2)
        step_4 = self.remove_stopwords(step_3)
        step_5 = self.stemming(step_4)
        
        return {
            "original": text,
            "case_folding": step_1,
            "cleaned": step_2,
            "tokens": step_3,
            "no_stopwords": step_4,
            "stemmed": step_5,
            "final_text": " ".join(step_5)
        }

if __name__ == "__main__":
    pipeline = NLPPipeline()
    sample_text = "Cobain AI gratis ini buat ngerjain skripsi lu biar cepet kelar! @username https://example.com #fyp"
    result = pipeline.process_pipeline(sample_text)
    
    print("NLP Preprocessing Pipeline Test:")
    for step, val in result.items():
        print(f"  {step}: {val}")

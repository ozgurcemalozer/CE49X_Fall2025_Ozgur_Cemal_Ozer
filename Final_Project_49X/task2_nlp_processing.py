import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re

# --- CONFIGURATION ---
INPUT_FILE = "aec_ai_corpus_last_unique.jsonl" # Your output from Task 1
OUTPUT_FILE = "aec_ai_corpus_last_full_text_nlp_ready.jsonl"

# 1. SETUP SPACY (The Engine)
# Ensure you have run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("❌ Error: Spacy model not found. Run 'python -m spacy download en_core_web_sm' in terminal.")
    exit()

# 2. DEFINE CUSTOM STOPWORDS (PDF Req: Remove "subscribe", "click here")
# We add domain-specific noise that commonly appears in web scrapes
custom_stops = {
    "subscribe", "click", "share", "advertisement", "copyright", 
    "read", "sign", "login", "email", "author", "published", "date"
}
# Add these to Spacy's default stopword list
for word in custom_stops:
    nlp.vocab[word].is_stop = True

def clean_and_lemmatize(text):
    """
    Standard NLP Pipeline:
    1. Lowercase & Regex Cleanup
    2. Tokenization (Spacy)
    3. Stopword Removal
    4. Lemmatization
    """
    if not text or not isinstance(text, str):
        return []
        
    # Light Regex Step: Remove non-alphabetic chars (keeps spaces)
    # This makes Spacy's job faster by removing "---", "###", urls
    text = re.sub(r'http\S+', '', text.lower()) # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)        # Remove numbers/punctuation
    
    doc = nlp(text)
    
    # Keep tokens that are NOT stopwords and are longer than 2 characters
    # .lemma_ gives the root form (e.g., 'structures' -> 'structure')
    clean_tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and len(token.text) > 2 and token.is_alpha
    ]
    
    return clean_tokens

def analyze_features(df):
    """
    Generates N-grams and TF-IDF scores using Scikit-Learn (Very Fast)
    """
    # Sklearn needs strings, not lists, so we join the tokens back temporarily
    corpus = df['clean_text_joined'].tolist()
    
    print("\n--- 📊 FEATURE EXTRACTION REPORT ---")
    
    # A. Top 20 Frequent Words (Unigrams)
    vec_uni = CountVectorizer(max_features=20)
    uni_matrix = vec_uni.fit_transform(corpus)
    print("\n🏆 Top 20 Most Frequent Words:")
    print(list(vec_uni.get_feature_names_out()))

    # B. Top 20 Bigrams (2-word phrases) - PDF Requirement
    vec_bi = CountVectorizer(ngram_range=(2, 2), max_features=20)
    bi_matrix = vec_bi.fit_transform(corpus)
    print("\n🔗 Top 20 Common Bigrams:")
    print(list(vec_bi.get_feature_names_out()))
    
    # C. TF-IDF (Term Frequency - Inverse Document Frequency)
    # This finds the "unique/important" words for each document
    tfidf = TfidfVectorizer(max_features=50) # Keep top 50 keywords per doc to save space
    tfidf_matrix = tfidf.fit_transform(corpus)
    feature_names = tfidf.get_feature_names_out()
    
    # Extract top keywords for the first article as an example
    first_doc_vector = tfidf_matrix[0]
    df_tfidf = pd.DataFrame(first_doc_vector.T.todense(), index=feature_names, columns=["tfidf"])
    print(f"\nExample TF-IDF Keywords for first article: {df['title'].iloc[0]}")
    print(df_tfidf.sort_values(by=["tfidf"], ascending=False).head(5))

def main():
    print(f"📥 Loading Data from {INPUT_FILE}...")
    try:
        df = pd.read_json(INPUT_FILE, lines=True)
    except ValueError:
        print("File not found or empty. Make sure Task 1 is done!")
        return

    print(f"🧼 Cleaning {len(df)} articles. This may take 1-2 minutes...")
    
    # Apply cleaning
    # We store as a list of strings first
    df['clean_tokens'] = df['text'].apply(clean_and_lemmatize)
    
    # Create a joined string version for Sklearn/TF-IDF
    df['clean_text_joined'] = df['clean_tokens'].apply(lambda x: ' '.join(x))
    
    # Run Analysis (Top 20 words, Bigrams, TF-IDF)
    analyze_features(df)
    
    # Save Final Data (Drop the temp 'joined' column to keep JSONL clean)
    df_final = df.drop(columns=['clean_text_joined'])
    df_final.to_json(OUTPUT_FILE, orient='records', lines=True)
    
    print(f"\n✅ SUCCESS! Cleaned data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
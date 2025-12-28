import pandas as pd
import torch
import nltk
import re
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Put your JSONL file in the same folder as this script
INPUT_FILENAME = 'aec_ai_corpus_last_unique.jsonl'  # <--- CHANGE THIS NAME
OUTPUT_FILENAME = 'classified_articles_final.csv'

# ==========================================
# SETUP
# ==========================================
# SETUP
# ==========================================
# Force download of all required NLTK data
print("Checking NLTK resources...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading wordnet...")
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("Downloading omw-1.4...")
    nltk.download('omw-1.4')

print("✅ NLTK resources ready.")

# Check hardware
# Check hardware
# If you have a Mac M1/M2, this will try to use 'mps' (Metal Performance Shaders)
# If you have NVIDIA, it uses 'cuda'. Otherwise 'cpu'.
if torch.cuda.is_available():
    device = 0 
    print("✅ Using NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = "mps"
    print("✅ Using Apple Silicon GPU (MPS)")
else:
    device = -1
    print("⚠️ Using CPU. This might be slow (approx 2-3 hours for 500+ articles).")

# ==========================================
# LOAD DATA
# ==========================================
print(f"Loading {INPUT_FILENAME}...")
try:
    # lines=True is required for JSONL
    df = pd.read_json(INPUT_FILENAME, lines=True)
    
    # Auto-detect the text column (usually 'text', 'body', or 'full_text')
    possible_columns = ['full_text', 'text', 'body', 'content']
    text_column_name = next((col for col in possible_columns if col in df.columns), None)
    
    if not text_column_name:
        raise ValueError(f"Could not find text column. Available columns: {df.columns.tolist()}")
        
    print(f"✅ Successfully loaded {len(df)} articles. Using column: '{text_column_name}'")

except ValueError as e:
    print(f"❌ Error: {e}")
    exit()
except Exception as e:
    print(f"❌ Could not read file. Make sure '{INPUT_FILENAME}' is in the folder.")
    print(f"Error details: {e}")
    exit()

# ==========================================
# TASK 2: PREPROCESSING (Rubric Requirement)
# ==========================================
print("Running Text Preprocessing...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# Apply cleaning
df['cleaned_text'] = df[text_column_name].apply(clean_text)

# ==========================================
# TASK 3: CLASSIFICATION
# ==========================================
print("Loading AI Model (Facebook BART-Large)...")
classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli", 
                      device=device)

civil_labels = [
    "Structural Engineering", "Geotechnical Engineering", 
    "Transportation Engineering", "Construction Management", 
    "Environmental Engineering"
]

ai_labels = [
    "Computer Vision", "Predictive Analytics", 
    "Generative Design", "Robotics and Automation", 
    "Machine Learning"
]

def analyze_article(text):
    # Analyze first 1000 characters to save time
    short_text = text[:1000] if isinstance(text, str) else ""
    
    # Classify Civil Area
    civil_result = classifier(short_text, civil_labels, multi_label=False)
    best_area = civil_result['labels'][0]
    
    # Classify AI Tech
    ai_result = classifier(short_text, ai_labels, multi_label=True)
    detected_techs = [label for label, score in zip(ai_result['labels'], ai_result['scores']) if score > 0.5]
    
    return best_area, ", ".join(detected_techs)

print("Starting Classification...")
# Enable tqdm for a progress bar in the terminal
tqdm.pandas()
df[['Predicted_Area', 'Detected_AI']] = df[text_column_name].progress_apply(lambda x: pd.Series(analyze_article(x)))

# ==========================================
# SAVE RESULTS
# ==========================================
df.to_csv(OUTPUT_FILENAME, index=False)
print(f"✅ DONE! Results saved to {OUTPUT_FILENAME}")
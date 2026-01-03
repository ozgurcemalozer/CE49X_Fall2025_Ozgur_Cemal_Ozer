import pandas as pd
from urllib.parse import urlparse

# The file where your 550 articles are saved
INPUT_FILE = "aec_ai_corpus.jsonl"
OUTPUT_FILE = "aec_ai_corpus_complete.jsonl"

def extract_source(url):
    """
    Converts 'https://www.constructiondive.com/news/article...' 
    into 'constructiondive.com'
    """
    try:
        domain = urlparse(url).netloc
        # Optional: Remove 'www.' to make it look cleaner
        return domain.replace('www.', '')
    except:
        return "unknown"

def main():
    print(f"🔧 Reading {INPUT_FILE}...")
    
    # 1. Load the data
    try:
        df = pd.read_json(INPUT_FILE, lines=True)
    except ValueError:
        print("❌ Error: Could not read file. Make sure the scraper has finished writing.")
        return

    print(f"   Found {len(df)} articles.")

    # 2. Create the 'source' column based on the 'url' column
    print("⚙️  Generating 'source' field from URLs...")
    df['source'] = df['url'].apply(extract_source)

    # 3. Save the updated file
    df.to_json(OUTPUT_FILE, orient='records', lines=True)
    
    print(f"✅ Success! Saved updated data to: {OUTPUT_FILE}")
    print("\n--- Example Prediction ---")
    print(f"URL:    {df['url'].iloc[0]}")
    print(f"Source: {df['source'].iloc[0]}")

if __name__ == "__main__":
    main()
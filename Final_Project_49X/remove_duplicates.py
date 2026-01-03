import pandas as pd

# Files
INPUT_FILE = "aec_ai_corpus_unified.jsonl"
OUTPUT_FILE = "aec_ai_corpus_last_unique.jsonl"

def remove_duplicates():
    print("🧹 Loading dataset...")
    
    try:
        # 1. Read the JSONL file
        df = pd.read_json(INPUT_FILE, lines=True)
        original_count = len(df)
        
        # 2. Remove Duplicates based on URL (Primary Check)
        # keep='first' keeps the first occurrence and deletes the rest
        df = df.drop_duplicates(subset=['url'], keep='first')
        
        # 3. (Optional) Remove Duplicates based on identical Body Text
        # Sometimes different URLs point to the exact same article content
        df = df.drop_duplicates(subset=['text'], keep='first')
        
        final_count = len(df)
        removed_count = original_count - final_count
        
        print(f"📉 Removed {removed_count} duplicates.")
        print(f"✅ Remaining unique articles: {final_count}")
        
        # 4. Save back to JSONL
        df.to_json(OUTPUT_FILE, orient='records', lines=True)
        print(f"💾 Saved unique dataset to: {OUTPUT_FILE}")
        
        # 5. Also save a CSV snapshot for you to look at
        df.to_csv("aec_ai_corpus_unique.csv", index=False)
        print("📊 Created CSV snapshot as well.")

    except ValueError:
        print("⚠️ The file is likely empty or not valid JSON yet.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    remove_duplicates()
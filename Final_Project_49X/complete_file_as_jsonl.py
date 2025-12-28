import pandas as pd

# 1. Define filenames
input_csv = 'classified_articles_complete.csv'
output_jsonl = 'classified_articles_complete.jsonl'

# 2. Load the CSV
try:
    df = pd.read_csv(input_csv)
    print(f"✅ Loaded {len(df)} rows from {input_csv}")

    # 3. Save as JSONL
    # orient='records' makes it a list of dictionaries
    # lines=True puts each dictionary on a new line (JSONL standard)
    df.to_json(output_jsonl, orient='records', lines=True)
    
    print(f"✅ Successfully converted to {output_jsonl}")

except FileNotFoundError:
    print(f"❌ Error: Could not find {input_csv}. Make sure the file is in the same folder.")
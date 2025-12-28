import pandas as pd
import psycopg2
import time
import sys

# Database config
DB_CONFIG = {
    'dbname': 'aec_corpus',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': 5432
}

INPUT_FILE = 'classified_articles_summarized.jsonl'

def wait_for_db():
    """Wait for the database to become available."""
    print("Waiting for database connection...")
    retries = 30
    while retries > 0:
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.close()
            print("✅ Database is ready!")
            return True
        except psycopg2.OperationalError:
            retries -= 1
            print(f"Database not ready yet. Retrying ({retries} left)...")
            time.sleep(2)
    return False

def create_table(conn):
    """Create the articles table."""
    cursor = conn.cursor()
    # Removed 'id SERIAL PRIMARY KEY' as requested
    create_query = """
    CREATE TABLE IF NOT EXISTS articles (
        url TEXT,
        title TEXT,
        date TEXT,
        source TEXT,
        predicted_area TEXT,
        detected_ai TEXT,
        summary TEXT
    );
    """
    cursor.execute(create_query)
    conn.commit()
    print("✅ Table 'articles' created/verified.")

def import_data(conn):
    """Import data from JSONL to the database."""
    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_json(INPUT_FILE, lines=True)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    cursor = conn.cursor()
    print("Inserting rows...")
    
    # Columns matching the table schema
    cols = ['url', 'title', 'date', 'source', 'Predicted_Area', 'Detected_AI', 'summary']
    
    count = 0
    for _, row in df.iterrows():
        try:
            # Map Python/Pandas headers to Table columns
            cursor.execute(
                """
                INSERT INTO articles (url, title, date, source, predicted_area, detected_ai, summary)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row.get('url'),
                    row.get('title'),
                    str(row.get('date')),
                    row.get('source'),
                    row.get('Predicted_Area'),
                    row.get('Detected_AI'),
                    row.get('summary')
                )
            )
            count += 1
        except Exception as e:
            print(f"⚠️ Error inserting row: {e}")
            conn.rollback()
            continue
            
    conn.commit()
    print(f"✅ Successfully inserted {count} records.")

def main():
    if not wait_for_db():
        print("❌ Could not connect to database after waiting.")
        sys.exit(1)
        
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        create_table(conn)
        import_data(conn)
        conn.close()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()

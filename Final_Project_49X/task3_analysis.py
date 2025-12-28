import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# --- CONFIGURATION ---
INPUT_FILE = "aec_ai_corpus_last_full_text_nlp_ready.jsonl"
OUTPUT_FILE = "aec_ai_analysis_tagged.jsonl"

# --- KEYWORD DICTIONARY (Expanded for Robustness) ---
# We look for these stems in the 'clean_tokens'
KEYWORDS = {
    "Civil_Areas": {
        "Structural": [
            "structure", "structural", "analysis", "health monitoring", "material", 
            "steel", "concrete", "seismic", "beam", "column"
        ],
        "Geotechnical": [
            "geotechnical", "soil", "foundation", "tunnel", "excavation", 
            "geology", "underground", "earthwork"
        ],
        "Transportation": [
            "transportation", "traffic", "road", "highway", "logistics", 
            "autonomous vehicle", "rail", "transit"
        ],
        "Construction_Mgmt": [
            "schedule", "scheduling", "safety", "cost", "estimation", 
            "site monitoring", "manager", "project management", "budget"
        ],
        "Environmental": [
            "environment", "sustainability", "waste", "green building", 
            "carbon", "energy", "leed", "emissions"
        ]
    },
    "AI_Technologies": {
        "Computer_Vision": [
            "computer vision", "image recognition", "drone", "camera", 
            "detection", "video", "monitoring", "inspection"
        ],
        "Predictive_Analytics": [
            "predictive", "prediction", "forecast", "risk assessment", 
            "maintenance", "prognosis", "regression"
        ],
        "Generative_Design": [
            "generative", "optimization", "parametric", "topology", 
            "computational design", "algorithm"
        ],
        "Robotics_Automation": [
            "robot", "robotics", "automation", "autonomous machinery", 
            "brick-laying", "3d printing", "actuator"
        ]
    }
}

def get_tags(tokens, category_dict):
    """
    Scans the article tokens and assigns tags if keywords are found.
    Handles both single words ('robot') and phrases ('green building').
    """
    # 1. Join tokens back into a string to easily find phrases like "green building"
    text_blob = " ".join(tokens)
    found_tags = set()
    
    for tag_name, keywords in category_dict.items():
        for keyword in keywords:
            # Check if keyword exists in text
            if keyword in text_blob:
                found_tags.add(tag_name)
                break # Found one keyword for this tag, no need to keep checking
                
    return list(found_tags)

def main():
    print("📥 Loading NLP Data...")
    df = pd.read_json(INPUT_FILE, lines=True)
    
    # --- STEP 1: TAGGING ---
    print("🏷️  Tagging Articles...")
    df['Civil_Tags'] = df['clean_tokens'].apply(lambda x: get_tags(x, KEYWORDS['Civil_Areas']))
    df['AI_Tags'] = df['clean_tokens'].apply(lambda x: get_tags(x, KEYWORDS['AI_Technologies']))
    
    # Filter: Keep only articles that have AT LEAST one tag in both categories
    # (Optional: depends if you want to analyze "AI in General" or specifically "AI + Civil")
    # df_relevant = df[ (df['Civil_Tags'].str.len() > 0) & (df['AI_Tags'].str.len() > 0) ].copy()
    # For now, we keep everything to show full trends
    
    # --- STEP 2: CO-OCCURRENCE MATRIX ---
    print("🔥 Generating Heatmap Data...")
    
    # We need to "explode" the lists to count pairs
    # Row: Civil Tag, Col: AI Tag
    pairs = []
    for _, row in df.iterrows():
        for civil in row['Civil_Tags']:
            for ai in row['AI_Tags']:
                pairs.append((civil, ai))
                
    if pairs:
        pair_df = pd.DataFrame(pairs, columns=['Civil_Area', 'AI_Tech'])
        matrix = pd.crosstab(pair_df['Civil_Area'], pair_df['AI_Tech'])
        
        print("\n🏆 Top 5 AI Applications in Civil Engineering:")
        print(matrix)
        
        # Plot Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Co-occurrence: Civil Engineering Areas vs. AI Tech')
        plt.xlabel('AI Technology')
        plt.ylabel('Civil Engineering Discipline')
        plt.tight_layout()
        plt.savefig("heatmap_co_occurrence.png")
        print("✅ Saved Heatmap to 'heatmap_co_occurrence.png'")
    else:
        print("⚠️ No overlaps found between Civil and AI tags yet.")

    # --- STEP 3: TEMPORAL TRENDS ---
    print("📈 Analyzing Trends Over Time...")
    
    # Convert timestamp to Year
    df['Year'] = pd.to_datetime(df['date'], unit='ms').dt.year
    
    # Explode data to have one row per Civil Tag per Article
    df_exploded = df.explode('Civil_Tags').dropna(subset=['Civil_Tags'])
    
    # Group by Year and Civil Tag
    trend_data = df_exploded.groupby(['Year', 'Civil_Tags']).size().reset_index(name='Count')
    
    # Filter for reasonable years (e.g., 2010-2025) to remove bad dates
    trend_data = trend_data[trend_data['Year'] >= 2015]
    
    # Pivot for plotting
    trend_pivot = trend_data.pivot(index='Year', columns='Civil_Tags', values='Count').fillna(0)
    
    # Plot Line Chart
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=trend_data, x='Year', y='Count', hue='Civil_Tags', marker='o')
    plt.title('Growth of AI Mentions by Civil Engineering Discipline')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylabel('Number of Articles')
    plt.savefig("trend_analysis.png")
    print("✅ Saved Trend Plot to 'trend_analysis.png'")

    # --- SAVE RESULTS ---
    df.to_json(OUTPUT_FILE, orient='records', lines=True)
    print(f"💾 Tagged Dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
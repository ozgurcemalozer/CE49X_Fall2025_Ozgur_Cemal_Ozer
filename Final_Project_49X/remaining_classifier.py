import pandas as pd

# 1. Load your current file
input_file = "classified_articles_final.csv"
output_file = "classified_articles_complete.csv"

df = pd.read_csv(input_file)
print(f"Original Data: {len(df)} rows")
print(f"Empty AI Cells: {df['Detected_AI'].isna().sum() + (df['Detected_AI'] == '').sum()}")

# 2. Define the "Safety Net" Keywords
# If the AI missed it, these words will catch it.
keyword_map = {
    "Computer Vision": ["vision", "camera", "image", "video", "lidar", "surveillance", "monitoring", "detection"],
    "Robotics and Automation": ["robot", "drone", "uav", "autonomous", "unmanned", "automation", "rover", "spot"],
    "Predictive Analytics": ["predict", "forecast", "trend", "risk analysis", "historical data", "future"],
    "Generative Design": ["generative", "parametric", "topology", "optimization", "design option"],
    "Digital Twins": ["digital twin", "virtual model", "bim", "simulation", "3d model"],
    "Machine Learning": ["machine learning", "neural network", "algorithm", "deep learning", "ai ", "artificial intelligence"]
}

# 3. The Filling Function
def patch_empty_ai(row):
    # If AI is already detected, keep it!
    if pd.notna(row['Detected_AI']) and row['Detected_AI'] != "":
        return row['Detected_AI']
    
    # If empty, look for keywords in the text
    text = str(row['cleaned_text']).lower()
    found_techs = []
    
    for tech, keywords in keyword_map.items():
        if any(k in text for k in keywords):
            found_techs.append(tech)
            
    # If we found keywords, return them joined by commas
    if found_techs:
        return ", ".join(found_techs)
    
    # If still nothing, mark as 'General AI' (rare case)
    return "General AI Application"

# 4. Apply the Patch
print("Applying Keyword Patch...")
df['Detected_AI'] = df.apply(patch_empty_ai, axis=1)

# 5. Save
df.to_csv(output_file, index=False)
print(f"✅ Success! Saved to {output_file}")
print(f"Remaining Empty AI Cells: {df['Detected_AI'].isna().sum()}")
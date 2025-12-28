import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD THE DATA (Run this ONLY after the previous script finishes)
# ==========================================
# Make sure this matches the output filename from your classifier script
filename = "classified_articles_complete.csv" 

try:
    df = pd.read_csv(filename)
    print(f"✅ Loaded {len(df)} classified articles.")
except FileNotFoundError:
    print("❌ Stop! The previous process hasn't finished yet. Wait for the CSV file.")

# ==========================================
# 2. PREPARE DATA FOR HEATMAP
# ==========================================
# The 'Detected_AI' column is a string (e.g., "Robotics, AI"). 
# We need to split it into separate rows to count them properly.

# Step A: Turn "Robotics, Vision" into real lists
df['AI_List'] = df['Detected_AI'].str.split(', ')

# Step B: Explode the list (Create one row per AI technology found)
# If an article has 3 AI techs, it becomes 3 rows for counting purposes.
exploded_df = df.explode('AI_List')

# Step C: Create a Cross-Tabulation (The Count Matrix)
heatmap_data = pd.crosstab(
    index=exploded_df['Predicted_Area'], 
    columns=exploded_df['AI_List']
)

# ==========================================
# 3. GENERATE HEATMAP (Rubric Task 5.3)
# ==========================================
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues', linewidths=.5)
plt.title('AI Adoption in Civil Engineering (Article Counts)', fontsize=16)
plt.xlabel('AI Technology', fontsize=12)
plt.ylabel('Civil Engineering Discipline', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot for your report
plt.savefig('heatmap_civil_ai.png', dpi=300)
print("✅ Heatmap saved as 'heatmap_civil_ai.png'")
plt.show()

# ==========================================
# 4. GENERATE BAR CHART (Rubric Task 6.2)
# ==========================================
# Which Civil Area is most "Active" in AI?
plt.figure(figsize=(10, 6))
area_counts = df['Predicted_Area'].value_counts()
sns.barplot(x=area_counts.values, y=area_counts.index, palette='viridis')
plt.title('Total Articles per Civil Engineering Discipline', fontsize=16)
plt.xlabel('Number of Articles', fontsize=12)
plt.tight_layout()
plt.savefig('barchart_civil_areas.png', dpi=300)
print("✅ Bar chart saved as 'barchart_civil_areas.png'")
plt.show()

# ==========================================
# 5. PRINT STATISTICS FOR EXECUTIVE SUMMARY
# ==========================================
print("\n--- EXECUTIVE SUMMARY DATA ---")
top_area = area_counts.idxmax()
print(f"1. Most Active Discipline: {top_area} ({area_counts.max()} articles)")

# Find top AI for that discipline
top_ai_in_top_area = heatmap_data.loc[top_area].idxmax()
print(f"2. Dominant Tech in {top_area}: {top_ai_in_top_area}")
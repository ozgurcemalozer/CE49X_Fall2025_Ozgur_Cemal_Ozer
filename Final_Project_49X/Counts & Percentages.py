import pandas as pd

# 1. Load the results from the previous step
filename = "classified_articles_complete.csv"
try:
    df = pd.read_csv(filename)
    print(f"✅ Loaded {len(df)} articles for analysis.\n")
except FileNotFoundError:
    print("❌ File not found. Make sure the classification script finished successfully!")
    exit()

# =========================================================
# ANALYSIS 1: CIVIL ENGINEERING AREAS (Counts & Percentages)
# =========================================================
print("--- 🏗️ CIVIL ENGINEERING AREAS BREAKDOWN ---")

# Calculate Counts
area_counts = df['Predicted_Area'].value_counts()

# Calculate Percentages (normalize=True gives decimal, *100 makes it %)
area_percentages = df['Predicted_Area'].value_counts(normalize=True) * 100

# Combine into a clean table
area_summary = pd.DataFrame({
    'Count': area_counts,
    'Percentage': area_percentages.round(1).astype(str) + '%'
})

print(area_summary)
print("\n" + "="*50 + "\n")

# =========================================================
# ANALYSIS 2: AI TECHNOLOGIES (Counts & Percentages)
# =========================================================
print("--- 🤖 AI TECHNOLOGIES BREAKDOWN ---")

# Step A: Split the comma-separated strings (e.g., "Robotics, AI" -> ["Robotics", "AI"])
# We use .dropna() to ignore articles where no AI was found
all_ai_techs = df['Detected_AI'].dropna().str.split(', ').explode()

# Calculate Counts
ai_counts = all_ai_techs.value_counts()

# Calculate Percentages (relative to total mentions)
ai_percentages = all_ai_techs.value_counts(normalize=True) * 100

# Combine into a clean table
ai_summary = pd.DataFrame({
    'Count': ai_counts,
    'Percentage': ai_percentages.round(1).astype(str) + '%'
})

print(ai_summary)

# =========================================================
# OPTIONAL: Save these tables to Excel for your Report
# =========================================================
with pd.ExcelWriter("project_statistics.xlsx") as writer:
    area_summary.to_excel(writer, sheet_name="Civil_Areas")
    ai_summary.to_excel(writer, sheet_name="AI_Techs")
print("\n✅ Statistics saved to 'project_statistics.xlsx'. You can copy these tables into your report.")
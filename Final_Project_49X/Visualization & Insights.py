import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
from collections import Counter
import itertools

# ==========================================
# 1. SETUP & LOAD DATA
# ==========================================
filename = "classified_articles_complete.csv"

try:
    df = pd.read_csv(filename)
    # Convert string representation of list back to actual list if needed
    # (Because CSVs save lists as strings like "['Tech A', 'Tech B']")
    # If your column is just "Tech A, Tech B", simple split works:
    df['AI_List'] = df['Detected_AI'].fillna("").apply(lambda x: x.split(', ') if x else [])
    
    print(f"✅ Loaded {len(df)} articles for visualization.")
except FileNotFoundError:
    print("❌ File not found. Make sure you ran the classification script first.")
    exit()

# ==========================================
# 2. BAR CHART (Rubric: Articles per Civil Area)
# ==========================================
print("Generating Bar Chart...")
plt.figure(figsize=(10, 6))
area_counts = df['Predicted_Area'].value_counts()
sns.barplot(x=area_counts.values, y=area_counts.index, palette='viridis', hue=area_counts.index, legend=False)
plt.title('AI Interest Level: Number of Articles per Discipline', fontsize=15)
plt.xlabel('Number of Articles')
plt.tight_layout()
plt.savefig('task4_bar_chart.png', dpi=300)
plt.show()

# ==========================================
# 3. NETWORK GRAPH (Rubric: Relationships)
# ==========================================
print("Generating Network Graph...")
# We want to link "Civil Areas" to "AI Technologies"
# 1. Create a list of edges: (Area) -> (AI Tech)
edges = []
for index, row in df.iterrows():
    area = row['Predicted_Area']
    for ai_tech in row['AI_List']:
        if ai_tech: # Skip empty
            edges.append((area, ai_tech))

# 2. Count frequency of connections (to weight the lines)
edge_counts = Counter(edges)
# Keep only the top connections to keep the graph readable
most_common_edges = edge_counts.most_common(50) 

# 3. Build the Graph
G = nx.Graph()
for (source, target), weight in most_common_edges:
    G.add_edge(source, target, weight=weight)

# 4. Draw
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=0.5, seed=42) # k regulates the distance between nodes
# Draw nodes
# Differentiate colors: Civil Areas vs AI Techs
node_colors = []
for node in G.nodes():
    if node in df['Predicted_Area'].unique():
        node_colors.append('#ff9999') # Red-ish for Civil Areas
    else:
        node_colors.append('#99ccff') # Blue-ish for AI Techs

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
# Draw edges with varying thickness based on weight
weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, edge_color='gray')

plt.title('Network Graph: Integration of Civil Engineering & AI', fontsize=16)
plt.axis('off')
plt.savefig('task4_network_graph.png', dpi=300)
plt.show()

# ==========================================
# 4. WORD CLOUDS (Rubric: Top words per Sub-discipline)
# ==========================================
print("Generating Word Clouds...")
# Get the top 3 major disciplines to generate clouds for
top_areas = area_counts.head(3).index.tolist()

for area in top_areas:
    # 1. Filter text for this area
    area_text = " ".join(df[df['Predicted_Area'] == area]['cleaned_text'].astype(str))
    
    # 2. Generate Cloud
    # (background_color='white' is better for reports)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(area_text)
    
    # 3. Plot
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud: {area}', fontsize=16)
    filename = f"wordcloud_{area.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.show()

# ==========================================
# 5. FINAL CONCLUSION (Rubric: Rank by AI Maturity)
# ==========================================
print("\n--- 🏆 FINAL RANKING: AI MATURITY ---")
# Metric: "Maturity" = Volume of articles + Diversity of AI used
ranking_data = []

for area in df['Predicted_Area'].unique():
    subset = df[df['Predicted_Area'] == area]
    article_count = len(subset)
    # Count unique AI techs used in this area
    all_techs = list(itertools.chain.from_iterable(subset['AI_List']))
    unique_techs = len(set(all_techs))
    
    ranking_data.append({
        'Area': area,
        'Article_Volume': article_count,
        'Tech_Diversity': unique_techs,
        'Maturity_Score': article_count + (unique_techs * 2) # Arbitrary simple weight
    })

ranking_df = pd.DataFrame(ranking_data).sort_values('Article_Volume', ascending=False)
ranking_df['Rank'] = range(1, len(ranking_df) + 1)

print(ranking_df[['Rank', 'Area', 'Article_Volume', 'Tech_Diversity']])

# Export this for your report
ranking_df.to_excel("final_ai_maturity_ranking.xlsx", index=False)
print("\n✅ All deliverables generated and saved.")
# 🏗️ AEC & AI: Trends, Applications, and Maturity Analysis

This project explores the intersection of **Architecture, Engineering, and Construction (AEC)** with **Artificial Intelligence (AI)**. By analyzing a corpus of **576 articles**, we identify key trends, classify disciplines, and visualize the maturity of AI adoption across the civil engineering sector.

## 📂 Project Structure

The project follows a linear data science pipeline, from data collection to advanced visualization.

| Step | Script | Description | Output File |
|------|--------|-------------|-------------|
| **1. Data Collection** | `1. task1_data_collection.py` | Scrapes 576+ articles from specialized AEC and general tech sources. Filters for relevance. | `aec_ai_corpus.jsonl` |
| **2. NLP & Cleaning** | `task2_nlp_processing.py` | Cleans text, removes stopwords, lemmatizes, and extracts features (Bigrams, TF-IDF). | `aec_ai_corpus_last_full_text_nlp_ready.jsonl` |
| **3. Tagging & Analysis** | `task3_analysis.py` | specific keyword matching to tag Civil disciplines and AI technologies. Generates heatmaps & trends. | `aec_ai_analysis_tagged.jsonl` |
| **4. Classification** | `classifier.py` | Uses **Zero-Shot Classification** (`facebook/bart-large-mnli`) to intelligently categorize articles. | `classified_articles_complete.csv` |
| **5. Visualization** | `Visualization & Insights.py` | Generates final insights: Network Graphs, Word Clouds, and AI Maturity Rankings. | Images & `final_ai_maturity_ranking.xlsx` |

---

## 🚀 Pipeline Details

### 1. Data Collection & Corpus Creation
**Goal:** Build a robust dataset of industry news and technical articles.
- **Sources:** Specialized sites (e.g., *BIMPlus*, *AEC Magazine*) and Tech feeds (*TechCrunch*, *The Verge*).
- **Filtering Logic:**
    - **AEC Sources:** Must contain at least one AI-related term.
    - **General Tech Sources:** Must contain **both** a Civil Engineering term AND an AI term.
- **Result:** A corpus of 576 articles stored in JSON Lines format.

### 2. NLP Preprocessing
**Goal:** Prepare raw text for machine analysis.
- **Techniques Used:**
    - **Spacy Pipeline:** Tokenization, Lemmatization, POS Tagging.
    - **Custom Stopwords:** Removed web-noise (e.g., "subscribe", "cookies", "advertisement").
    - **N-Grams:** Identified top 20 Bigrams (common 2-word phrases).
    - **TF-IDF:** Extracted unique keywords per document.

### 3. Keyword Tagging & Trend Analysis
**Goal:** Map the landscape using deterministic rules.
- **Method:** Dictionary-based tagging.
    - *Civil Tags:* Structural, Geotechnical, Transportation, Construction Mgmt, Environmental.
    - *AI Tags:* Computer Vision, Predictive Analytics, Generative Design, Robotics.
- **Key Insight:** Generated a **Co-occurrence Heatmap** showing which AI techs are most used in which Civil discipline (e.g., *Computer Vision* is heavily linked to *Construction Management* for site monitoring).

### 4. Zero-Shot Classification
**Goal:** Leverage Large Language Models (LLMs) for smarter categorization.
- **Model:** `facebook/bart-large-mnli` (Hugging Face).
- **Process:** Instead of simple keyword matching, the model "reads" the article to determine the **primary** Civil Engineering discipline and **all** relevant AI technologies.
- **Benefit:** Catches nuance that keyword search misses (e.g., understanding "site safety monitoring" implies Computer Vision even if the exact word isn't used).

### 5. Final Visualizations & Maturity Score
**Goal:** Synthesize findings into actionable insights.

#### 📊 Visualizations
- **Bar Chart:** Shows the total volume of articles per discipline (Proxy for "Hype/Interest").
- **Network Graph:** Visualizes the complex web of connections between specific AI tools and Civil sectors. Nodes are colored by type (Red=Civil, Blue=AI).
- **Word Clouds:** Generated for each discipline to show the specific vocabulary used (e.g., "Sustainability" in Environmental vs. "Optimization" in Structural).

#### 🏆 AI Maturity Ranking
We developed a composite "Maturity Score" to rank Civil Engineering disciplines:
$$ \text{Maturity Score} = \text{Article Volume} \times \text{AI Diversity} $$
- **Article Volume:** How much is being written?
- **AI Diversity:** How many *different* types of AI are being applied?

**Top Ranked Disciplines:**
1. **Construction Management** (High volume, diverse use of Vision, Robotics, & Analytics)
2. **Transportation Engineering** (Strong focus on Autonomous Systems & Big Data)
3. **Structural Engineering** (Growing interest in Generative Design)

---

## 🛠️ How to Run

1.  **Install Dependencies:**
    ```bash
    pip install pandas spacy trafilatura scikit-learn seaborn matplotlib networkx wordcloud torch transformers nltk tqdm
    python -m spacy download en_core_web_sm
    ```

2.  **Run Pipeline (Sequential):**
    ```bash
    python "1. task1_data_collection.py"
    python task2_nlp_processing.py
    python task3_analysis.py
    python classifier.py
    python "Visualization & Insights.py"
    ```

3.  **View Outputs:**
    - Check the folder for generated `.png` and `.jpeg` charts.
    - Open `final_ai_maturity_ranking.xlsx` for the detailed scores.

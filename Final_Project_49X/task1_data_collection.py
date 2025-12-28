import requests
import trafilatura
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import time
import random
import json
import logging
import pandas as pd

# --- CONFIGURATION ---
TARGET_COUNT = 550
OUTPUT_FILE = "aec_ai_corpus.jsonl"
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- SOURCE DEFINITIONS ---
# We categorize sources to apply different filter logic
SOURCES = {
    "specialized_aec": [
        # These sites are ALREADY about construction. 
        # Filter: Just look for AI/Tech terms.
        "https://www.bimplus.co.uk/sitemap.xml",
        "https://www.aecmag.com/sitemap_index.xml",
        "https://www.constructiondive.com/feeds/news/",  # RSS Feed (More reliable than sitemap for this site)
        "https://www.enr.com/rss/headlines", # RSS Feed
        "https://globalconstructionreview.com/feed/",
        "https://www.geospatialworld.net/sitemap_index.xml"
    ],
    "general_tech": [
        # These sites are generic. 
        # Filter: Must find (Civil Term AND AI Term).
        "https://techcrunch.com/feed/", 
        "https://venturebeat.com/feed/",
        "https://www.wired.com/feed/category/business/latest/rss",
        "https://www.theverge.com/rss/index.xml"
    ]
}

# --- KEYWORDS ---
CIVIL_TERMS = [
    "construction", "structural", "civil engineer", "concrete", "infrastructure", 
    "bridge", "tunnel", "building information modeling", "bim", "aec", 
    "architecture", "excavation", "safety", "site management", "autodesk", "bentley"
]

AI_TERMS = [
    "artificial intelligence", " ai ", "machine learning", "computer vision", 
    "generative", "neural network", "robot", "automation", "predictive", 
    "digital twin", "llm", "chatgpt", "algorithm", "data analytics"
]

def get_links_from_rss(feed_url):
    """Parses RSS Feeds (Easier/More reliable than Sitemaps for some sites)"""
    links = []
    try:
        resp = requests.get(feed_url, headers={'User-Agent': USER_AGENT}, timeout=10)
        root = ET.fromstring(resp.content)
        # Find all <link> tags in <item>
        for item in root.findall('.//item'):
            link = item.find('link').text
            if link:
                links.append(link)
        logger.info(f"📡 RSS: Found {len(links)} links in {feed_url}")
    except Exception as e:
        logger.warning(f"⚠️ RSS Fail {feed_url}: {e}")
    return links

def get_links_from_sitemap(sitemap_url):
    """Parses XML Sitemaps (including nested indices)"""
    links = []
    try:
        resp = requests.get(sitemap_url, headers={'User-Agent': USER_AGENT}, timeout=10)
        root = ET.fromstring(resp.content)
        
        # Namespaces are annoying in XML, try generic find
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Check if Index
        sitemaps = root.findall('ns:sitemap/ns:loc', ns)
        if sitemaps:
            # It's an index, grab the first 2 sub-sitemaps (usually newest)
            for sm in sitemaps[:2]: 
                links.extend(get_links_from_sitemap(sm.text))
        else:
            # It's a urlset
            urls = root.findall('ns:url/ns:loc', ns)
            for url in urls:
                links.append(url.text)
                
        logger.info(f"🗺️ Sitemap: Found {len(links)} links in {sitemap_url}")
    except Exception as e:
        logger.warning(f"⚠️ Sitemap Fail {sitemap_url}: {e}")
    return links

def is_relevant(text, url, source_type):
    """
    Context-Aware Filtering:
    - If Source is AEC: Accept if it has AI terms.
    - If Source is Tech: Accept ONLY if it has (AI terms AND Civil terms).
    """
    content = (text + " " + url).lower()
    
    has_ai = any(t in content for t in AI_TERMS)
    has_civil = any(t in content for t in CIVIL_TERMS)
    
    if source_type == "specialized_aec":
        return has_ai # Relaxed filter for industry sites
    else:
        return has_ai and has_civil # Strict filter for generic sites

def run_harvester():
    all_links = []
    
    print("--- 1. COLLECTION PHASE ---")
    # 1. Specialized Sources
    for source in SOURCES["specialized_aec"]:
        if "feed" in source or "rss" in source:
            links = get_links_from_rss(source)
        else:
            links = get_links_from_sitemap(source)
        # Tag them so we know which filter to use later
        for l in links: all_links.append((l, "specialized_aec"))
            
    # 2. General Tech Sources
    for source in SOURCES["general_tech"]:
        if "feed" in source or "rss" in source:
            links = get_links_from_rss(source)
        else:
            links = get_links_from_sitemap(source)
        for l in links: all_links.append((l, "general_tech"))

    # Remove duplicates
    unique_links = list(set(all_links))
    print(f"✅ Unique URLs Candidate Pool: {len(unique_links)}")
    
    print("\n--- 2. EXTRACTION PHASE ---")
    collected_count = 0
    
    # Open file in Append mode
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for url, source_type in unique_links:
            if collected_count >= TARGET_COUNT:
                break
                
            try:
                # Polite delay
                time.sleep(random.uniform(0.5, 1.2))
                
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    # Extract
                    result = trafilatura.extract(downloaded, output_format="json", with_metadata=True, include_comments=False)
                    if result:
                        doc = json.loads(result)
                        full_text = doc.get('text', '')
                        
                        # Apply Smart Filter
                        if is_relevant(full_text, url, source_type):
                            entry = {
                                "url": url,
                                "title": doc.get('title'),
                                "date": doc.get('date'),
                                "source_type": source_type,
                                "text": full_text
                            }
                            
                            # Write Line
                            f.write(json.dumps(entry) + "\n")
                            collected_count += 1
                            print(f"[{collected_count}/{TARGET_COUNT}] Saved: {doc.get('title')[:50]}...")
                        else:
                            # Optional: Print why it failed (comment out to reduce noise)
                            # print(f"Skipped (Irrelevant): {url}")
                            pass

            except Exception as e:
                # print(f"Error on {url}: {e}")
                continue

    print(f"\n🎉 Finished! Collected {collected_count} articles in {OUTPUT_FILE}")

if __name__ == "__main__":
    run_harvester()

df = pd.read_json("aec_ai_corpus.jsonl", lines=True)
print(df['source_type'].value_counts())
import os
import json
import pandas as pd
import re
from bs4 import BeautifulSoup

# 📂 Path where your JSON files are stored
data_dir = r"C:\Users\t_pal\Documents\PYTHON\Minor\lawcases"   # change this to your actual folder path

records = []

for filename in os.listdir(data_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # ---------------- Extract Document Text ----------------
            raw_html = data.get("doc", "")
            if not raw_html:
                continue

            soup = BeautifulSoup(raw_html, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text)

            # ---------------- Extract Citations ----------------
            citations_html = re.findall(
                r"<h3 class=\"doc_citations\">(.*?)</h3>", raw_html, re.DOTALL
            )
            citations_list = []
            if citations_html:
                citations_text = BeautifulSoup(citations_html[0], "html.parser").get_text()
                # Split by commas, semicolons, or newlines
                citations_list = [c.strip() for c in re.split(r"[,\n;]", citations_text) if c.strip()]

            # ---------------- Optional: Extract Sections of Law ----------------
            # e.g. “Section 302 IPC”, “u/s 420 of the IPC”, etc.
            sections = re.findall(r"(Section\s+\d+[A-Z]*|u/s\s*\d+[A-Z]*)", text, re.IGNORECASE)
            sections = [s.strip() for s in sections if s.strip()]

            # Merge citations + sections
            all_citations = list(set(citations_list + sections))  # remove duplicates

            if not all_citations:
                continue  # skip if no citation or section found

            # ---------------- Add to Records ----------------
            records.append({
                "doc_body": text,
                "citations_applied": all_citations
            })

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

# ---------------- Save to CSV ----------------
df = pd.DataFrame(records)
df.to_csv("citations_dataset.csv", index=False, encoding="utf-8")

print(f"✅ Saved {len(df)} cleaned cases with citations and sections to citations_dataset.csv")
print(df.head())

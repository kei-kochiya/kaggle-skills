import json
import os
import sys
import re

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

manifest_path = os.path.join("docs", "raw_writeups", "manifest.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    items = json.load(f)

def clean_html(text):
    text = re.sub(r'</?(p|li|h[1-6]|ul|ol|table|tr|th|td|blockquote)[^>]*>', '\n', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

# Let's inspect Topic 749637 (3rd place DSM full writeup)
dsm = next((x for x in items if x["id"] == 749637), None)
if dsm:
    with open(dsm["json_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    content = clean_html(data["messages"][0]["content"])
    print("=== DSM 3rd PLACE FULL WRITEUP SECTIONS ===")
    # Print sections 4, 5, 6, 7
    for sec in ["4. Data and preprocessing", "5. Model", "6. Training", "7. Inference-time post-processing", "8. Engines and evaluation", "10. What we expected"]:
        idx = content.find(sec)
        if idx != -1:
            print(f"\n--- {sec} ---")
            print(content[idx:idx+1500])

import json
import os
import sys
import re

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

manifest_path = os.path.join("docs", "raw_writeups", "manifest.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    items = json.load(f)

key_ids = [
    745073, 751369, 749637, 745119, 750068, 750159, 752035, 751681, 745367, 
    750403, 750617, 745258, 749586, 750220, 751866, 749669, 745783, 749775, 
    750212, 750151
]

def clean_html(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

for tid in key_ids:
    item = next((it for it in items if it["id"] == tid), None)
    if not item:
        continue
    json_path = item["json_path"]
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    title = item["title"]
    messages = data.get("messages", [])
    print(f"\n========================================================")
    print(f"TOPIC {tid}: {title}")
    print(f"Messages count: {len(messages)}")
    if messages:
        content = messages[0].get("content", "")
        print(f"Length: {len(content)} chars")
        # Extract headings or sections
        h_tags = re.findall(r'<h[1-4]>(.*?)</h[1-4]>', content)
        if h_tags:
            print("Headings:", h_tags)
        else:
            # check markdown headers
            md_headers = re.findall(r'#+\s*(.*)', content)
            if md_headers:
                print("MD Headers:", md_headers[:10])
        clean = clean_html(content)
        print("Summary snippet:", clean[:300], "...")

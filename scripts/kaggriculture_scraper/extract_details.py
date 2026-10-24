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

check_ids = [752035, 751681, 750403, 750617, 750220, 751866, 749775, 750212, 749586, 750068]

for tid in check_ids:
    it = next((x for x in items if x["id"] == tid), None)
    if not it:
        continue
    with open(it["json_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    print("=" * 80)
    print(f"TOPIC {tid}: {it['title']}")
    print("=" * 80)
    messages = data.get("messages", [])
    if messages:
        c = clean_html(messages[0].get("content", ""))
        print(c[:2500])
        print("\n... [TRUNCATED] ...\n")
        if len(c) > 2500:
            print(c[-1000:])

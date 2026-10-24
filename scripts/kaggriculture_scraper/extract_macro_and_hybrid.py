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

for tid in [750151, 749775]:
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
        lines = c.split('\n')
        print(f"Total lines: {len(lines)}")
        for l in lines[:50]:
            if l.strip():
                print(l[:120])
        print("\n--- MIDDLE SNIPPET ---")
        for l in lines[100:150]:
            if l.strip():
                print(l[:120])

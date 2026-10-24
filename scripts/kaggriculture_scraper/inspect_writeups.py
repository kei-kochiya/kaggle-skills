import json
import os

manifest_path = os.path.join("docs", "raw_writeups", "manifest.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    items = json.load(f)

print(f"Total items in manifest: {len(items)}\n")
for item in items:
    print(f"ID: {item['id']} | Msgs: {item['message_count']} | Title: {item['title']}")

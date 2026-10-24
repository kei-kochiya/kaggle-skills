import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

manifest_path = os.path.join("docs", "raw_writeups", "manifest.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    items = json.load(f)

# Priority gold medal / top writeups:
# 1st: 745073
# 3rd: 751369 & 749637
# 4th: 745119 & 750068
# 6th: 752035
# 7th: 751681 & 745367
# 9th: 750403
# 10th: 750617 & 745258
# 19th: 750220
# 22nd: 751866
# 26th: 749775 & 745783
# 27th: 750212
# 28th: 750151
# Key themes: 750159 ("SFT all the way down"), 749586 ("Under $60 compute"), 749669 ("Colab")

key_ids = [
    745073, 751369, 749637, 745119, 750068, 750159, 752035, 751681, 745367, 
    750403, 750617, 745258, 749586, 750220, 751866, 749669, 745783, 749775, 
    750212, 750151
]

for tid in key_ids:
    item = next((it for it in items if it["id"] == tid), None)
    if not item:
        continue
    json_path = item["json_path"]
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n{'='*70}")
    print(f"TOPIC {tid}: {item['title']}")
    print(f"{'='*70}")
    messages = data.get("messages", [])
    if messages:
        # Print first message (the main writeup)
        first_msg = messages[0].get("content", "")
        # Strip some html tags or print lines
        lines = first_msg.splitlines()
        print(f"Length of writeup: {len(first_msg)} chars, {len(lines)} lines")
        print("\n--- FIRST 40 LINES ---")
        for line in lines[:40]:
            print(line[:120])
        if len(lines) > 40:
            print("\n--- LAST 20 LINES ---")
            for line in lines[-20:]:
                print(line[:120])
    else:
        print("NO MESSAGES")

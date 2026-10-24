import json
import subprocess
import os
import re
import sys

# Force utf-8 stdout
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open("all_topics.json", "r", encoding="utf-8") as f:
    topics = json.load(f)

# The target topics from user & discoveries
target_map = {
    745073: "1st-place-preview-m-m-p-q",
    751369: "3rd-place-dsm-solution",
    749637: "3rd-place-dsm-commentary",
    745119: "4th-place-one-pass-transformer",
    750068: "farmcore-one-pass-transformer",
    750159: "its-sft-all-the-way-down",
    752035: "6th-place-write-up",
    751681: "7th-place-decem-bc-plus-rules",
    745367: "7th-place-preview-decem",
    750403: "9th-place-from-rl-to-bc",
    750617: "10th-place-bc-ppo-single-rtx-5090",
    745258: "10th-place-preview-lejepa",
    749586: "gold-medal-under-60-compute",
    750220: "19th-place-solo-gold-bc-of-top-teams",
    751866: "22nd-place-bc-ppo-diverse-opponents",
    749669: "cloning-best-farmer-on-colab-rtx-5070",
    745783: "24th-31st-place-macro-bc-day-compiler",
    749775: "26th-place-gold-macro-bc-day-compiler",
    750212: "27th-place-gold-monsaraida",
    750151: "28th-place-gold-how-we-taught-a-farm",
}

# Scan for any additional place/solution writeups in the topics list
for t in topics:
    tid = t["id"]
    title = t.get("title", "")
    t_lower = title.lower()
    if tid not in target_map:
        if any(w in t_lower for w in ["place", "solution", "write-up", "writeup", "gold"]):
            if not any(skip in t_lower for skip in ["beware", "scammer", "comment on", "violation", "servers", "overloaded"]):
                slug = re.sub(r'[^a-z0-9]+', '-', t_lower).strip('-')[:40]
                target_map[tid] = f"extra-{slug}"

print(f"Total topics to download: {len(target_map)}")

out_dir = os.path.join("docs", "raw_writeups")
os.makedirs(out_dir, exist_ok=True)

results = []

for tid, slug in target_map.items():
    json_path = os.path.join(out_dir, f"{tid}_{slug}.json")
    md_path = os.path.join(out_dir, f"{tid}_{slug}.md")
    
    t_meta = next((t for t in topics if t["id"] == tid), {})
    title = t_meta.get("title", slug)
    
    # Check if already downloaded
    if os.path.exists(json_path) and os.path.getsize(json_path) > 100:
        print(f"Already downloaded: {tid}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            messages = data.get("messages", [])
    else:
        print(f"Fetching topic {tid}...")
        cmd = f'kaggle competitions topic-messages kaggriculture {tid} --format json'
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        out = res.stdout
        
        idx = out.find('[')
        end_idx = out.rfind(']')
        messages = []
        if idx != -1 and end_idx != -1:
            try:
                messages = json.loads(out[idx:end_idx+1])
            except Exception as e:
                print(f"  Error parsing JSON for {tid}: {e}")
                
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"topic_id": tid, "metadata": t_meta, "messages": messages}, f, indent=2)
            
        with open(md_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(f"# Topic {tid}: {title}\n\n")
            f.write(f"- **URL**: https://www.kaggle.com/competitions/kaggriculture/discussion/{tid}\n")
            f.write(f"- **Author**: {t_meta.get('authorName', 'Unknown')}\n")
            f.write(f"- **Post Date**: {t_meta.get('postDate', '')}\n")
            f.write(f"- **Votes**: {t_meta.get('votes', 0)}\n\n")
            f.write("## Messages\n\n")
            for i, m in enumerate(messages):
                f.write(f"### Message {i+1} by {m.get('authorName', 'Author')} (Votes: {m.get('votes', 0)})\n\n")
                f.write(f"{m.get('content', '')}\n\n---\n\n")
                
    results.append({
        "id": tid,
        "title": title,
        "slug": slug,
        "message_count": len(messages),
        "json_path": json_path,
        "md_path": md_path
    })

with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nAll writeups successfully downloaded and saved in docs/raw_writeups/!")

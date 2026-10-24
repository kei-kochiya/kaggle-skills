import json
import re
import subprocess
import os

with open("all_topics.json", "r", encoding="utf-8") as f:
    topics = json.load(f)

user_targets = [
    {"target": "745073", "desc": "1st place initial discussion"},
    {"target": "kaggriculture-3rd-place-dsm-solution", "desc": "3rd place DSM"},
    {"target": "farmcore-a-one-pass-transformer-trained-on-what-t", "desc": "Farmcore one-pass transformer"},
    {"target": "its-sft-all-the-way-down", "desc": "It's SFT all the way down"},
    {"target": "6th-place-write-up", "desc": "6th place write-up"},
    {"target": "7th-place-my-solution-only-behavior-cloning-t", "desc": "7th place DECEM only BC"},
    {"target": "from-reinforcement-learning-to-a-final-bc-agent", "desc": "From RL to final BC agent"},
    {"target": "10th-place-bc-ppo-on-a-single-rtx-5090", "desc": "10th place BC+PPO on RTX 5090"},
    {"target": "gold-medalist-with-70-of-compute", "desc": "Gold medalist with 70% of compute"},
    {"target": "19th-place-solo-gold-a-behaviour-clone", "desc": "19th place solo gold a behaviour clone"},
    {"target": "22nd-place-behavior-cloning-and-ppo-with-diverse", "desc": "22nd place BC and PPO with diverse"},
    {"target": "cloning-the-best-farmer-on-one-rtx-5070", "desc": "Cloning the best farmer on one RTX 5070"},
    {"target": "31th-place-currently-our-writeup-macro-bc-d", "desc": "31st place macro BC"},
    {"target": "monsaraida-27th-place-solution", "desc": "monsaraida 27th place solution"},
    {"target": "28th-place-gold-solution-how-we-taught-a-farm-to", "desc": "28th place gold solution"}
]

print(f"Total topics indexed: {len(topics)}")

matched_topics = []

for item in user_targets:
    target = item["target"]
    matched = None
    
    # Check exact ID
    for t in topics:
        if str(t.get("id")) == target:
            matched = t
            break
            
    if not matched:
        # Match keywords
        words = [w for w in re.split(r'[-_\s]+', target.lower()) if len(w) > 2 and w not in ["the", "and", "with", "for", "solution", "place", "writeup", "write", "our"]]
        best_score = 0
        best_t = None
        for t in topics:
            title = t.get("title", "").lower()
            score = sum(1 for w in words if w in title)
            if score > best_score:
                best_score = score
                best_t = t
        if best_score >= 2 or (len(words) == 1 and best_score >= 1):
            matched = best_t
            
    if matched:
        print(f"MATCH: '{target}' -> ID: {matched['id']} | Title: '{matched['title']}'")
        matched_topics.append({"target": target, "desc": item["desc"], "topic": matched})
    else:
        print(f"NO MATCH: '{target}'")

# Save matched targets
with open("matched_targets.json", "w", encoding="utf-8") as f:
    json.dump(matched_topics, f, indent=2)

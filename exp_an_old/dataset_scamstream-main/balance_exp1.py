import json
import random

random.seed(42)

files = [
    r"exp1_prompt_vs_viscam\train\Vanilla.json",
    r"exp1_prompt_vs_viscam\train\Viscam.json",
]

# Load all files first to compute global min per label
all_data = {}
for path in files:
    with open(path, "r", encoding="utf-8") as f:
        all_data[path] = json.load(f)

n_scam     = min(len([d for d in v if d["label"] == "scam"])    for v in all_data.values())
n_harmless = min(len([d for d in v if d["label"] == "harmless"]) for v in all_data.values())
print(f"Global min: scam={n_scam}, harmless={n_harmless}\n")

for path, data in all_data.items():
    scam     = [d for d in data if d["label"] == "scam"]
    harmless = [d for d in data if d["label"] == "harmless"]

    print(f"{path}")
    print(f"  Before: scam={len(scam)}, harmless={len(harmless)}")

    random.shuffle(scam)
    random.shuffle(harmless)
    balanced = scam[:n_scam] + harmless[:n_harmless]
    random.shuffle(balanced)

    print(f"  After:  scam={n_scam}, harmless={n_harmless}, total={len(balanced)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(balanced, f, ensure_ascii=False, indent=2)

    print(f"  Saved.\n")

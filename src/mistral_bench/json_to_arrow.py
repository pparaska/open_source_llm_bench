import json
from datasets import Dataset

# Load filtered JSON
with open("D:\mistral-bench-compat\src\mistral_bench\compact.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to Hugging Face Dataset
ds = Dataset.from_list(data)

# Save in Arrow dataset format
ds.save_to_disk("D:/mistral-bench-compat/data/hotpotqa/hotpotqa_1")

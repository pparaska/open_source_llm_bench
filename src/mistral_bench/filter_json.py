import json

# IDs to keep
keep_ids = {
    "3dd1ade26fd583844dd2af051d61eb20856ccc8e23abae59",
    "ab016faac478c4acc6e9ed3ed87f9e29dbed47fd946be244",
    "3a75aba6bc2496016339474aef28ffdc9dbebdc00ccc166b",
    "b6352c61b4a748448ce38882861cd5ae5f7f2869a81e92a1",
    "670498f9d7fd2582c5791cad2fe212f580d316fa49a8a984",
    "de85aaa5c535ba20bdcad5fbabfe13a03687374265a57b44",
    "f536646dd72a8a4f5ebddec61a67b8319dc9187d47bb7fc6",
    "5e7675388334946fc508c4c8ac4ab0c491910fcc0e048bfd",
    "a7c311f9f6423063e02c8236bdd83cc8059fcb9fe78b4fff",
    "4d6aafffd109cb61fd543f44bda770fda0cf1e1c3798548b",
    "1fd7df3ba224e985f87e976c7a4a67ddc1012a249a542bea",
    "3faa54c1f192293c709684aa60c6e2e8cfe34ff55c8b6248",
    "d7d92c85b8becc33e60fc03507933f8eaf53b5e1067286de",
    "78e73e15102e688a5b5f43155b11658907db8caa30fdb16e",
    "302ee593ca790dfb2dcbfd4341817fc64f8d106f65f61867",
    "3acb0cc0a04a3f71078ee120e1b7ca4f089548b7cc8a8056",
    "90caafc7fe51eee4143874971f67bb3e0107d26179c437f7",
    "c9f8b2166f7c44bf2e0843b2c77fb9fa5738af275555ca73",
    "4567b6935bf5c64a1974b1089d67d176d9acac6b29836cb7",
    "c274ce731f680eb107c70386e2e341615378165ecc22799e",
    "c1cfe7334235bf4a9ef11293d8be2d4055a70ac89334be2c",
    "d3b197d5b4da1198fc9c232524faced0f528b89f919b8061",
    "4670f2a542ac06d27e47a74a042cf57f743fba34722a7dc5",
    "5da42038bb9d29bb30e4094504ad06dc2d43afa373568686"


}

# Input and output file names
input_file = "D:\mistral-bench-compat\src\mistral_bench\outpt.json"   # your original file
output_file = "filtered.json"

# Load JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter based on 'id' field
filtered_data = [entry for entry in data if entry.get("_id") in keep_ids]

# Save filtered data
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print(f"Filtered {len(filtered_data)} entries saved to {output_file}")

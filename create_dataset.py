import json
import os
import shutil
import random

json_path = "D:\\IP_Project\\annotations.json"
image_folder = "D:\\IP_Project\\images"

# 1. Define your target species (5 classes x 100 images = 500 images total)
target_species = ["bobcat", "coyote", "raccoon", "deer", "empty"]

# 2. Create specific folders for our targets
for species in target_species:
    os.makedirs(f"dataset/{species}", exist_ok=True)

with open(json_path) as f:
    data = json.load(f)

# 3. Map category IDs to their specific names
cat_map = {cat["id"]: cat["name"] for cat in data["categories"]}

# Match image IDs to their actual specific species name
image_labels = {}
for ann in data["annotations"]:
    img_id = ann["image_id"]
    category_name = cat_map.get(ann["category_id"], "unknown")
    image_labels[img_id] = category_name

id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

# Group image paths by specific species
species_images = {species: [] for species in target_species}

for img_id, label in image_labels.items():
    if label in target_species and img_id in id_to_file:
        path = os.path.join(image_folder, id_to_file[img_id])
        if os.path.exists(path):
            species_images[label].append(path)

# 4. Sample exactly 100 from each to get a perfect 500-image subset
for species in target_species:
    # Adding a safeguard in case a specific category has slightly less than 100 images
    sample_size = min(100, len(species_images[species]))
    sample = random.sample(species_images[species], sample_size)
    
    for i, p in enumerate(sample):
        shutil.copy(p, f"dataset/{species}/{i}.jpg")

print(f"Multi-class dataset created successfully across {len(target_species)} categories!")
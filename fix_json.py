import json

# Path to the incorrect JSON file
json_path = "/home/fahadabul/mask_rcnn_skyhub/final_images/val_annotations (copy).json"

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Fix segmentation format
for ann in data["annotations"]:
    if "segmentation" in ann:
        segm = ann["segmentation"]
        if isinstance(segm, list) and len(segm) > 0 and isinstance(segm[0], (int, float)):
            # Convert flat list to nested list (COCO polygon format)
            print(ann["id"])
            ann["segmentation"] = [segm]

# Save the fixed JSON
fixed_json_path = "/home/fahadabul/mask_rcnn_skyhub/final_images/val_annotations_fixed.json"
with open(fixed_json_path, "w") as f:
    json.dump(data, f, indent=4)

print(f"Fixed JSON saved at: {fixed_json_path}")

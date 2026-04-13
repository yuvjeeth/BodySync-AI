from datasets import load_dataset
import os

dataset = load_dataset("openfoodfacts/nutrition-table-detection")

os.makedirs("dataset/images/train", exist_ok=True)
os.makedirs("dataset/images/val", exist_ok=True)
os.makedirs("dataset/labels/train", exist_ok=True)
os.makedirs("dataset/labels/val", exist_ok=True)


def clip(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def convert(split, save_split):
    for i, item in enumerate(dataset[split]):
        img = item["image"]
        boxes = item["objects"]["bbox"]
        w, h = img.size

        img_path = f"dataset/images/{save_split}/{i}.jpg"
        label_path = f"dataset/labels/{save_split}/{i}.txt"

        img.save(img_path)

        with open(label_path, "w") as f:
            for box in boxes:
                # OpenFoodFacts format: (y_min, x_min, y_max, x_max)
                y_min, x_min, y_max, x_max = box

                # Handle both normalized and pixel annotations.
                if max(y_min, x_min, y_max, x_max) > 1.0:
                    y_min /= h
                    y_max /= h
                    x_min /= w
                    x_max /= w

                y_min = clip(y_min)
                x_min = clip(x_min)
                y_max = clip(y_max)
                x_max = clip(x_max)

                left, right = sorted((x_min, x_max))
                top, bottom = sorted((y_min, y_max))

                width = clip(right - left)
                height = clip(bottom - top)
                x_center = clip((left + right) / 2)
                y_center = clip((top + bottom) / 2)

                # Skip invalid boxes.
                if width > 0 and height > 0:
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")


convert("train", "train")
convert("val", "val")

print("dataset ready")
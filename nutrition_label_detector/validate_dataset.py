from pathlib import Path
import statistics


def parse_simple_yaml(yaml_path: Path) -> dict:
    data = {}
    names = {}
    in_names = False

    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("names:"):
            in_names = True
            continue

        if in_names and ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k.isdigit():
                names[int(k)] = v
            continue

        if in_names and not raw_line.startswith(" "):
            in_names = False

        if not in_names and ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip().strip('"').strip("'")

    data["names"] = names
    return data


def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in exts])


def list_labels(folder: Path):
    return sorted(folder.glob("*.txt"))


def check_split(root: Path, split_name: str, names_count: int):
    images_dir = root / "images" / split_name
    labels_dir = root / "labels" / split_name

    print(f"\n[{split_name}]")
    print(f"images_dir: {images_dir}")
    print(f"labels_dir: {labels_dir}")

    if not images_dir.exists() or not labels_dir.exists():
        print("ERROR: split folders missing")
        return False

    images = list_images(images_dir)
    labels = list_labels(labels_dir)
    print(f"images: {len(images)} | labels: {len(labels)}")

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    if missing_labels:
        print(f"ERROR: {len(missing_labels)} images have no label file")
    if missing_images:
        print(f"ERROR: {len(missing_images)} label files have no image")

    invalid_lines = 0
    out_of_range = 0
    tiny_boxes = 0
    empty_files = 0
    areas = []

    for label_file in labels:
        text = label_file.read_text(encoding="utf-8").strip()
        if not text:
            empty_files += 1
            continue

        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                invalid_lines += 1
                continue

            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                invalid_lines += 1
                continue

            if cls < 0 or cls >= names_count:
                out_of_range += 1

            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                out_of_range += 1

            area = w * h
            areas.append(area)
            if area < 1e-4:
                tiny_boxes += 1

    print(f"empty label files: {empty_files}")
    print(f"invalid label lines: {invalid_lines}")
    print(f"out-of-range labels: {out_of_range}")

    if areas:
        print(f"box area min/median/max: {min(areas):.6g} / {statistics.median(areas):.6g} / {max(areas):.6g}")
        print(f"tiny boxes (area < 1e-4): {tiny_boxes}")

    ok = (
        not missing_labels
        and not missing_images
        and invalid_lines == 0
        and out_of_range == 0
        and (len(areas) == 0 or statistics.median(areas) >= 1e-4)
    )

    if ok:
        print("STATUS: OK")
    else:
        print("STATUS: CHECK FAILED")

    return ok


def main():
    base = Path(__file__).resolve().parent
    yaml_path = base / "data.yaml"

    if not yaml_path.exists():
        print(f"ERROR: missing {yaml_path}")
        return

    cfg = parse_simple_yaml(yaml_path)
    dataset_root = base / cfg.get("path", "dataset")
    names_count = max(cfg.get("names", {0: "class0"}).keys(), default=0) + 1

    print("=== Dataset Pre-Training Validation ===")
    print(f"yaml: {yaml_path}")
    print(f"dataset root: {dataset_root}")
    print(f"class count: {names_count}")

    train_ok = check_split(dataset_root, "train", names_count)
    val_ok = check_split(dataset_root, "val", names_count)

    print("\n=== Final Verdict ===")
    if train_ok and val_ok:
        print("PASS: dataset looks valid for training")
    else:
        print("FAIL: fix dataset/labels before training")


if __name__ == "__main__":
    main()

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def get_classes(json_path: str = "info_semantic.json"):
    classes = []
    with open(json_path, "r") as f:
        data = json.load(f)

    for obj in data["classes"]:
        classes.append(obj["name"])
    return classes


def get_npz_files_from_dir(directory: str, recursive: bool = True):
    """Zwraca tylko pliki .npz (ignoruje katalogi i inne rozszerzenia)."""
    directory = Path(directory)

    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Podana ścieżka nie jest katalogiem: {directory}")

    it = directory.rglob("*.npz") if recursive else directory.glob("*.npz")
    files = [p for p in it if p.is_file()]
    return sorted(files)


def count_classes_in_npz(npz_files, classes):
    num_classes = len(classes)
    counts = defaultdict(int)

    for npz_path in npz_files:
        # Dodatkowa ochrona (na wszelki wypadek)
        if not npz_path.is_file():
            print(f"[SKIP] To nie jest plik: {npz_path}")
            continue

        try:
            data = np.load(npz_path)
        except Exception as e:
            print(f"[WARN] Nie można wczytać {npz_path}: {e}")
            continue

        if "labels" not in data.files:
            print(f"[WARN] Brak labels.npy w {npz_path} (dostępne: {data.files})")
            continue

        labels = data["labels"]

        # zliczamy wszystkie elementy niezależnie od kształtu
        labels = np.asarray(labels).reshape(-1)

        for label in labels:
            label = int(label)
            if 0 <= label < num_classes:
                counts[label] += 1
            else:
                counts["unknown"] += 1

    return counts


def main():
    if len(sys.argv) < 2:
        print("Użycie:")
        print("  python count_labels.py <ścieżka_do_folderu> [--no-recursive] [--json info_semantic.json]")
        sys.exit(1)

    folder = sys.argv[1]
    recursive = True
    json_path = "info_semantic.json"

    # proste parsowanie flag
    args = sys.argv[2:]
    if "--no-recursive" in args:
        recursive = False
    if "--json" in args:
        idx = args.index("--json")
        if idx + 1 >= len(args):
            print("[ERROR] Po --json podaj ścieżkę do pliku JSON")
            sys.exit(1)
        json_path = args[idx + 1]

    classes = get_classes(json_path)

    try:
        npz_files = get_npz_files_from_dir(folder, recursive=recursive)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not npz_files:
        print(f"[WARN] Nie znaleziono żadnych plików .npz w: {folder} (recursive={recursive})")
        sys.exit(0)

    print(f"[INFO] Znaleziono {len(npz_files)} plików .npz (recursive={recursive})")

    counts = count_classes_in_npz(npz_files, classes)

    print("\n=== Liczba wystąpień klas ===")
    for idx, name in enumerate(classes):
        print(f"{name:30s} : {counts.get(idx, 0)}")

    if "unknown" in counts:
        print(f"\n[INFO] Nieznane etykiety (poza zakresem klas): {counts['unknown']}")


if __name__ == "__main__":
    main()

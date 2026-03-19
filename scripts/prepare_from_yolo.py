import os
import cv2
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import Counter


# Параметры
BASE_DIR = Path(__file__).parent.parent.absolute()
INPUT_ROOT = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_ROOT = os.path.join(BASE_DIR, "data", "processed")
TARGET_SIZE = (224, 224)

if not os.path.exists(INPUT_ROOT):
    print(f"❌ Папка {INPUT_ROOT} не существует!")
    print("Убедитесь, что структура data/raw/... создана")
    sys.exit(1)

# Соответствие class_id -> имя класса
CLASS_NAMES = {1: "without_mask", 0: "with_mask"}

# Создаём структуру папок
for split in ["train", "val", "test"]:
    for cls_name in CLASS_NAMES.values():
        os.makedirs(os.path.join(OUTPUT_ROOT, split, cls_name), exist_ok=True)

# Соответствие папок и сплитов
split_map = {
    "_training_set": "train",
    "_validation_set": "val",
    "_test_set": "test"
}

# Статистика по каждому сплиту
stats = {split: {"with_mask": 0, "without_mask": 0} for split in ["train", "val", "test"]}

for raw_split, target_split in split_map.items():
    raw_path = os.path.join(INPUT_ROOT, raw_split)
    if not os.path.isdir(raw_path):
        print(f"⚠️ Папка {raw_split} не найдена, пропускаем")
        continue

    print(f"\n📁 Обработка {raw_split} -> {target_split}")
    
    png_files = list(Path(raw_path).glob("*.png"))
    print(f"   Найдено PNG файлов: {len(png_files)}")
    
    for png_path in tqdm(png_files, desc=f"Обработка {raw_split}"):
        txt_path = png_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"   ⚠️ Нет аннотации для {png_path.name}")
            continue

        image = cv2.imread(str(png_path))
        if image is None:
            print(f"   ⚠️ Ошибка загрузки {png_path.name}")
            continue
        h, w = image.shape[:2]

        with open(txt_path, "r") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            if class_id not in CLASS_NAMES:
                continue
            class_name = CLASS_NAMES[class_id]

            # Нормализованные координаты
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            bbox_width = float(parts[3]) * w
            bbox_height = float(parts[4]) * h

            x1 = int(x_center - bbox_width / 2)
            y1 = int(y_center - bbox_height / 2)
            x2 = int(x_center + bbox_width / 2)
            y2 = int(y_center + bbox_height / 2)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            face = image[y1:y2, x1:x2]
            face_resized = cv2.resize(face, TARGET_SIZE)

            out_filename = f"{png_path.stem}_face{idx:03d}.jpg"
            out_path = os.path.join(OUTPUT_ROOT, target_split, class_name, out_filename)
            cv2.imwrite(out_path, face_resized)
            
            # Считаем статистику
            stats[target_split][class_name] += 1

# Выводим итоговую статистику
print("\n" + "="*50)
print("ИТОГОВАЯ СТАТИСТИКА ДАТАСЕТА")
print("="*50)
for split in ["train", "val", "test"]:
    total = stats[split]["with_mask"] + stats[split]["without_mask"]
    print(f"\n{split.upper()}: всего {total} изображений")
    print(f"  with_mask: {stats[split]['with_mask']} ({stats[split]['with_mask']/total*100:.1f}%)")
    print(f"  without_mask: {stats[split]['without_mask']} ({stats[split]['without_mask']/total*100:.1f}%)")

print("\n✅ Конвертация завершена!")
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import glob

# Загрузка модели
model = load_model('defect_detection_continued.h5')
print("✅ Модель загружена")

# Сканируем директорию на наличие фото
photo_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
image_files = []

for ext in photo_extensions:
    image_files.extend(glob.glob(ext))

print(f"📁 Найдено фото: {len(image_files)}")

# Анализ каждого фото
results = {}

for image_path in image_files:
    try:
        # Загрузка и обработка изображения
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Предсказание
        prediction = model.predict(img_array, verbose=0)
        defect_prob = float(prediction[0][0])

        # Определяем результат
        if defect_prob >= 0.5:
            status = "defect"
        else:
            status = "not_defect"

        results[image_path] = status
        print(f"🔍 {image_path}: {status} ({defect_prob:.3f})")

    except Exception as e:
        print(f"❌ Ошибка с {image_path}: {e}")
        results[image_path] = "error"

# Выводим итог
print("\n📊 ИТОГ:")
for photo, result in results.items():
    print(f"{photo}: {result}")

# Переменная results содержит все результаты
# Пример: {'photo1.jpg': 'defect', 'photo2.jpg': 'not_defect'}
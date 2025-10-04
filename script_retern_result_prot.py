import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Загрузка модели (делается один раз при импорте)
model = load_model('defect_detection_continued.h5')
print("✅ Модель загружена")


def analyze_photo(image_path):
    """
    Анализирует фото на наличие дефектов

    Args:
        image_path (str): путь к фотографии

    Returns:
        str: 'defect' или 'not_defect'
    """
    try:
        # Проверяем существование файла
        if not os.path.exists(image_path):
            print(f"❌ Файл {image_path} не найден")
            return "error"

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
            result = "defect"
        else:
            result = "not_defect"

        print(f"🔍 {image_path}: {result} ({defect_prob:.3f})")
        return result

    except Exception as e:
        print(f"❌ Ошибка с {image_path}: {e}")
        return "error"


# ПРИМЕР использования функции:
if __name__ == "__main__":
    # Тестируем на конкретном фото
    result = analyze_photo('DSC_2760.jpg')
    print(f"Результат: {result}")

# Просто вызываешь функцию с путем к фото
result = analyze_photo('DSC_2715.jpg')
# result будет содержать 'defect' или 'not_defect'

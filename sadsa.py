import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os


def load_defect_model(model_path='defect_model.h5'):
    """Загрузка модели один раз при старте программы"""
    try:
        model = load_model(model_path)
        print("Модель успешно загружена")
        return model
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None


def predict_defects(image_path, model, img_size=(224, 224)):
    """
    Основная функция для предсказания дефектов

    Args:
        image_path (str): путь к изображению
        model: загруженная модель
        img_size (tuple): размер изображения для модели

    Returns:
        np.array: результат предсказания
    """
    try:
        # Загрузка и предобработка изображения
        img = Image.open(image_path)

        # Конвертация в RGB если нужно
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Изменение размера
        img = img.resize(img_size)

        # Конвертация в numpy array и нормализация
        img_array = np.array(img) / 255.0

        # Добавление размерности батча
        img_array = np.expand_dims(img_array, axis=0)

        # Предсказание
        prediction = model.predict(img_array)

        return prediction

    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None


# ИНИЦИАЛИЗАЦИЯ (делается один раз при старте)
model = load_defect_model('defect_detection_continued.h5')

# ИСПОЛЬЗОВАНИЕ В ВАШЕМ КОДЕ
if model is not None:
    # Пример вызова функции
    result = predict_defects('DSC_2760.jpg', model)

    if result is not None:
        print(f"Результат предсказания: {result}")
        # Теперь переменная result содержит данные для передачи в другую программу

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
import os


class DefectClassifier:
    def __init__(self, model_path):
        """
        Инициализация классификатора дефектов

        Args:
            model_path (str): путь к файлу модели .h5
        """
        self.model = load_model(model_path)
        self.img_height, self.img_width = self.get_input_shape()

    def get_input_shape(self):
        """Получить размер входного изображения из модели"""
        input_shape = self.model.input_shape
        return input_shape[1], input_shape[2]  # высота, ширина

    def preprocess_image(self, img_path):
        """
        Предобработка изображения для модели

        Args:
            img_path (str): путь к изображению

        Returns:
            numpy array: предобработанное изображение
        """
        # Загрузка изображения
        img = image.load_img(img_path, target_size=(self.img_height, self.img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Нормализация (если модель ожидает значения в диапазоне [0,1])
        img_array = img_array / 255.0

        return img_array

    def predict(self, img_path):
        """
        Предсказание дефекта на изображении

        Args:
            img_path (str): путь к изображению

        Returns:
            tuple: (prediction, confidence, class_name)
        """
        # Предобработка изображения
        processed_img = self.preprocess_image(img_path)

        # Предсказание
        prediction = self.model.predict(processed_img)
        confidence = prediction[0][0]

        # Определение класса
        # Предполагаем, что модель возвращает вероятность класса "defect"
        if confidence > 0.5:
            class_name = "defect"
            confidence = confidence
        else:
            class_name = "not_defect"
            confidence = 1 - confidence

        return class_name, confidence, prediction[0][0]


# Использование
def main():
    # Путь к вашей модели
    model_path = "C:\jdsajhdsa\defect_detection_continued.h5"  # замените на путь к вашей модели

    # Инициализация классификатора
    classifier = DefectClassifier(model_path)

    # Путь к тестовому изображению
    test_image_path = "test_image.jpg"  # замените на путь к вашему изображению

    # Предсказание
    try:
        class_name, confidence, raw_pred = classifier.predict(test_image_path)

        print(f"Результат анализа:")
        print(f"Класс: {class_name}")
        print(f"Уверенность: {confidence:.4f}")
        print(f"Сырое предсказание: {raw_pred:.4f}")

    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")


if __name__ == "__main__":
    main()
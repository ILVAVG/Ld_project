# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io


class DefectClassifierApp:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.img_height, self.img_width = self.model.input_shape[1:3]

    def preprocess_image(self, uploaded_file):
        img = Image.open(uploaded_file)
        img = img.resize((self.img_width, self.img_height))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array, img

    def predict(self, uploaded_file):
        processed_img, original_img = self.preprocess_image(uploaded_file)
        prediction = self.model.predict(processed_img)
        confidence = prediction[0][0]

        if confidence > 0.5:
            class_name = "defect"
            final_confidence = confidence
        else:
            class_name = "not_defect"
            final_confidence = 1 - confidence

        return class_name, final_confidence, original_img


# Streamlit приложение
def main():
    st.title("🔍 Классификатор дефектов")
    st.write("Загрузите изображение для анализа на наличие дефектов")

    # Загрузка модели (кешируется)
    @st.cache_resource
    def load_classifier():
        return DefectClassifierApp("defect_detection_continued.h5")

    classifier = load_classifier()

    # Загрузка изображения
    uploaded_file = st.file_uploader(
        "Выберите изображение",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        # Предсказание
        class_name, confidence, img = classifier.predict(uploaded_file)

        # Отображение результатов
        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Загруженное изображение", use_column_width=True)

        with col2:
            st.subheader("Результат анализа:")

            if class_name == "defect":
                st.error(f"🚨 Обнаружен дефект!")
                st.write(f"Вероятность: {confidence:.2%}")
            else:
                st.success(f"✅ Дефектов не обнаружено")
                st.write(f"Вероятность: {confidence:.2%}")

            st.write(f"**Статус:** {class_name}")
            st.write(f"**Уверенность:** {confidence:.4f}")


if __name__ == "__main__":
    main()
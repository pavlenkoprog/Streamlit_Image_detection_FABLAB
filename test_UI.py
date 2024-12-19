import os

import streamlit as st
from PIL import Image
from ultralytics import YOLO

def main():
    st.markdown("<h2 style='text-align: center'>"
                "Детектирование элементов на изображении</h2>", unsafe_allow_html=True)

    model_files = [f for f in os.listdir('Models') if f.endswith('.pt')]
    selected_model = st.selectbox("Выберите модель", model_files)
    st.write(f"Вы выбрали: {selected_model}")

    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image1 = Image.open(uploaded_file)

        file_base_name, file_extension = os.path.splitext(uploaded_file.name)
        new_filename = f"{file_base_name}_result{file_extension}"

        model = YOLO(f"Models/{selected_model}")
        results = model(image1)
        results[0].save(f"IMG/{new_filename}")

        image2 = Image.open(f"IMG/{new_filename}")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h3 style='text-align: center'>"
                        "Исходное изображение</h3>", unsafe_allow_html=True)
            st.image(image1, use_container_width=True)

        with col2:
            st.markdown("<h3 style='text-align: center'>"
                        "Результат детекции</h3>", unsafe_allow_html=True)
            st.image(image2, use_container_width=True)

if __name__ == "__main__":
    main()
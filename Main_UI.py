import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from Pages import info_blocks

def detection_block(selected_model, class_mapping):
    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
        file_base_name, file_extension = os.path.splitext(uploaded_file.name)
        new_filename = f"{file_base_name}_result{file_extension}"

        model = YOLO(selected_model)

        results = model(uploaded_image)
        classes_names = results[0].names
        print(classes_names)
        for result in results:
            for cls_id, custom_label in class_mapping.items():
                if cls_id in result.names:  # check if the class id is in the results
                    result.names[cls_id] = custom_label  # replace the class name with the custom label

        result.save(f"IMG/{new_filename}")
        result_image = Image.open(f"IMG/{new_filename}")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h3 style='text-align: center'>"
                        "Исходное изображение</h3>", unsafe_allow_html=True)
            st.image(uploaded_image, use_container_width=True)

        with col2:
            st.markdown("<h3 style='text-align: center'>"
                        "Результат детекции</h3>", unsafe_allow_html=True)
            st.image(result_image, use_container_width=True)


def show_preview(original_img_list, result_img_list):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='text-align: center'>"
                    "Исходное изображение</h3>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h3 style='text-align: center'>"
                    "Результат детекции</h3>", unsafe_allow_html=True)

    for original_img, result_img in zip(original_img_list, result_img_list):
        with col1:
            st.image(original_img, use_container_width=True)
        with col2:
            st.image(result_img, use_container_width=True)

def main():

    # Боковая панель
    st.sidebar.header('Выберите модель машинного обучения')
    dictionary = {
        'Общее':
            ['Общая модель'],
        'Медицина':
            ['Распознавание переломов костей на рентгеновских снимках',
            'Распознавание камней в почках на рентгеновских снимках',
            'Распознавание опухолей головного мозга на МРТ',
            'Детекция разных типов клеток крови на микроснимках'],
        'Социальные направления':
            ['Поиск пропавших людей в лесу'],
            # 'Детекция дыма от пожаров'],
        'Агрономия':
            ['Распознавание некоторых растений и их болезней']}
            # 'Распознавание болезней винограда']}


    selected_section = st.sidebar.selectbox("Выберите тематику:", dictionary.keys())
    selected_page = st.sidebar.radio("Модель:", dictionary[selected_section])

    # Центральная панель
    st.markdown("<h2 style='text-align: center'>"f"{selected_page}</h2>", unsafe_allow_html=True)

    if selected_page == 'Общая модель':
        #st.title('Добро пожаловать на страницу Распознавание переломов ')
        selected_model = 'Models/yolo11n.pt'
        class_mapping = {
            0: 'Человек'}

        detection_block(selected_model, class_mapping)

        original_img_list = ["Preview/yolo11/original/1.jpg",
                             "Preview/yolo11/original/2.jpg",
                             "Preview/yolo11/original/3.jpg"]
        result_img_list = ["Preview/yolo11/result/1_result.jpg",
                           "Preview/yolo11/result/2_result.jpg",
                           "Preview/yolo11/result/3_result.jpg"]
        show_preview(original_img_list, result_img_list)
        info_blocks.yolo_info()

    elif selected_page == 'Распознавание переломов костей на рентгеновских снимках':
        #st.title('Добро пожаловать на страницу Распознавание переломов ')
        selected_model = 'Models/bone_fracture_50_ep_best.pt'
        class_mapping = {
            0: 'Смещение',
            1: 'Перелом',
            2: 'Трещина',
            3: 'Сильное смещение'}

        detection_block(selected_model, class_mapping)

        original_img_list = ["Preview/BoneFracture/original/1.jpg",
                             "Preview/BoneFracture/original/2.jpg",
                             "Preview/BoneFracture/original/3.jpg"]
        result_img_list = ["Preview/BoneFracture/result/1_result.jpg",
                           "Preview/BoneFracture/result/2_result.jpg",
                           "Preview/BoneFracture/result/3_result.jpg"]
        show_preview(original_img_list, result_img_list)
        info_blocks.bone_fracture_info()

    elif selected_page == 'Распознавание камней в почках на рентгеновских снимках':
        selected_model = 'Models/kidney_stone_100ep_best.pt'
        class_mapping = {
            0: 'Камень в почке'}
        detection_block(selected_model, class_mapping)

        original_img_list = ["Preview/KidneyStone/original/1.jpg",
                             "Preview/KidneyStone/original/2.jpg",
                             "Preview/KidneyStone/original/3.jpg"]
        result_img_list = ["Preview/KidneyStone/result/1_result.jpg",
                           "Preview/KidneyStone/result/2_result.jpg",
                           "Preview/KidneyStone/result/3_result.jpg"]
        show_preview(original_img_list, result_img_list)
        info_blocks.kidney_stone_info()


    elif selected_page == 'Распознавание опухолей головного мозга на МРТ':
        selected_model = 'Models/brain_tumor_100ep_best.pt'
        class_mapping = {
            0: 'Глиома',
            1: 'Менингиома',
            2: 'Опухоль гипофиза'}
        detection_block(selected_model, class_mapping)

        original_img_list = ["Preview/BrainTumor/original/1.jpg",
                             "Preview/BrainTumor/original/2.jpg",
                             "Preview/BrainTumor/original/3.jpg"]
        result_img_list = ["Preview/BrainTumor/result/1_result.jpg",
                           "Preview/BrainTumor/result/2_result.jpg",
                           "Preview/BrainTumor/result/3_result.jpg"]
        show_preview(original_img_list, result_img_list)
        info_blocks.brain_tumor_detection_info()


    elif selected_page == 'Детекция разных типов клеток крови на микроснимках':
        selected_model = 'Models/blood_cel_100ep_best_v2.pt'
        class_mapping = {
            0: 'Эритроцит',1: 'Лейкоцит',2: 'Тромбоцит'}
        detection_block(selected_model, class_mapping)

        original_img_list = ["Preview/BloodCels/original/1.jpg",
                             "Preview/BloodCels/original/2.jpg",
                             "Preview/BloodCels/original/3.jpg"]
        result_img_list = ["Preview/BloodCels/result/1_result.jpg",
                           "Preview/BloodCels/result/2_result.jpg",
                           "Preview/BloodCels/result/3_result.jpg"]
        show_preview(original_img_list, result_img_list)
        info_blocks.blood_cell_recognition_info()


    elif selected_page == 'Поиск пропавших людей в лесу':
        selected_model = 'Models/aero_vision_100ep_best.pt'
        class_mapping = {
            0: 'Человек'}
        detection_block(selected_model, class_mapping)

        original_img_list = ["Preview/AeroVision/original/1.JPG",
                             "Preview/AeroVision/original/2.JPG",
                             "Preview/AeroVision/original/3.JPG"]
        result_img_list = ["Preview/AeroVision/result/1_result.JPG",
                           "Preview/AeroVision/result/2_result.JPG",
                           "Preview/AeroVision/result/3_result.JPG"]
        show_preview(original_img_list, result_img_list)
        info_blocks.search_victims_info()


    elif selected_page == 'Детекция дыма от пожаров':
        selected_model = 'Models/aero_vision_100ep_best.pt'
        class_mapping = {
            0: 'Человек'}
        detection_block(selected_model, class_mapping)

        original_img_list = ["Preview/AeroVision/original/1.JPG",
                             "Preview/AeroVision/original/2.JPG",
                             "Preview/AeroVision/original/3.JPG"]
        result_img_list = ["Preview/AeroVision/result/1_result.JPG",
                           "Preview/AeroVision/result/2_result.JPG",
                           "Preview/AeroVision/result/3_result.JPG"]
        show_preview(original_img_list, result_img_list)
        info_blocks.forest_fire_detection_info()


    elif selected_page == 'Распознавание некоторых растений и их болезней':
        selected_model = 'Models/PlantLeaf_100_ep_best.pt'
        class_mapping = {
            0: 'Черный горошек здоровый', 1: 'Черный горошек пятнистость листьев',
            2: 'Фасоль долихос пятнистость листьев', 3: 'Фасоль долихос здоровый',
            4: 'Арахис здоровый', 5: 'Арахис пятнистость листьев',
            6: 'Просо здоровый', 7: 'Просо ржавчинный грибок',
            8: 'Томат грибок Alternaria solani', 9: 'Томат здоровый'}
        detection_block(selected_model, class_mapping)

        original_img_list = ["Preview/PlantLeaf/original/1.jpg",
                             "Preview/PlantLeaf/original/2.jpg",
                             "Preview/PlantLeaf/original/3.jpg"]
        result_img_list = ["Preview/PlantLeaf/result/1_result.jpg",
                           "Preview/PlantLeaf/result/2_result.jpg",
                           "Preview/PlantLeaf/result/3_result.jpg"]
        show_preview(original_img_list, result_img_list)
        info_blocks.plant_disease_info()


    elif selected_page == 'Распознавание болезней винограда':
        selected_model = 'Models/aero_vision_100ep_best.pt'
        class_mapping = {
            0: 'Человек'}
        detection_block(selected_model, class_mapping)

        original_img_list = ["Preview/AeroVision/original/1.JPG",
                             "Preview/AeroVision/original/2.JPG",
                             "Preview/AeroVision/original/3.JPG"]
        result_img_list = ["Preview/AeroVision/result/1_result.JPG",
                           "Preview/AeroVision/result/2_result.JPG",
                           "Preview/AeroVision/result/3_result.JPG"]
        show_preview(original_img_list, result_img_list)
        info_blocks.grape_disease_detection_info()

    else:
        selected_model = 'Models/bone_fracture_50_ep_best.pt'
        class_mapping = {
            0: 'Смещение',
            1: 'Перелом',
            2: 'Трещина',
            3: 'Сильное смещение'}

        detection_block(selected_model, class_mapping)

        original_img_list = ["Preview/BoneFracture/original/1.jpg",
                             "Preview/BoneFracture/original/2.jpg",
                             "Preview/BoneFracture/original/3.jpg"]
        result_img_list = ["Preview/BoneFracture/result/1_result.jpg",
                           "Preview/BoneFracture/result/2_result.jpg",
                           "Preview/BoneFracture/result/3_result.jpg"]
        show_preview(original_img_list, result_img_list)
        info_blocks.bone_fracture_info()

if __name__ == "__main__":
    main()
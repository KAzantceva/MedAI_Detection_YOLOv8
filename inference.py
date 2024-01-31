import os
from PIL import Image
import Augmentor

def create_annotations(input_dir, output_dir):
    # Создаем объект Augmentor для директории с изображениями
    pipeline = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)

    # Добавляем операции аугментации
    pipeline.rotate(probability=1.0, max_left_rotation=5, max_right_rotation=5)
    pipeline.flip_left_right(probability=0.5)
    pipeline.zoom_random(probability=0.5, percentage_area=0.8)

    # Запускаем процесс аугментации
    pipeline.sample(1000)  # Увеличьте количество создаваемых аугментированных изображений

def create_annotations_txt(input_dir, output_dir):
    # Перебираем поддиректории (классы)
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)

        if os.path.isdir(class_dir):
            # Создаем подкаталог для каждого класса в директории аннотаций
            class_output_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)

            # Перебираем изображения в каждой поддиректории (классе)
            for image_name in os.listdir(class_dir):
                if image_name.endswith(".jpg"):
                    image_path = os.path.join(class_dir, image_name)

                    # Получаем размер изображения
                    with Image.open(image_path) as img:
                        width, height = img.size

                    # Создаем файл аннотации в формате YOLOv8
                    with open(os.path.join(class_output_dir, image_name.replace(".jpg", ".txt")), "w") as annotation_file:
                        # Записываем пример аннотации (здесь 0 - индекс класса)
                        annotation_file.write(f"0 {width * 0.5 / width} {height * 0.5 / height} {width / width} {height / height}\n")

if __name__ == "__main__":
    # Директория с тренировочными изображениями и аугментированными данными
    train_input_directory = "Alzheimer_s Dataset/train"
    train_output_directory = os.path.join(train_input_directory, "annotations")

    # Директория с тестовыми изображениями и аннотациями
    test_input_directory = "Alzheimer_s Dataset/test"
    test_output_directory = os.path.join(test_input_directory, "annotations")

    # Создаем аннотации для тренировочных изображений
    create_annotations(train_input_directory, train_output_directory)

    # Создаем файлы аннотаций .txt на основе созданных изображений для тренировочной выборки
    create_annotations_txt(train_input_directory, train_output_directory)

    # Создаем аннотации для тестовых изображений
    create_annotations(test_input_directory, test_output_directory)

    # Создаем файлы аннотаций .txt на основе созданных изображений для тестовой выборки
    create_annotations_txt(test_input_directory, test_output_directory)

    # Выводим информацию для конфигурационного файла YOLOv8
    print(f"train: ../{train_input_directory}")
    print(f"val: ../{test_input_directory}")
    print("nc: 4  # количество классов (MildDemented, ModerateDemented, NonDemented, VeryMildDemented)")
    print("names: ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']")
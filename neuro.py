import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter


IMG_SIZE = (854, 480)
BATCH_SIZE = 32
EPOCHS = 20

FEATURE_COLUMNS = ["angle", "size", "style", "pressure", "space", "evenness", "symetric", "unic", "stable", "direction", "readable"]

ENCODE_COLUMNS = ["angle", "style", "pressure", "evenness", "symetric", "unic", "stable", "direction", "readable"]



def handle_nan(features):
    for key, value in features.items():
        if np.isnan(value):
            features[key] = np.nanmean(list(features.values()))  # или можно заменить на медиану
    return features

def extract_handwriting_features(image):
    features = []

    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    binary = binary.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Угол наклона
    angle = calculate_tilt_angle(contours)
    features.append(angle)

    # Размер
    size = calculate_size(contours)
    features.append(size)

    # Стиль - наклонность букв
    style = calculate_style(contours)
    features.append(style)

    # Давление - толщина линий
    pressure = calculate_pressure(binary)
    features.append(pressure)

    # Пространство между словами
    space_between_words = calculate_space_between_words(contours)
    features.append(space_between_words)

    # Ровность - стандартное отклонение координат y контуров
    smoothness = calculate_smoothness(contours)
    features.append(smoothness)

    # Симметрия - сравнение левой и правой частей контура
    symmetry = calculate_symmetry(contours)
    features.append(symmetry)

    # Уникальность - количество уникальных форм
    uniqueness = calculate_uniqueness(contours)
    features.append(uniqueness)

    # Стабильность - стандартное отклонение толщины линий
    stability = calculate_stability(binary)
    features.append(stability)

    # Направление - направление письма
    direction = calculate_direction(contours)
    features.append(direction)

    # Читаемость - по количеству разрывов в контурах
    readability = calculate_readability(contours)
    features.append(readability)

    print(features)

    return np.array(features)








# def extract_handwriting_features(image):
#     features = []

#     # Преобразование в бинарное изображение
#     _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
#     binary = binary.astype(np.uint8)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Вычисление наклона букв
#     moments = cv2.moments(binary)
#     if moments["mu20"] + moments["mu02"] > 0:
#         angle = 0.5 * np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])
#         angle = np.degrees(angle)
#         if angle > 10:
#             slope = "Правый"
#         elif angle < -10:
#             slope = "Левый"
#         else:
#             slope = "Вертикальный"
#     else:
#         slope = "Вертикальный"

#     features.append(slope)

#     # Вычисление расстояния между словами
#     hor_proj = np.sum(binary, axis=0)
#     word_spaces = np.diff(np.where(hor_proj == 0)[0])
#     avg_word_space = np.mean(word_spaces) if len(word_spaces) > 0 else 0
#     features.append(avg_word_space)

#     # Оценка ровности строк
#     ver_proj = np.sum(binary, axis=1)
#     nonzero_rows = np.where(ver_proj > 0)[0]
#     row_variability = np.std(np.diff(nonzero_rows))
#     straightness = "Ровные" if row_variability < 5 else "Неровные"
#     features.append(straightness)

#     # Симметрия элементов букв
#     sym_diff = np.sum(np.abs(binary[:, :binary.shape[1]//2] - np.fliplr(binary[:, binary.shape[1]//2:])))
#     symmetry = "Высокая" if sym_diff < 5000 else "Низкая"
#     features.append(symmetry)

#     # Количество исправлений (по числу пересечений контуров)
#     corrections = len(contours)
#     features.append(corrections)

#     # Определение стиля написания
#     avg_contour_len = np.mean([cv2.arcLength(cnt, closed=False) for cnt in contours]) if contours else 0
#     if avg_contour_len > 100:
#         writing_style = "Курсивный"
#     elif avg_contour_len > 50:
#         writing_style = "Полукурсивный"
#     else:
#         writing_style = "Печатный"
#     features.append(writing_style)

#     # Оценка стабильности почерка (дрожание линий)
#     line_variability = np.mean([cv2.arcLength(cnt, closed=False) / len(cnt) for cnt in contours if len(cnt) > 0]) if contours else 0
#     if line_variability < 2:
#         stability = "Высокая"
#     elif line_variability < 4:
#         stability = "Средняя"
#     else:
#         stability = "Низкая"
#     features.append(stability)

#     # Определение использования прописных букв
#     upper_case_ratio = sum(1 for cnt in contours if cv2.boundingRect(cnt)[3] > binary.shape[0] * 0.6) / len(contours) if contours else 0
#     if upper_case_ratio > 0.5:
#         upper_case_usage = "Часто"
#     elif upper_case_ratio > 0.2:
#         upper_case_usage = "Иногда"
#     else:
#         upper_case_usage = "Редко"
#     features.append(upper_case_usage)

#     # Определение пропусков в словах
#     has_gaps = "Есть" if avg_word_space > 20 else "Нет"
#     features.append(has_gaps)

#     # Определение направления штрихов
#     sobel_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=5)
#     sobel_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=5)
#     stroke_direction = "Однородное" if np.std(sobel_x) < 20 and np.std(sobel_y) < 20 else "Смешанное"
#     features.append(stroke_direction)

#     # Оценка общей читаемости
#     black_pixel_ratio = np.sum(binary == 255) / binary.size
#     if black_pixel_ratio > 0.6:
#         readability = "Высокая"
#     elif black_pixel_ratio > 0.3:
#         readability = "Средняя"
#     else:
#         readability = "Низкая"
#     features.append(readability)

#     # return features

#     # return features
#     return np.array(features)


# def extract_handwriting_features(image):
#     features = {}

#     # Преобразование в бинарное изображение
#     _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
#     binary = binary.astype(np.uint8)

#     # Поиск контуров на изображении
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Признак: Наклон букв
#     def calculate_slope(contour):
#         # Определим наклон букв на основе ориентации внешнего контура
#         rect = cv2.minAreaRect(contour)
#         angle = rect[2]
#         if -10 <= angle <= 10:
#             return 'Вертикальный'
#         elif 10 < angle < 45:
#             return 'Правый'
#         elif -45 < angle < -10:
#             return 'Левый'
#         else:
#             return 'Вертикальный'

#     slopes = [calculate_slope(c) for c in contours]
#     features['Наклон букв'] = max(set(slopes), key=slopes.count)  # Определяем наиболее частый наклон

#     # Признак: Стиль написания (печатный, курсивный и т.д.)
#     # Для примера, просто определим стиль по углу наклона
#     def determine_style():
#         if features['Наклон букв'] in ['Правый', 'Левый']:
#             return 'Курсивный'
#         else:
#             return 'Печатный'

#     features['Стиль написания'] = determine_style()

#     # Признак: Пространство между словами
#     def calculate_word_spacing(contours):
#         distances = []
#         for i in range(len(contours) - 1):
#             (x1, _, _, _) = cv2.boundingRect(contours[i])
#             (x2, _, _, _) = cv2.boundingRect(contours[i + 1])
#             distances.append(x2 - (x1 + contours[i][2][0]))
#         return np.mean(distances)  # Возвращаем среднее расстояние между словами

#     word_spacing = calculate_word_spacing(contours)
#     features['Пространство между словами'] = word_spacing  # Преобразовать в мм по разрешению изображения

#     # Признак: Ровность строк
#     def check_line_smoothness():
#         heights = [cv2.boundingRect(c)[3] for c in contours]
#         return 'Ровные' if max(heights) - min(heights) < 10 else 'Неровные'

#     features['Ровность строк'] = check_line_smoothness()

#     # Признак: Симметрия элементов букв
#     def check_symmetry():
#         symmetries = ['Высокая' if len(c) < 50 else 'Низкая' for c in contours]
#         return max(set(symmetries), key=symmetries.count)

#     features['Симметрия элементов букв'] = check_symmetry()

#     # Признак: Стабильность почерка
#     features['Стабильность почерка'] = 'Средняя'  # Это пример, стабилизацию можно настраивать по дополнительным признакам.

#     # Признак: Использование прописных букв
#     def check_capital_usage():
#         return 'Часто' if len([c for c in contours if cv2.boundingRect(c)[3] > 50]) > len(contours) / 2 else 'Редко'

#     features['Использование прописных букв'] = check_capital_usage()

#     # Признак: Наличие пропусков в словах
#     def check_spacing_gaps():
#         spacing = calculate_word_spacing(contours)
#         return 'Есть' if spacing > 30 else 'Нет'

#     features['Наличие пропусков в словах'] = check_spacing_gaps()

#     # Признак: Направление штрихов
#     def check_stroke_direction():
#         directions = ['Однородное' if len(c) < 40 else 'Смешанное' for c in contours]
#         return max(set(directions), key=directions.count)

#     features['Направление штрихов'] = check_stroke_direction()

#     # Признак: Количество исправлений
#     def count_corrections():
#         return np.random.randint(0, 5)  # Для примера возвращаем случайное число

#     features['Количество исправлений'] = count_corrections()

#     # Признак: Общая читаемость
#     features['Общая читаемость'] = 'Средняя'  # Это пример, можно настраивать по признакам контуров






#     categorical_features = {
#         'Наклон букв': {'Вертикальный': 0, 'Правый': 1, 'Левый': 2},
#         'Стиль написания': {'Печатный': 0, 'Курсивный': 1},
#         'Ровность строк': {'Ровные': 0, 'Неровные': 1},
#         'Симметрия элементов букв': {'Высокая': 0, 'Низкая': 1},
#         'Стабильность почерка': {'Средняя': 0},
#         'Использование прописных букв': {'Часто': 1, 'Редко': 0},
#         'Наличие пропусков в словах': {'Нет': 0, 'Есть': 1},
#         'Направление штрихов': {'Однородное': 0, 'Смешанное': 1},
#         'Общая читаемость': {'Средняя': 0}
#     }

#     for feature, mapping in categorical_features.items():
#         features[feature] = mapping.get(features[feature], -1)

#     # Пространство между словами нормализуем (например, делим на 100)
#     features['Пространство между словами'] /= 100.0

#     # Количество исправлений тоже нормализуем
#     features['Количество исправлений'] /= 5.0


#     features = handle_nan(features)

#     return np.array(list(features.values()))



#     # return features
#     # return np.array(features)


def calculate_tilt_angle(contours):
    # # вычисление угла наклона с помощью метода наименьших квадратов
    # angles = []
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     # if w > 10 and h > 10:  # слишком маленькие контуры
    #     vx, vy, x0, y0 = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    #     print("vx vy ", vx, vy)
    #     angle = np.arctan2(vy, vx) * 180 / np.pi
    #     print("angle ", angle)
    #     angles.append(angle)
    # return np.mean(angles) if angles else 0

    # Предполагаем, что contours - это список контуров, найденных с помощью cv2.findContours
    # Для простоты берем первый контур
    contour = contours[0]

    # Преобразуем контур в массив точек
    points = contour.reshape(-1, 2)

    # Разделяем координаты x и y
    x = points[:, 0]
    y = points[:, 1]

    # Вычисляем коэффициенты прямой линии с помощью метода наименьших квадратов
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    # Вычисляем угол наклона в радианах
    angle_radians = np.arctan(m)

    # Преобразуем угол в градусы
    angle_degrees = np.degrees(angle_radians)
    print("erfvevf ", angle_degrees)

    return angle_degrees

def calculate_size(contours):
    # вычисление среднего размера контуров
    sizes = [cv2.contourArea(contour) for contour in contours]
    return np.mean(sizes) if sizes else 0

def calculate_style(contours):
    angles = []
    for contour in contours:
        # слишком маленькие контуры
        if len(contour) > 5:  # Минимум 5 точек для аппроксимации
            ellipse = cv2.fitEllipse(contour)
            angle = ellipse[2]  # Угол наклона эллипса
            angles.append(angle)

    if not angles:
        return 0

    return np.mean(angles)

def calculate_pressure(binary_image):
    # Инвертируем бинарное изображение - фон 0, линии 1
    binary_inv = cv2.bitwise_not(binary_image)

    dist_transform = cv2.distanceTransform(binary_inv, cv2.DIST_L2, 5)
    dist_transform_normalized = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    mean_thickness = np.mean(dist_transform[binary_inv == 255])
    return mean_thickness

def calculate_space_between_words(contours):
    if not contours:
        return 0

    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    bounding_boxes.sort(key=lambda x: x[0])

    words = []
    current_word = [bounding_boxes[0]]

    for box in bounding_boxes[1:]:
        last_box = current_word[-1]
        # Если расстояние между текущим и предыдущим bounding box'ом меньше порога, считаем их одним словом
        if box[0] - (last_box[0] + last_box[2]) < 20:  # Порог можно настроить
            current_word.append(box)
        else:
            words.append(current_word)
            current_word = [box]
    words.append(current_word)

    # bounding box'ы для каждого слова
    word_boxes = []
    for word in words:
        x_coords = [box[0] for box in word]
        y_coords = [box[1] for box in word]
        widths = [box[2] for box in word]
        heights = [box[3] for box in word]

        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords[i] + widths[i] for i in range(len(word)))
        y_max = max(y_coords[i] + heights[i] for i in range(len(word)))

        word_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    # расстояния между словами
    distances = []
    for i in range(1, len(word_boxes)):
        prev_box = word_boxes[i - 1]
        curr_box = word_boxes[i]
        distance = curr_box[0] - (prev_box[0] + prev_box[2])
        distances.append(distance)

    if not distances:
        return 0

    return np.mean(distances)

def calculate_smoothness(contours):
    # вычисление ровности - стандартное отклонение координат y контуров
    y_coords = [point[0][1] for contour in contours for point in contour]
    return np.std(y_coords) if y_coords else 0

def calculate_symmetry(contours):
    # вычисление симметрии - сравнение левой и правой частей контура

    if len(contours) == 0:
        return 0

    symmetry_scores = []

    for cnt in contours:
        # Разделение контура на левую и правую части
        # центр масс контура
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cX = int(M["m10"] / M["m00"])  # Центр масс по X

        # на две части по X
        left_part = [point for point in cnt if point[0][0] < cX]
        right_part = [point for point in cnt if point[0][0] > cX]
        # сходство между левой и правой частями
        if len(left_part) == 0 or len(right_part) == 0:
            symmetry_scores.append(0)
        else:
            # cреднее расстояние между соответствующими точками
            left_points = np.array([point[0] for point in left_part])
            right_points = np.array([point[0] for point in right_part])

            # Выравниваем правую часть по центру
            right_points[:, 0] = right_points[:, 0] - 2 * (right_points[:, 0] - cX)

            distances = np.linalg.norm(left_points - right_points, axis=1)
            symmetry_score = np.mean(distances)
            symmetry_scores.append(symmetry_score)

    return np.mean(symmetry_scores) if symmetry_scores else 0

def calculate_uniqueness(contours, threshold=0.1):
    if not contours:
        return 0

    # Вычисляем Hu-моменты для каждого контура
    hu_moments_list = []
    for contour in contours:
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments)
        # Нормализуем Hu-моменты для сравнения
        hu_moments = np.log(np.abs(hu_moments))  # Логарифмирование для уменьшения диапазона значений
        hu_moments_list.append(hu_moments.flatten())

    # Сравниваем Hu-моменты и группируем контуры по уникальным формам
    unique_forms = []
    for i, hu_moments in enumerate(hu_moments_list):
        is_unique = True
        for form in unique_forms:
            # Вычисляем евклидово расстояние между Hu-моментами
            distance = np.linalg.norm(hu_moments - form["hu_moments"])
            if distance < threshold:  # Если расстояние меньше порога, формы считаются одинаковыми
                is_unique = False
                break
        if is_unique:
            unique_forms.append({"hu_moments": hu_moments, "contour": contours[i]})

    # Возвращаем количество уникальных форм
    return len(unique_forms)

def calculate_stability(binary_image):
    # Пример: вычисление стабильности (стандартное отклонение толщины линий)
    # Здесь можно использовать анализ толщины линий в бинарном изображении
    return np.std(binary_image) / 255

def calculate_direction(contours):

    if not contours:
        return 5

    angles = []
    for contour in contours:
        # Игнорируем слишком маленькие контуры
        if len(contour) > 5:  # Минимум 5 точек для аппроксимации
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = np.arctan2(vy, vx) * 180 / np.pi  # Угол в градусах
            angles.append(angle)

    if not angles:
        return 5

    # Определяем преобладающее направление
    mean_angle = np.mean(angles)
    if -45 <= mean_angle <= 45:
        return -1  # Слева
    elif 135 <= mean_angle <= 180 or -180 <= mean_angle <= -135:
        return 1  # Справа
    else:
        return 0

def calculate_readability(contours):
    if len(contours) == 0:
        return 0  # Нет контуров, читаемость равна 0

    break_count = 0

    for cnt in contours:
        # Преобразуем контур в бинарное изображение
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, 1)

        # Применяем морфологическую операцию для обнаружения разрывов
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)  # Расширяем контуры

        # Вычисляем разницу между исходным и расширенным изображением
        diff = cv2.absdiff(mask, dilated_mask)

        # Если разница не равна нулю, значит, есть разрыв
        break_count += np.sum(diff == 255)

    return break_count

def load_images_and_features(folder, labels_file):
    df = pd.read_csv(labels_file, sep=";")
    images, features, labels = [], [], []

    label_encoder = LabelEncoder()
    df["class"] = label_encoder.fit_transform(df["class"])


    for index, row in df.iterrows():
        img_path = os.path.join(folder, row['filename'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        images.append(img)
        features.append(extract_handwriting_features(img))
        labels.append(row['class'])

        print(images)
        print(features)
        print(labels)


    np.save("class_labels.npy", label_encoder.classes_)

    return (np.array(images).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1),
            np.array(features, dtype=np.float32),
            np.array(labels, dtype=np.int32))




def main():

    # Загрузка данных
    X_images, X_features, y = load_images_and_features("dataset/images", "dataset/labels.csv")
    X_img_train, X_img_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(X_images, X_features, y, test_size=0.2, random_state=42)

    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
    datagen.fit(X_img_train)

    img_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
    cnn = Conv2D(32, (3,3), activation='relu')(img_input)
    cnn = MaxPooling2D((2,2))(cnn)
    cnn = Conv2D(64, (3,3), activation='relu')(cnn)
    cnn = MaxPooling2D((2,2))(cnn)
    cnn = Flatten()(cnn)

    feat_input = Input(shape=(11,))  # 11 признакво
    feat_dense = Dense(16, activation='relu')(feat_input)

    merged = concatenate([cnn, feat_dense])
    output = Dense(6, activation='softmax')(merged)  # 6 классов

    model = Model(inputs=[img_input, feat_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    model.fit([X_img_train, X_feat_train], y_train, epochs=EPOCHS, validation_data=([X_img_test, X_feat_test], y_test))


    model.save("handwriting_model.h5")



# def predict_handwriting(image_path, model, n_attempts=5):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, IMG_SIZE)
#     img = img / 255.0

#     features = extract_handwriting_features(img)
#     print(features)

#     img = img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
#     features = features.reshape(1, -1)

#     # prediction = model.predict([img, features])
#     # print(prediction)
#     # return np.argmax(prediction)

#     predictions = []
#     for _ in range(n_attempts):
#         prediction = model.predict([img, features])
#         predictions.append(np.argmax(prediction))  # Запоминаем индекс с максимальной вероятностью
#     most_common = Counter(predictions).most_common(1)[0][0]
#     print(f"Предсказания: {predictions}")
#     print(f"Итоговый класс: {most_common}")
#     return most_common


# # Пример предсказания
# loaded_model = tf.keras.models.load_model("handwriting_model.h5")
# print(predict_handwriting("test/academ.png", loaded_model))
# # 0 Academic
# # 1 Caligraphic
# # 2 Expression
# # 3 Haus
# # 4 Individual
# # 5 Machinelike

# main()
from neuro import *
from tensorflow.keras.preprocessing import image


IMG_SIZE = (854, 480)
BATCH_SIZE = 32
EPOCHS = 20

def predict_handwriting(image_path, model, n_attempts=5):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0

    features = extract_handwriting_features(img)
    print("Извлечённые признаки:", features)

    img = img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    features = features.reshape(1, -1)

    predictions = []
    for _ in range(n_attempts):
        prediction = model.predict([img, features])
        predictions.append(np.argmax(prediction))


    most_common = Counter(predictions).most_common(1)[0][0]
    print(f"Предсказания: {predictions}")
    print(f"Итоговый класс: {most_common}")
    return most_common






def add_to_model(src, n_attempts=5):
    print("add_to_model")
    tf.config.run_functions_eagerly(True)

    loaded_model = tf.keras.models.load_model("handwriting_model.h5")
    loaded_model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0

    features = extract_handwriting_features(img)
    features = np.nan_to_num(features, nan=0.0)
    print("Извлечённые признаки:", features)

    img = img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    features = features.reshape(1, -1)


    predictions = []
    for _ in range(n_attempts):
        prediction = loaded_model.predict([img, features])
        predictions.append(np.argmax(prediction))

    most_common = Counter(predictions).most_common(1)[0][0]
    print(f"Предсказания: {predictions}")
    print(f"Итоговый класс: {most_common}")


    new_data = [img, features]
    new_labels = np.array([most_common])



    print("new_data shape:", [x.shape for x in new_data])
    print("new_labels shape:", new_labels.shape)




    loaded_model.fit(new_data, new_labels, epochs=10, batch_size=32)
    loaded_model.save("updated_handwriting_model.h5")

    return "Готово"









def Main_Predict(src):

    # Пример предсказания
    loaded_model = tf.keras.models.load_model("handwriting_model.h5")
    return predict_handwriting(src, loaded_model)
    # print(predict_handwriting("test/haos.png", loaded_model))
    # 0 Academic
    # 1 Caligraphic
    # 2 Expression
    # 3 Haus
    # 4 Individual
    # 5 Machinelike

# if __name__ == "__main__":
#     add_to_model("./test/haos.png")
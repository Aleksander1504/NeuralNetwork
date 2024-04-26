from PIL import Image, ImageDraw, ImageFont
import numpy as np

#размер изображения
image_size = (28, 28)

#создание набора данных изображений
symbols = ['A', 'B', 'C', 'D']
font_paths = ['font1.ttf', 'font2.ttf', 'font3.ttf', 'font4.ttf']  #пути к разным шрифтам

#создание обучающего набора данных
train_data = []
for symbol_index, symbol in enumerate(symbols):
    for font_index, font_path in enumerate(font_paths):
        image = Image.new("L", image_size, color=255)  #создание изображения
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, 15)
        draw.text((6, 6), symbol, font=font, fill=0)  #рисование символа на изображении
        train_data.append((np.array(image), symbol_index))  #добавление меток классов

#создание тестового набора данных
test_data = []
for symbol_index, symbol in enumerate(symbols):
    for font_index, font_path in enumerate(font_paths):
        image = Image.new("L", image_size, color=255)  #создание изображения
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, 15)
        draw.text((6, 6), symbol, font=font, fill=0)  #дисование символа на изображении
        test_data.append((np.array(image), symbol_index))  #добавление меток классов

#сохранение тестовых изображений и их визуализация
for i, (test_image_array, symbol_index) in enumerate(test_data):
    test_image = Image.fromarray(test_image_array)  #преобразование массива numpy в объект Image
    test_image_path = f"test_image_{symbols[symbol_index]}.png"  #имя файла будет соответствовать символу
    test_image.save(test_image_path)

    #визуализация тестового изображения
    test_image.show()

#вывод информации о количестве образов в наборах данных
print("Количество обучающих образов:", len(train_data))
print("Количество тестовых образов:", len(test_data))
print("Количество входных сигналов =", image_size[0] * image_size[1])  #размерность изображения
print("Количество выходных сигналов =", len(symbols))  #количество классов

class NeuralNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.1, stop_criteria=0.01, weight_range=(-0.5, 0.5)):
        self.weights = np.random.uniform(*weight_range, size=(input_size, output_size))  #инициализация весов с заданным диапазоном
        self.learning_rate = learning_rate
        self.stop_criteria = stop_criteria

    def activation(self, x):
        return np.where(x >= 0, 1, 0)  #пороговая функция активации

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        return self.activation(weighted_sum)

    def train(self, training_data, epochs):
        for epoch in range(epochs):
            np.random.shuffle(training_data)  #перемешиваем данные на каждой эпохе
            total_error = 0
            for image, label in training_data:
                image_flat = image.flatten()  #преобразуем изображение в одномерный массив
                prediction = self.predict(image_flat)
                error = label - prediction
                self.weights += self.learning_rate * error * np.reshape(image_flat, (len(image_flat), 1)) #Дельта-правило (или правило Видроу-Хоффа) обновляет веса нейрона в соответствии с ошибкой, которая является разностью между ожидаемым выходом и фактическим выходом.
                total_error += np.abs(error)
            print(f"Эпоха {epoch + 1}, средняя ошибка: {total_error / len(training_data)}")

#создание нейрона
input_size = image_size[0] * image_size[1]  #размер входного вектора (размер изображения)
output_size = 4  #количество выходных сигналов (4 символа)
learning_rate = 0.1
stop_criteria = 0.01
weight_range = (-0.5, 0.5)

neuron = NeuralNetwork(input_size, output_size, learning_rate, stop_criteria, weight_range)

#объединение обучающих данных и меток классов
training_data = [(data.flatten(), label) for data, label in train_data]  #преобразуем изображения в одномерные массивы и добавляем метки классов

#обучение нейрона
epochs = 100
neuron.train(training_data, epochs)

#тестирование на тестовых изображениях
correct_predictions = 0
total_predictions = len(test_data)
for test_image_array, symbol_index in test_data:
    prediction = neuron.predict(test_image_array.flatten())  #предсказание на одном тестовом изображении
    print(f"Предсказанный символ для изображения {symbols[symbol_index]} с шрифтом {font_index}:", prediction)
    if np.argmax(prediction) == symbol_index:  #сравниваем индекс максимального элемента с меткой символа
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
print("Точность модели:", accuracy)
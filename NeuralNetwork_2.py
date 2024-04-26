from PIL import Image, ImageDraw, ImageFont
import numpy as np

#размер изображения
image_size = (28, 28)

class NeuralNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size, output_size)  #инициализация весов рандомными значениями
        self.learning_rate = learning_rate

    def activation(self, x):
        return np.where(x >= 0, 1, 0)  #пороговая функция активации

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        return self.activation(weighted_sum)

    def train(self, training_data, epochs):
        for epoch in range(epochs):
            total_error = 0
            for image, label in training_data:
                image_flat = image.flatten()  #преобразуем изображение в одномерный массив
                prediction = self.predict(image_flat)
                error = label - prediction
                gradient = -error * np.reshape(image_flat, (len(image_flat), 1))  #градиент функции потерь
                self.weights -= self.learning_rate * gradient  #обновление весов в направлении антиградиента
                total_error += np.abs(error)
            print(f"Эпоха {epoch + 1}, средняя ошибка: {total_error / len(training_data)}")

#создание обучающего и тестового набора данных
symbols = ['A', 'B', 'C', 'D']
font_paths = ['font1.ttf', 'font2.ttf', 'font3.ttf', 'font4.ttf']  #пути к разным шрифтам

train_data = []
for symbol_index, symbol in enumerate(symbols):
    for font_index, font_path in enumerate(font_paths):
        image = Image.new("L", image_size, color=255)  #создание изображения
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, 15)
        draw.text((6, 6), symbol, font=font, fill=0)  #рисование символа на изображении
        train_data.append((np.array(image), symbol_index))

test_data = []
for symbol_index, symbol in enumerate(symbols):
    for font_index, font_path in enumerate(font_paths):
        other_font_path = font_paths[(font_index + 1) % len(font_paths)]  #выбираем другой шрифт для теста
        image = Image.new("L", image_size, color=255)  #создание изображения
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(other_font_path, 15)
        draw.text((6, 6), symbol, font=font, fill=0)  #рисование символа на изображении
        test_data.append((np.array(image), symbol_index))

#количество входных и выходных сигналов
input_size = image_size[0] * image_size[1]  #размер входного вектора (размер изображения)
output_size = len(symbols)  #количество выходных сигналов (4 символа)
print("Количество входных сигналов =", input_size)
print("Количество выходных сигналов =", output_size)

#создание нейрона
learning_rate = 0.1
neuron = NeuralNetwork(input_size, output_size, learning_rate)

#объединение обучающих данных и меток классов
training_data = [(data.flatten(), [1 if i == label else 0 for i in range(output_size)]) for data, label in train_data]

#обучение нейрона
epochs = 100
neuron.train(training_data, epochs)

#тестирование на тестовых данных
correct_predictions = 0
for image, label in test_data:
    prediction = neuron.predict(image.flatten())  #предсказание на тестовом изображении
    if np.argmax(prediction) == label:  #сравниваем индекс максимального элемента с меткой класса
        correct_predictions += 1

accuracy = correct_predictions / len(test_data)
print("Точность модели:", accuracy)

#сохранение тестовых изображений и их визуализация
for i, (test_image_array, symbol_index) in enumerate(test_data):
    test_image = Image.fromarray(test_image_array)  #преобразование массива numpy в объект Image
    test_image_path = f"test_image_{symbols[symbol_index]}.png"  #имя файла будет соответствовать символу
    test_image.save(test_image_path)

    #визуализация тестового изображения
    test_image.show()

#тестирование на случайном изображении
random_image = np.random.randint(0, 256, size=image_size)  #генерация случайного изображения
prediction = neuron.predict(random_image.flatten())  #предсказание на случайном изображении
print("Предсказанный класс для случайного изображения:", np.argmax(prediction))
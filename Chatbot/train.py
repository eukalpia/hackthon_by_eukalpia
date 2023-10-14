import numpy as np  # Импорт библиотеки numpy
import random  # Импорт модуля random
import json  # Импорт модуля json
import torch  # Импорт библиотеки torch
import torch.nn as nn  # Импорт модуля nn из библиотеки torch
from torch.utils.data import Dataset, DataLoader  # Импорт классов Dataset и DataLoader из модуля torch.utils.data
from nltk_utils import bag_of_words, tokenize, stem  # Импорт функций bag_of_words, tokenize, stem из модуля nltk_utils
from model import NeuralNet  # Импорт класса NeuralNet из модуля model
with open('intents.json', 'r') as f:  # Открытие файла intents.json в режиме чтения и его присвоение переменной f
    intents = json.load(f)  # Загрузка данных из файла intents.json и их присвоение переменной intents
all_words = []  # Создание пустого списка all_words
tags = []  # Создание пустого списка tags
xy = []  # Создание пустого списка xy
for intent in intents['intents']:  # Итерация по элементам списка intents['intents']
    tag = intent['tag']  # Получение значения ключа 'tag' из текущего элемента и его присвоение переменной tag
    tags.append(tag)  # Добавление значения переменной tag в список tags
    for pattern in intent['patterns']:  # Итерация по элементам списка значений ключа 'patterns' текущего элемента
        w = tokenize(pattern)  # Токенизация текущего элемента и его присвоение переменной w
        all_words.extend(w)  # Добавление элементов переменной w в список all_words
        xy.append((w, tag))  # Добавление кортежа (w, tag) в список xy
ignore_words = ['?', '.', '!']  # Создание списка ignore_words с заданными значениями
all_words = [stem(w) for w in all_words if w not in ignore_words]  # Применение функции stem к каждому элементу списка all_words, если элемент не находится в списке ignore_words
all_words = sorted(set(all_words))  # Преобразование списка all_words в отсортированное множество
tags = sorted(set(tags))  # Преобразование списка tags в отсортированное множество
print(len(xy), "patterns")  # Вывод количества элементов списка xy и строки "patterns"
print(len(tags), "tags:", tags)  # Вывод количества элементов списка tags и строки "tags:", а затем вывод списка tags
print(len(all_words), "Уникальные слова:", all_words)  # Вывод количества элементов списка all_words и строки "Уникальные слова:", а затем вывод списка all_words
X_train = []  # Создание пустого списка X_train
y_train = []  # Создание пустого списка y_train

for (pattern_sentence, tag) in xy:  # Итерация по элементам списка xy
    bag = bag_of_words(pattern_sentence, all_words)  # Вызов функции bag_of_words с аргументами pattern_sentence и all_words и присвоение результата переменной bag
    X_train.append(bag)  # Добавление значения переменной bag в список X_train
    label = tags.index(tag)  # Получение индекса значения переменной tag в списке tags и его присвоение переменной label
    y_train.append(label)  # Добавление значения переменной label в список y_train
X_train = np.array(X_train)  # Преобразование списка X_train в массив numpy
y_train = np.array(y_train)  # Преобразование списка y_train в массив numpy
num_epochs = 1000  # Присвоение переменной num_epochs значения 1000
batch_size = 8  # Присвоение переменной batch_size значения 8
learning_rate = 0.001  # Присвоение переменной learning_rate значения 0.001
input_size = len(X_train[0])  # Присвоение переменной input_size значения длины первого элемента массива X_train
hidden_size = 8  # Присвоение переменной hidden_size значения 8
output_size = len(tags)  # Присвоение переменной output_size значения длины списка tags
print(input_size, output_size)  # Вывод значений переменных input_size и output_size
class ChatDataset(Dataset):  # Определение класса ChatDataset, наследующего от класса Dataset
    def __init__(self):  # Определение метода __init__
        self.n_samples = len(X_train)  # Присвоение переменной n_samples значения длины массива X_train
        self.x_data = X_train  # Присвоение переменной x_data значения массива X_train
        self.y_data = y_train  # Присвоение переменной y_data значения массива y_train
    def __getitem__(self, index):  # Определение метода __getitem__
        return self.x_data[index], self.y_data[index]  # Возвращение элементов массивов x_data и y_data по заданному индексу
    def __len__(self):  # Определение метода __len__
        return self.n_samples  # Возвращение значения переменной n_samples
dataset = ChatDataset()  # Создание экземпляра класса ChatDataset и присвоение его переменной dataset
train_loader = DataLoader(dataset=dataset,  # Создание объекта DataLoader с аргументом dataset
                          batch_size=batch_size,  # Задание размера пакета
                          shuffle=True,  # Перемешивание данных
                          num_workers=0)  # Задание количества рабочих потоков
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Определение устройства для обучения (GPU или CPU)
model = NeuralNet(input_size, hidden_size, output_size).to(device)  # Создание экземпляра класса NeuralNet и присвоение его переменной model
criterion = nn.CrossEntropyLoss()  # Определение функции потерь
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Определение оптимизатора
for epoch in range(num_epochs):  # Итерация по эпохам
    for (words, labels) in train_loader:  # Итерация по пакетам данных
        words = words.to(device)  # Перемещение данных на устройство
        labels = labels.to(dtype=torch.long).to(device)  # Перемещение меток на устройство
        outputs = model(words)  # Получение выходных данных модели
        loss = criterion(outputs, labels)  # Вычисление функции потерь
        optimizer.zero_grad()  # Обнуление градиентов
        loss.backward()  # Обратное распространение ошибки
        optimizer.step()  # Обновление весов модели
    if (epoch+1) % 100 == 0:  # Вывод информации о потере каждые 100 эпох
        print(f'Эпохи [{epoch+1}/{num_epochs}], Потеря: {loss.item():.4f}')
print(f'Максимальная Потеря: {loss.item():.4f}')  # Вывод максимальной потери
data = {  # Создание словаря data
    "model_state": model.state_dict(),  # Сохранение состояния модели
    "input_size": input_size,  # Сохранение размера входных данных
    "hidden_size": hidden_size,  # Сохранение размера скрытого слоя
    "output_size": output_size,  # Сохранение размера выходных данных
    "all_words": all_words,  # Сохранение списка всех слов
    "tags": tags  # Сохранение списка тегов
}
files = "eukalpias.pth"  # Задание имени файла для сохранения
torch.save(data, files)  # Сохранение данных в файл
print(f'Обучение Завершенно.')  # Вывод сообщения о завершении обучения

import torch  # Импортируем библиотеку torch
import torch.nn as nn  # Импортируем модуль nn из torch
class NeuralNet(nn.Module):  # Определяем класс NeuralNet, который расширяет nn.Module
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()  # Вызываем конструктор родительского класса
        self.l1 = nn.Linear(input_size, hidden_size)  # Создаем линейный слой
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Создаем линейный слой
        self.l3 = nn.Linear(hidden_size, num_classes)  # Создаем линейный слой
        self.relu = nn.ReLU()  # Создаем функцию активации ReLU
    def forward(self, x):
        out = self.l1(x)  # Проход через первый линейный слой
        out = self.relu(out)  # Применение функции активации ReLU
        out = self.l2(out)  # Проход через второй линейный слой
        out = self.relu(out)  # Применение функции активации ReLU
        out = self.l3(out)  # Проход через третий линейный слой
        return out

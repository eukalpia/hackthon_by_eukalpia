import random  # Импорт модуля для работы с случайными числами
import json  # Импорт модуля для работы с JSON
import torch  # Импорт модуля PyTorch
from model import NeuralNet  # Импорт модуля с определением модели нейронной сети
from nltk_utils import bag_of_words, tokenize  # Импорт модулей для обработки текста
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Определение устройства для обучения модели
with open('intents.json', 'r') as json_data:  # Открытие файла intents.json для загрузки данных
    intents = json.load(json_data)  # Загрузка данных из файла
files = "eukalpias.pth"  # Путь к файлу с сохраненными параметрами модели
data = torch.load(files)  # Загрузка параметров модели из файла
input_size = data["input_size"]  # Размер входного слоя модели
hidden_size = data["hidden_size"]  # Размер скрытого слоя модели
output_size = data["output_size"]  # Размер выходного слоя модели
all_words = data['all_words']  # Список всех слов в наборе данных
tags = data['tags']  # Список всех тегов в наборе данных
model_state = data["model_state"]  # Состояние модели
model = NeuralNet(input_size, hidden_size, output_size).to(device)  # Создание экземпляра модели
model.load_state_dict(model_state)  # Загрузка состояния модели
model.eval()  # Перевод модели в режим оценки
bot_name = "Quasar"  # Имя бота
print("Готов к вопросам")  # Вывод сообщения о готовности к вопросам
while True:  # Бесконечный цикл для взаимодействия с ботом
    sentence = input("Вы: ")  # Ввод пользовательского сообщения
    if sentence == "выйти":  # Проверка на выход из программы
        break
    sentence = tokenize(sentence)  # Токенизация введенного сообщения
    X = bag_of_words(sentence, all_words)  # Преобразование токенизированного сообщения в мешок слов
    X = X.reshape(1, X.shape[0])  # Изменение формы массива входных данных
    X = torch.from_numpy(X).to(device)  # Преобразование массива входных данных в тензор PyTorch и отправка на устройство
    output = model(X)  # Получение выходных данных модели
    _, predicted = torch.max(output, dim=1)  # Получение индекса класса с наибольшей вероятностью
    tag = tags[predicted.item()]  # Получение тега, соответствующего предсказанному классу
    probs = torch.softmax(output, dim=1)  # Применение функции softmax к выходным данным модели
    prob = probs[0][predicted.item()]  # Получение вероятности предсказанного класса
    if prob.item() > 0.75:  # Проверка на достоверность предсказания
        for intent in intents['intents']:  # Поиск соответствующего тега в наборе данных
            if tag == intent["tag"]:  # Если тег найден
                print(f"{bot_name}: {random.choice(intent['responses'])}")  # Вывод случайного ответа из набора данных
    else:
        print(f"{bot_name}: Я вас не понимаю...")  # Вывод сообщения о непонимании запроса

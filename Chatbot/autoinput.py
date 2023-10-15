import random
import json

class Autoin:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        try:
            with open(self.file_path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {"intents": []}
        return data

    def save_data(self):
        with open(self.file_path, 'w') as file:
            json.dump(self.data, file, indent=2)

    def add_data(self, tag, patterns, responses):
        new_intent = {
            "tag": tag,
            "patterns": patterns,
            "responses": responses
        }
        self.data["intents"].append(new_intent)
        self.save_data()

class Chatbot:
    def __init__(self, intents_file):
        self.auto_input = Autoin(intents_file)

    def train_bot(self, tag, patterns, responses):
        self.auto_input.add_data(tag, patterns, responses)

    def chat(self):
        print("Готов к вопросам")
        while True:
            lesson_topic = input("Введите тему урока: ")
            if lesson_topic.lower() == "выйти":
                break
            correct_answer = input("Введите правильный ответ: ")

            tag = lesson_topic  # Используем тему урока в качестве тега
            patterns = [lesson_topic]  # Тема урока в паттернах
            responses = [correct_answer]  # Правильный ответ

            self.train_bot(tag, patterns, responses)
            print("Данные успешно добавлены в JSON файл.")

        print("Обучение завершено. Готов к вопросам ученика.")

# Используйте класс Chatbot для создания объекта чатбота и взаимодействия с учителем
chatbot = Chatbot('intents.json')
chatbot.chat()

from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.textfield import MDTextField
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivy.uix.scrollview import ScrollView
import random
import torch
import json
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

KV = '''
BoxLayout:
    orientation: 'vertical'
    ScrollView:
        MDLabel:
            id: chat_history
            text: "Здесь будет история чата"
            size_hint_y: None
            height: dp(400)
    MDTextField:
        id: user_input
        hint_text: "Введите ваш вопрос"
        helper_text: " "
        helper_text_mode: "on_focus"
        multiline: False
        size_hint_y: None
        height: dp(50)
    MDRaisedButton:
        text: "Отправить"
        size_hint_y: None
        height: dp(50)
        on_release: app.send_message()
'''


class ChatBotApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bot_name = "Quasar"
        self.load_model()

    def load_model(self):
        files = "eukalpias.pth"
        data = torch.load(files)
        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data['all_words']
        tags = data['tags']
        model_state = data["model_state"]

        self.model = NeuralNet(input_size, hidden_size, output_size)
        self.model.load_state_dict(model_state)
        self.model.eval()

        self.all_words = all_words
        self.tags = tags

        with open('intents.json', 'r') as json_data:
            intents = json.load(json_data)
        self.intents = intents

    def get_response(self, user_input):
        user_input = tokenize(user_input)
        X = bag_of_words(user_input, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).float()
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in self.intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
        else:
            response = "Извините, я не понимаю ваш вопрос."

        return response

    def send_message(self):
        user_input = self.root.ids.user_input.text
        response = self.get_response(user_input)
        self.root.ids.chat_history.text += f"\nВы: {user_input}\n{self.bot_name}: {response}"
        self.root.ids.user_input.text = ""  # Очистка поля ввода

    def build(self):
        return Builder.load_string(KV)


if __name__ == "__main__":
    ChatBotApp().run()

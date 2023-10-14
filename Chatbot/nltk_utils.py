import numpy as np  # Импортируем библиотеку numpy
import nltk  # Импортируем библиотеку nltk
from nltk.stem.porter import PorterStemmer  # Импортируем класс PorterStemmer из модуля nltk.stem
stems = PorterStemmer()  # Создаем экземпляр класса PorterStemmer
def tokenize(sentence):
    return nltk.word_tokenize(sentence)  # Токенизируем предложение с помощью функции word_tokenize из модуля nltk
def stem(word):
    return stems.stem(word.lower())  # Применяем стемминг к слову с помощью метода stem класса PorterStemmer
def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]  # Применяем стемминг ко всем словам в токенизированном предложении
    bag = np.zeros(len(words), dtype=np.float32)  # Создаем массив нулей заданной длины
    for idx, w in enumerate(words):
        if w in sentence_words:  # Если слово присутствует в токенизированном предложении
            bag[idx] = 1  # Устанавливаем соответствующий индекс в массиве в 1
    return bag

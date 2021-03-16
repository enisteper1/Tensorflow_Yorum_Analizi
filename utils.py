import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from TurkishStemmer import TurkishStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
import string


class Preprocess:
    def __init__(self,csv="data/hb.csv", resume=False, word_num=20000, val_perc=0.2):
        self.word_num = word_num
        self.val_perc = val_perc
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        self.stopwords = stopwords
        self.ts = TurkishStemmer()  # Kelimelerini köklerine indirgmek için kütüphane
        if resume:
            df = pd.read_csv("data/Preprocessed_hb.csv", encoding="utf-8")
            self.reviews = df["reviews"].to_list()
            self.ratings = df["ratings"].to_list()
        else:
            print("Veri ön işleme aşaması başlatılıyor. Bu kısım biraz uzun sürebilir.")
            df = pd.read_csv("data/hb.csv", encoding="utf-8")
            # Shuffle
            df = df.sample(frac=1)
            self.reviews = df["Review"].to_list()
            self.ratings = df["Rating (Star)"].to_list()
            # işlenmiş liste
            preprocessed_reviews = list()
            for i, review in enumerate(self.reviews):
                review = re.sub(r"\d+", "", review)  # Sayılar silindi
                review = review.translate(str.maketrans('', '', string.punctuation))  # Noktalama İşaretleri silindi
                review = review.lower()  # Tüm harfler küçültüldü
                review = review.split()  # Her text kelimelerine ayrıldı
                # Türkçe stopword kelimeleri silindi ve kalan kelimeler köklerine indirildi
                review = [self.ts.stem(word) for word in review if not word in set(stopwords.words("turkish"))]
                review = " ".join(review)  # kelimeler tekrardan cümle olarak birleştirildi
                preprocessed_reviews.append(review)  # Yeni listeye eklendi
            # Yenilenen bu veri seti tekrardan kullanılabilmek için kaydedildi
            df_preprocessed = pd.DataFrame(list(zip(preprocessed_reviews, self.ratings)), columns=["reviews", "ratings"])
            df_preprocessed.to_csv("data/Preprocessed_hb.csv", index=False)
            self.reviews = preprocessed_reviews
        # Veri setini olumlu ve olumsuz diye ayırmak için
        # 3'ün altı 0 (olumsuz) geri kalanlar 1 (olumlu) olarak ayarlandı
        reduced_reviews, reduced_ratings = list(), list()
        for i in range(len(self.ratings)):
            if self.ratings[i] != 3:
                reduced_reviews.append(self.reviews[i])
                reduced_ratings.append(self.ratings[i])
        self.reviews, self.ratings = reduced_reviews, reduced_ratings

        for i in range(len(self.ratings)):
            if self.ratings[i] == 1 or self.ratings[i] == 2:
                self.ratings[i] = 0
            else:
                self.ratings[i] = 1

    def tokenizing(self):
        # Her kelimeye sayı atamak için tokenizer kullanıldı
        # Buradaki num_words en çok kullanılan kelimeleri almak için belirlenen bir sınır
        self.tokenizer = Tokenizer(num_words=self.word_num)
        self.tokenizer.fit_on_texts(self.reviews)
        # Textler tokenleştirildi
        reviews_tokenized = self.tokenizer.texts_to_sequences(self.reviews)
        # Her text için kullanılan token sayısı alındı
        tokens_num = np.array([len(tokenized_review) for tokenized_review in reviews_tokenized])
        # token sayılarının ortalaması + standart sapmasının 2 katı
        # 1 inputta maksimum alınabilecek kelime sayısı olarak belirlendi
        self.max_inp_length = int(np.mean(tokens_num) + 2 * np.std(tokens_num))
        # padding işlemi yapıldı boş kalan kısımlar 0 tokeni ile dolduruldu
        reviews_padded = pad_sequences(reviews_tokenized, maxlen=self.max_inp_length)
        percentage = int((1 - self.val_perc) * len(self.reviews))
        # Train test split kısmı
        x_train, x_test = np.array(reviews_padded[:percentage]), np.array(reviews_padded[percentage:])
        y_train, y_test = np.array(self.ratings[:percentage]), np.array(self.ratings[percentage:])

        return x_train, y_train, x_test, y_test, self.max_inp_length

    def prediction_process(self, inp):
        preprocessed_inp = list()
        review = re.sub(r"\d+", "", inp)  # Sayılar silindi
        review = review.translate(str.maketrans('', '', string.punctuation))  # Noktalama İşaretleri silindi
        review = review.lower()  # Tüm harfler küçültüldü
        review = review.split()  # Her text kelimelerine ayrıldı
        # Türkçe stopword kelimeleri silindi ve kalan kelimeler köklerine indirildi
        review = [self.ts.stem(word) for word in review if not word in set(self.stopwords.words("turkish"))]
        review = " ".join(review)  # kelimeler tekrardan cümle olarak birleştirildi
        preprocessed_inp.append(review)  # Yeni listeye eklendi
        inp = self.tokenizer.texts_to_sequences(preprocessed_inp)
        inp = pad_sequences(inp, maxlen=self.max_inp_length)
        return inp


def visualize(history, epochs):
    axis1 = plt.subplot2grid((2, 1), (0, 0))
    axis2 = plt.subplot2grid((2, 1), (1, 0), sharex=axis1)
    axis1.plot(history["accuracy"], label="Accuracy")
    axis1.plot(history["val_accuracy"], label="Validation Accuracy")
    axis1.legend(loc=2)
    axis2.plot(history["loss"], label="Loss")
    axis2.plot(history["val_loss"], label="Validation Loss")
    axis2.legend(loc=2)
    plt.savefig("results.png")
    plt.show()
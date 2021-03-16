import os
import numpy as np
import pandas as pd
import argparse
from utils import *
from model import create_model
import tensorflow as tf


def train(args):
    data_folder, ckpt, resume, epochs, val_perc, word_num, output_dim = args.data_folder, args.ckpt, args.resume,\
                                                                        args.epochs, args.val_perc, args.word_num, args.output_dim
    # Ön işleme için class tanıtıldı
    preprocesser = Preprocess(csv=data_folder,
              resume=resume,
              word_num=word_num,
              val_perc=val_perc)
    # İşlenmiş olan veriler alındı
    x_train, y_train, x_test, y_test, max_inp_length = preprocesser.tokenizing()

    print("Veri ön işleme tamamlandı!")
    model = create_model(max_inp_length, input_dim=word_num,
                         output_dim=output_dim,
                         resume=resume,
                         ckpt=ckpt)
    print("Model tanıtıldı!")
    # Ağırlıkların kaydedilmesi için ayarlar yapıldı
    checkpoint = "./checkpoint_dir/ckpt_{epoch}"
    callback = tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss",
                                                  verbose=1, save_best_only=False,
                                                  save_weights_only=True, mode="auto",
                                                  save_freq="epoch", options=None)
    # Eğitim
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=64,
                        validation_data=(x_test, y_test),
                        callbacks=[callback],
                        verbose=1)
    # Görselleştirme
    visualize(history=history.history, epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="data/hb.csv", help="Indirilen csv dosyasının konumu")
    parser.add_argument("--ckpt", type=str, default="./checkpoint_dir/ckpt_3", help="yüklenecek ağırlık dosyası")
    parser.add_argument("--resume", action="store_true", help="Eğitime baştan mı başlandığı devam mı ettiği")
    parser.add_argument("--epochs", type=int, default=10, help="Eğitilecek epoch sayısı")
    parser.add_argument("--val_perc", type=float, default=0.2, help="Validation Oranı")
    parser.add_argument("--word_num", type=int, default=20000, help="Tokenize edilecek maks kelime sayısı")
    parser.add_argument("--output_dim", type=int, default=64, help="kaçlık vektör oluşturalacağı")
    args = parser.parse_args()
    print(args)

    train(args)
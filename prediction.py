import numpy as np
import pandas as pd
import argparse
from model import create_model
from utils import Preprocess

def predict(args):
    data_folder, ckpt, word_num, output_dim, inp_length = args.data_folder, args.ckpt, args.word_num, args.output_dim, args.inp_length
    preprocesser = Preprocess(resume=True, word_num=word_num)
    preprocesser.tokenizing()
    model = create_model(inp_length, input_dim=word_num,
                         output_dim=output_dim,
                         resume=True,
                         ckpt=ckpt)

    while True:
        inp = str(input("Analiz edilecek yorumu yazın: \n"))
        inp = preprocesser.prediction_process(inp=inp)
        predicted_val = model.predict(inp)[0]
        print("Yorum olumlu" if predicted_val> 0.5 else "Yorum olumsuz!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="data/Preprocessed_hb.csv", help="Indirilen csv dosyasının konumu")
    parser.add_argument("--ckpt", type=str, default="./checkpoint_dir/ckpt_10", help="yüklenecek ağırlık dosyası")
    parser.add_argument("--word_num", type=int, default=20000, help="Tokenize edilecek maks kelime sayısı")
    parser.add_argument("--output_dim", type=int, default=64, help="kaçlık vektör oluşturalacağı")
    parser.add_argument("--inp_length", type=int, default=54, help="Inputta maksimum alınabilecek kelime sayısı")
    args = parser.parse_args()
    print(args)

    predict(args)

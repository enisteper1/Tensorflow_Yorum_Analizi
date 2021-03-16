from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import layers, optimizers


def create_model(input_length, input_dim=20000,  output_dim=64, resume=False, ckpt="checkpoint_dir/ckpt_1"):
    model = Sequential()
    # Kelimeler için 20000'e 64 genişliğinde vektör ağı oluşturuldu
    model.add(layers.Embedding(input_dim=input_dim,
                               output_dim=output_dim,
                               input_length=input_length))
    # LSTM yerine GRU da kullanılabilir
    # return_sequences=True diyerek bir sonraki LSTM layerına gelen input aktarılıyor
    model.add(layers.LSTM(32, activation="relu", return_sequences=True))
    model.add(layers.LSTM(16, activation="relu", return_sequences=True))
    # Sonrasında RNN layer gelmediği Dense layer geldiği için return_sequences=False
    model.add(layers.LSTM(8, activation="relu", return_sequences=False))
    model.add(layers.Dense(1, activation="sigmoid"))

    # Adam yerine SGD de kullanılabilir
    optimizer = optimizers.Adam(lr=0.001)
    # 2 class olduğu için BinaryCrossentropy kullanıldı
    loss = BinaryCrossentropy()
    metrics = ["accuracy"]

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    # Eğer devam edilecek bir train var ise veya prediction işlemi için ağırlıklar güncellendi
    if resume:
        try:
            model.load_weights(ckpt)
            print("Ağırlıklar güncellendi!")
        except:
            print("Ağırlıklar güncellenemedi!")

    return model


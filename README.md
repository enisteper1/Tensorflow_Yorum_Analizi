## Bilgilendirme
    Şu an doğal dil işleme için devir GPT'nin de temelini oluşturduğu transformerlar olsa da LSTM'in kullanılmasının
    nedeni LSTM ile de güzel sonuçlar alınabildiğini göstermek.
## Train
    Train için 
    $ python train.py
    Önceden yarım kalan veya üstüne tekrar eğitim yapmak için ise 
    $ python train.py --resume --data_folder ./checkpoint_dir/ckpt_3
    Model kaç epoch eğitildiyse ckpt sayısı doğrusal olarak gidecektir. Bu yüzden 10 epoch eğitim yapıldığında
    ckpt_3 yerine ckpt_10 yazılması gerekli.
    CPU ile her epoch yaklaşık 3 dakika sürmekte.
    Epoch 1/10
    3044/3044 [==============================] - 207s 66ms/step - loss: 0.2003 - accuracy: 0.9462 - val_loss: 0.1254 - val_accuracy: 0.9579

    Epoch 00001: saving model to ./checkpoint_dir\ckpt_1
    Epoch 2/10
    3044/3044 [==============================] - 194s 64ms/step - loss: 4.3022 - accuracy: 0.9570 - val_loss: 0.2340 - val_accuracy: 0.9370

    Epoch 00002: saving model to ./checkpoint_dir\ckpt_2
    Epoch 3/10
    3044/3044 [==============================] - 195s 64ms/step - loss: 0.1566 - accuracy: 0.9454 - val_loss: 0.1463 - val_accuracy: 0.9448
<img src=https://user-images.githubusercontent.com/45767042/111273275-accdd200-8644-11eb-96e3-7680cad8da21.png>

## Prediction
    $ python prediction.py
    Bu kısım while döngüsü içinde kullanıcıdan gelen inputları tek tek değerlendiriyor.
    Örn:
    Analiz edilecek yorumu yazın:
    Çok kötü paketleme önermiyorum
    Yorum olumsuz!
    
    Analiz edilecek yorumu yazın:
    güzel ve hızlı kargo teşekkürler
    Yorum olumlu

## Kaynak
    Kullanmış olduğum yorum datasetini https://www.kaggle.com/cebeci/turkishreviews adresinden aldım.

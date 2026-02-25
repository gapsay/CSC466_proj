## 466 Final Project: Song Classification

This project was based around the problem of song classification. We used [this](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data) as our dataset. The dataset contains 10 different genres with 30 second clips of songs from those genres, totaling to 1,000 songs. 

We initially converted the songs into mel spectrograms. A spectrogram us a visual way to represent a signal's loudness as it varies over time at different frequencies. The Y Axis would be the Hz, the X axis would be the Time, and the color would be the dB. A mel spectrogram is a spectrogram where frequencies are converted to the mel scale. The mel scale is where equal distances in pitches sound equally distant to the listener. Essentially, this allows us to get rid of frequencies tha the human cannot really tell the difference between.

Once we had our spectrograms sorted into the different directories, we created a CNN model. A CNN was chosen as it is particularly good at discerning images because of a few reasons. 

1. They learn small patterns first.
CNNs look at small regions of an image at a time. This helps them detect simple local features like edges, textures, or short sound patterns instead of trying to understand everything at once.

2. They build understanding step-by-step.
CNNs have many layers, and each layer learns a slightly more complex idea than the one before it. Early layers learn simple features, while deeper layers combine those features to recognize higher-level patterns.

3. They automatically learn useful features.
Instead of humans manually designing features, CNNs figure out what matters directly from the data during training. They can discover patterns that may not be obvious to humans.

I would like to point out that we specifically used a U-Net architecture of CNNs.

#### Evaluation
Info taken from test.py script

~~~
===== Classification Report =====
              precision    recall  f1-score   support

       blues     1.0000    0.5500    0.7097        20
   classical     0.8696    1.0000    0.9302        20
     country     0.4750    0.9500    0.6333        20
       disco     0.6552    0.9500    0.7755        20
      hiphop     0.9000    0.4500    0.6000        20
        jazz     1.0000    0.7500    0.8571        20
       metal     0.9000    0.9000    0.9000        20
         pop     0.5806    0.9000    0.7059        20
      reggae     0.7895    0.7500    0.7692        20
        rock     0.5000    0.0500    0.0909        20

    accuracy                         0.7250       200
   macro avg     0.7670    0.7250    0.6972       200
weighted avg     0.7670    0.7250    0.6972       200


===== Confusion Matrix =====

[[11  0  5  2  0  0  0  0  2  0]
 [ 0 20  0  0  0  0  0  0  0  0]
 [ 0  0 19  0  0  0  0  1  0  0]
 [ 0  0  0 19  0  0  0  1  0  0]
 [ 0  0  0  2  9  0  0  8  1  0]
 [ 0  1  2  0  0 15  0  0  1  1]
 [ 0  0  2  0  0  0 18  0  0  0]
 [ 0  0  2  0  0  0  0 18  0  0]
 [ 0  0  1  1  1  0  0  2 15  0]
 [ 0  2  9  5  0  0  2  1  0  1]]
~~~

#### Articles Referenced
[Mel Spectrograms](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)
[CNNs](https://medium.com/@khwabkalra1/convolutional-neural-networks-for-image-classification-f0754f7b94aa)
[U-Net](https://medium.com/@alejandro.itoaramendia/decoding-the-u-net-a-complete-guide-810b1c6d56d8)


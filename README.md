# GAN-Bert

Implementation in Pytorch of a GAN-Bert model for fine-tuning a text classifier based on Bert in a semi-supervised way from few labelled samples. It follows the method proposed by the paper [GAN-BERT: Generative Adversarial Learning for
Robust Text Classification with a Bunch of Labeled Examples](https://www.aclweb.org/anthology/2020.acl-main.191.pdf).

## Dataset

## Dataset

The dataset used is the [questions from UIUC's CogComp QC Dataset](https://github.com/amankedia/Question-Classification).
It contains around 5000 questions in the training set classed into 50 classes.
The model will only be allowed by default to use 10 samples per classes as labelled samples.
The remaining samples will be used in a semi-supervised way againt the adversarial sample of the generator.
The number of labelled samples allowd to be used per classes can be change by modifying the value of the setting `max_labels_per_classes` in `src/config.py`.

## Training

Use the command `python3 train.py` to run training.
The model will be trained for 30 epoch then its state will be save into the file `trained.chkpt`.
With only access to 10 labelled questions per classes, it should reach around 85% accuracy on the validation set.

## Prediction

Once trained, use the command `python3 predict.py trained.chkpt "Any question to classify"` to classify a custom question.
Refer to the file `labels.txt` for more information on the meaning of a label.

```
>>> python3 predict.py trained.chkpt "what is the best way to wash the hands to avoid to be sick"
Prediction : ENTY:techmeth         # techniques and methods
>>> python3 predict.py trained.chkpt "what is the most dangerous fish on earth"
Prediction : ENTY:animal           # animals
>>> python3 predict.py trained.chkpt "what is the most delicious fish in this restaurant"
Prediction : ENTY:food             # food
>>> python3 predict.py trained.chkpt "what is the third biggest town in Japan"
Prediction : LOC:city              # city
>>> python3 predict.py trained.chkpt "How long does it take to go to the moon"
Prediction : NUM:period            # period, the lasting of time of sth.
>>> python3 predict.py trained.chkpt "When did the first man walked on the moon"
Prediction : NUM:date              # date
>>> python3 predict.py trained.chkpt "How was it possible to go to the moon im 1969"
Prediction : DESC:manner           # manner of an action
```

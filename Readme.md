# Face recognition

Application based on [keras-facenet](https://github.com/faustomorales/keras-facenet), which is wrapper of [this facenet implementation](https://github.com/davidsandberg/facenet).
Classification is performed with RadiusNeighborsClassifier from [scikit-learn](https://scikit-learn.org/).

## Running
In data folder create directories for every person you want to recognize. Put images of this person in folder with their name.

To execute program:
```
$ python3 run.py
```
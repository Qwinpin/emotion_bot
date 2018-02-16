from keras.models import load_model
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class FACENN():
    def __init__(self, path='face_test.tflearn'):
        self.model = load_model(path)

    def prediction(self, picture):
        res = self.model.predict(picture)
        return res

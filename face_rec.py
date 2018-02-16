import numpy as np
from numpy import inf
import cv2
from PIL import Image, ImageOps
import pygal
from pygal import Config
from pygal.style import Style

from face_model import FACENN
from config import model_path


def crop_face(img):
    """Find face on the image and crop to it

    Args:
        img (list): image as a array of pixels

    Returns:
        list: reshaped array for keras prediction
        bool: True if the face was found
    """
    width = img.width
    height = img.height
    
    if width < 512 or height < 512:
        loss = max(512 // width, 512 // height)
        print(loss)
        new_size = width * loss, height * loss
        img = img.resize(new_size, Image.ANTIALIAS)
        width = img.width
        height = img.height

    size = 48, 48
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

    facenn = FACENN(path=model_path)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        face_ind = False
        crop = gray
        crop = Image.fromarray(crop)
        img_to_network = crop.resize(size)
        img_to_network = np.reshape(np.array(img_to_network), [1, size[0], size[1], 1])

        prediction = facenn.prediction(img_to_network)
        bar = visualization(img, prediction)
        img = ImageOps.expand(img, border=(0, 0, bar.width, 0), fill='white')
        img.paste(bar, (width, 0, width + bar.width, bar.height), bar)
    else:
        face_ind = True
        for (x, y, w, h) in faces:
            #crop image with extracted coordinates
            crop = gray[y:y + h, x:x + w]
            crop = Image.fromarray(crop)
            img_to_network = crop.resize(size)
            img_to_network = np.reshape(np.array(img_to_network), [1, size[0], size[1], 1])

            prediction = facenn.prediction(img_to_network)
            bar = visualization(crop, prediction)
            img = ImageOps.expand(img, border=(0, 0, bar.width, 0), fill='white')
            img.paste(bar, (x+w, y, x+w+bar.width, y+h), bar)
    return img, face_ind


def visualization(img, prediction):
    """Plot list of predictions on barchart

    Args:
        img (np.array): image as array
        prediction (list): array of emotions probabilities unregulated

    Returns:
        list: barchart as array

    """
    config = Config()
    config.show_legend = False
    config.print_labels = True
    config.show_y_labels = False
    config.show_y_guides = False
    config.show_x_guides = False
    config.max_scale = 6
    config.width = img.width // 2
    config.height = img.height

    custom_style = Style(
        background='transparent',
        plot_background='transparent',
        foreground='transparent',
        colors=('#19d5ff', '#19d5ff', '#19d5ff', '#19d5ff', '#19d5ff'),
        opacity='.8',
        value_label_font_size=img.width // 30,
        value__label_font_family='monospace')

    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    result = np.log(prediction[0] + 0.00000000000000000000000000000001)  #to avoid zero exception
    result[result == -inf] = 0  #to avoid inf exception in pygal
    #this is really important for human readability
    minimum = result.min(-1)
    data = list(zip(result - minimum, labels))  #result - minimun --- normalization of prediction
    data_dict = []
    for val, label in data:
        data_dict.append({'value': val, 'label': label})
    
    bar_chart = pygal.HorizontalBar(config=config, style=custom_style)
    bar_chart.add('', data_dict)
    bar_chart.render_to_png('./tmp.png')  #pygal, no comments
    bar = Image.open('./tmp.png')
    return bar

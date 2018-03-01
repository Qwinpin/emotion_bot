from io import BytesIO

import cv2
import numpy as np
import pygal
import requests
from numpy import inf
from PIL import Image, ImageOps
from pygal import Config
from pygal.style import Style

import config
from face_model import FACENN
import send

class Processing:
    def __init__(self):
        self.facenn = FACENN()
        self.face_cascade = cv2.CascadeClassifier(config.HAAR_MODEL_PATH)
        

    def process_image(self, img):
        """Find face on the image and crop to it

        Args:
            img (list): image as a array of pixels

        Returns:
            list: reshaped array for keras prediction
            bool: True if the face was found
        """
        width = img.width
        height = img.height
        scale_factor = 1640
        #resize image to default
        if width < scale_factor or height < scale_factor:
            loss = max(scale_factor // width, scale_factor // height)
            print(loss)
            new_size = width * loss, height * loss
            img = img.resize(new_size, Image.ANTIALIAS)
            width = img.width
            height = img.height

        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        #load cnn model and haar-like cascade
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        #if no faces - use original image
        if len(faces) == 0:
            crop = gray
            crop = Image.fromarray(crop)
            img_to_network = crop
            #make prediction and create vasialisation
            prediction = self.facenn.prediction(img_to_network)
            bar = self.visualization(img, prediction)
            #expand with free space for bar chart
            img = ImageOps.expand(img, border=(0, 0, bar.width, 0), fill='white')
            img.paste(bar, (width, 0, width + bar.width, bar.height), bar)

            return [img], 0
        else:
            #the same, but for each face
            face_ind = True
            face_list = []
            for (x, y, w, h) in faces:
                #crop image with extracted coordinates
                crop = gray[y:y + h, x:x + w]
                crop = Image.fromarray(crop)
                img_to_network = crop

                img_tmp = img.crop((x, y, x+w, y+h))
                width_crop = img_tmp.width
                height_crop = img_tmp.height
                if width_crop < scale_factor or height_crop < scale_factor:
                    loss = max(scale_factor // width_crop, scale_factor // height_crop)
                    print(loss)
                    new_size = width_crop * loss, height_crop * loss
                    img_tmp = img_tmp.resize(new_size, Image.ANTIALIAS)
                    w = img_tmp.width
                    h = img_tmp.height

                prediction = self.facenn.prediction(img_to_network)
                bar = self.visualization(img_tmp, prediction)
                img_crop = ImageOps.expand(img_tmp, border=(0, 0, bar.width, 0), fill='white')
                img_crop.paste(bar, (w, 0, w+bar.width, h), bar)
                face_list.append(img_crop)
            return face_list, len(faces)
        


    def visualization(self, img, prediction):
        """Plot list of predictions on barchart

        Args:
            img (np.array): image as array
            prediction (list): array of emotions probabilities

        Returns:
            list: barchart as array

        """
        config = Config()
        config.show_legend = False
        config.print_labels = True
        config.show_y_labels = False
        config.show_y_guides = False
        config.show_x_guides = False
        config.max_scale = 12
        config.width = img.width
        config.height = img.height

        custom_style = Style(
            background='transparent',
            plot_background='transparent',
            foreground='transparent',
            colors=('#19d5ff', '#19d5ff', '#19d5ff', '#19d5ff', '#19d5ff'),
            opacity='.8',
            value_label_font_size=img.width // 30,
            value__label_font_family='monospace')

        labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        data = list(zip(prediction, labels))
        data_dict = []
        for val, label in data:
            data_dict.append({'value': val, 'label': label})
        
        bar_chart = pygal.HorizontalBar(config=config, style=custom_style)
        bar_chart.add('', data_dict)
        imgByteArr = BytesIO()
        bar_chart.render_to_png(imgByteArr)  #pygal, no comments
        #bar = Image.open('./tmp.png')
        bar = Image.open(imgByteArr)
        return bar


def support(url, chat_id):
    photo = requests.get(url)
    if photo.status_code == 200:

        image = Image.open(BytesIO(photo.content))
        process_instance = Processing()
        send_instance = send.Sender()

        result_image, find_face = process_instance.process_image(image)
        for face in result_image:
            imgByteArr = BytesIO()
            face.save(imgByteArr, format='PNG')
            imgByteArr = imgByteArr.getvalue()
            send_instance.send_photo(chat_id, imgByteArr)
            if find_face != 0:
                send_instance.send_notice(chat_id, 'Detected {0} faces'.format(find_face))
            else:
                send_instance.send_notice(chat_id, 'Is anybody here?')
    else:
        send_instance.send_notice(chat_id, 'Could not download file')
        config.logger.INFO('Problem with request for image')
        
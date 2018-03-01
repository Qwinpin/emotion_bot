import logging
import telebot


TOKEN = 'XXX:XXXX'
MODEL_PATH = './model/trained.pt'
HAAR_MODEL_PATH = './model/haarcascade_frontalface_default.xml'

handler = logging.FileHandler("log.log")
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)d'
)

logger = logging.getLogger('default')
logger.setLevel(logging.INFO)

handler.setFormatter(formatter)
logger.addHandler(handler)

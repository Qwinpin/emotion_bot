import telebot
from config import TOKEN

class Sender:
    def __init__(self):
        self.bot = telebot.TeleBot(TOKEN)

    def send_photo(self, chat_id, img):
        self.bot.send_photo(chat_id, img, 
            caption='Hope u enjoy')

    def send_notice(self, chat_id, text):
        self.bot.send_message(chat_id, text)
        
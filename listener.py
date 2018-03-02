from io import BytesIO
import logging

import telebot

import face_rec
import config


bot = telebot.TeleBot(config.TOKEN)


@bot.message_handler(commands=['start'])
def handle_start_help(message):
    bot.reply_to(message, "I'm ready, Sonne! Send me your selfie and I'll recognize you emotion")
    config.logger.info('New customer!')


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    bot.reply_to(message, "I'm looking for with photo, wait a few seconds")
    file_info = bot.get_file(message.photo[-1].file_id)
    face_rec.support('https://api.telegram.org/file/bot{0}/{1}'.format(
        TOKEN, file_info.file_path), message.chat.id)
    config.logger.info('New request!')

bot.polling(none_stop=True)

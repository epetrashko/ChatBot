import logging
import os
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from bot import ChatBot

PORT = int(os.environ.get('PORT', 443))
TOKEN = "5324171051:AAGdPccBNvGEtPDTPOnaBRtiMM5eiNeK2vg"

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

chat_bot = ChatBot()


def start(update, _):
    update.message.reply_text('Hi! Write some message')


def respond(update, _):
    text = chat_bot.get_response(update.message.text)
    if text is not None:
        update.message.reply_text(text)
    else:
        update.message.reply_text("Didn't get your question")


def error(update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    updater = Updater(TOKEN, use_context=True)

    dp = updater.dispatcher

    dp.add_error_handler(error)

    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, respond))
    dp.add_handler(CommandHandler("start", start))

    # Use it when you want to deploy bot on heroku
    # updater.start_webhook(listen="0.0.0.0",
    #                       port=int(PORT),
    #                       url_path=TOKEN,
    #                       webhook_url='https://brawler-chat-bot.herokuapp.com/' + TOKEN)

    # Use it when you want to test on local machine
    updater.start_polling()

    updater.idle()


if __name__ == '__main__':
    main()

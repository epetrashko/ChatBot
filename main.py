#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""
import logging
import telegram
import os
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler

PORT = int(os.environ.get('PORT', 443))
TOKEN = "5324171051:AAGdPccBNvGEtPDTPOnaBRtiMM5eiNeK2vg"

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def start(update, _):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi! Write some message')


def echo(update, _):
    update.message.reply_text(update.message.text)


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    """Start the bot."""
    updater = Updater(TOKEN, use_context=True)

    dp = updater.dispatcher

    dp.add_error_handler(error)

    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))
    dp.add_handler(CommandHandler("start", start))

    # Start the Bot
    updater.start_webhook(listen="0.0.0.0",
                          port=int(PORT),
                          url_path=TOKEN,
                          webhook_url='https://brawler-chat-bot.herokuapp.com/' + TOKEN)
    updater.start_polling()

    updater.idle()


if __name__ == '__main__':
    bot = telegram.Bot(token=TOKEN)
    main()

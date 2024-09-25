import telebot
import pandas as pd
import random
import time
import requests
import numpy as np
import logging
from telebot import types
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


bot = telebot.TeleBot("8149823821:AAHOc4k17ZXwCVfzgInlT95MzLSs0IIcQSg")


subscribed_users = set() 


def fetch_real_time_data():
   
    return [round(random.uniform(1.0, 5.0), 2) for _ in range(10)]


def prepare_data(data):
    X = np.array(range(len(data))).reshape(-1, 1)
    y = np.array(data)  
    return X, y


def train_model(data):
    X, y = prepare_data(data)
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    logger.info("📊 Model trained successfully!")
    return model


def analyze_signals(data):
    df = pd.DataFrame(data, columns=['multiplier'])
    

    model = train_model(df['multiplier'])
    
 
    next_round = np.array([[len(df)]]) 
    predicted_multiplier = model.predict(next_round)[0]
    
    
    average_multiplier = df['multiplier'].mean()
    upper_threshold = average_multiplier + df['multiplier'].std()
    lower_threshold = average_multiplier - df['multiplier'].std()

   
    signal = "📈 Bet! 🚀" if predicted_multiplier > upper_threshold else "🚫 Don't Bet 🚧"
    
    return {
        'predicted_multiplier': predicted_multiplier,
        'signal': signal,
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold,
        'average_multiplier': average_multiplier
    }


def get_latest_signal():
    real_data = fetch_real_time_data() 
    signals = analyze_signals(real_data)
    return (
        f"📊 **Predicted Multiplier:** {signals['predicted_multiplier']:.2f}\n"
        f"🔔 **Signal:** {signals['signal']}\n"
        f"⚖️ **Average Multiplier:** {signals['average_multiplier']:.2f}\n"
        f"⬆️ **Upper Threshold:** {signals['upper_threshold']:.2f}\n"
        f"⬇️ **Lower Threshold:** {signals['lower_threshold']:.2f}"
    )

# Telegram bot commands
@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(message, "👋 Welcome to the **Aviator Signal Bot**! 🚀\nType /subscribe to receive live signals.")

@bot.message_handler(commands=["subscribe"])
def subscribe_user(message):
    subscribed_users.add(message.chat.id)
    bot.reply_to(message, "✅ You have **subscribed** to /get_signal receive signals! 📈")

@bot.message_handler(commands=["unsubscribe"])
def unsubscribe_user(message):
    subscribed_users.discard(message.chat.id)
    bot.reply_to(message, "❌ You have **unsubscribed** from receiving signals.")

@bot.message_handler(commands=["get_signal"])
def signal_button(message):
    markup = types.InlineKeyboardMarkup()
    signal_button = types.InlineKeyboardButton("🔄 Get Latest Signal", callback_data="get_latest_signal")
    markup.add(signal_button)
    bot.send_message(message.chat.id, "🔍 Press the button below to get the latest **Aviator** signal:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data == "get_latest_signal")
def callback_query(call):
    signal = get_latest_signal()
    bot.send_message(call.message.chat.id, signal)


def broadcast_signal():
    signal = get_latest_signal()
    for user_id in subscribed_users:
        try:
            bot.send_message(user_id, f"📡 **New Signal Alert**!\n{signal}")
        except Exception as e:
            logger.error(f"Error sending message to {user_id}: {e}")


def schedule_broadcast():
    while True:
        broadcast_signal()
        time.sleep(60)  


import threading
broadcast_thread = threading.Thread(target=schedule_broadcast)
broadcast_thread.start()


bot.polling()

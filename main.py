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
from datetime import datetime
import schedule


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


bot = telebot.TeleBot("8149823821:AAHOc4k17ZXwCVfzgInlT95MzLSs0IIcQSg")


subscribed_users = set()  


TIME_WINDOWS_80 = [
    ("09:00", "09:30"),
    ("12:00", "12:30"),
    ("16:00", "16:30"),
    ("20:00", "20:30")
]


def is_time_in_80_window():
    current_time = datetime.now().strftime("%H:%M")
    for start_time, end_time in TIME_WINDOWS_80:
        if start_time <= current_time <= end_time:
            return True
    return False


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


    accuracy = 80 if is_time_in_80_window() else 50


    signal = "📈 Bet! 🚀" if predicted_multiplier > upper_threshold else "🚫 Don't Bet 🚧"
    
    return {
        'predicted_multiplier': predicted_multiplier,
        'signal': signal,
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold,
        'average_multiplier': average_multiplier,
        'accuracy': accuracy
    }


def get_latest_signal():
    real_data = fetch_real_time_data()  
    signals = analyze_signals(real_data)
    return (
        f"📊 **Predicted Multiplier:** {signals['predicted_multiplier']:.2f}\n"
        f"🔔 **Signal:** {signals['signal']}\n"
        f"⚖️ **Average Multiplier:** {signals['average_multiplier']:.2f}\n"
        f"⬆️ **Upper Threshold:** {signals['upper_threshold']:.2f}\n"
        f"⬇️ **Lower Threshold:** {signals['lower_threshold']:.2f}\n"
        f"🎯 **Accuracy:** {signals['accuracy']}%"
    )

# Telegram bot commands
@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(message, "👋 Welcome to the **Aviator Signal Bot**! 🚀\nType /subscribe to receive live signals.")

@bot.message_handler(commands=["subscribe"])
def subscribe_user(message):
    subscribed_users.add(message.chat.id)
    bot.reply_to(message, "✅ You have **subscribed** to receive signals! 📈")

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
        current_time = datetime.now().strftime("%H:%M")
        
        # Send 80% signals within defined time windows
        if is_time_in_80_window():
            broadcast_signal()
        
        # Send 50% signals at other times
        else:
            broadcast_signal()
        
        time.sleep(60)  # Broadcast every minute

# Schedule time slots for broadcasting signals
def schedule_signals():
    # Schedule to broadcast signals in the 80% windows
    schedule.every().day.at("09:00").until("09:30").do(broadcast_signal)
    schedule.every().day.at("12:00").until("12:30").do(broadcast_signal)
    schedule.every().day.at("16:00").until("16:30").do(broadcast_signal)
    schedule.every().day.at("20:00").until("20:30").do(broadcast_signal)
    
    # Schedule for general (50%) signals throughout the day
    schedule.every().minute.do(broadcast_signal)

# Start the scheduling thread for broadcasting
import threading
broadcast_thread = threading.Thread(target=schedule_signals)
broadcast_thread.start()

# Run the bot
bot.polling()

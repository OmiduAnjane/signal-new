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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Initialize the bot with your token
bot = telebot.TeleBot("8149823821:AAHOc4k17ZXwCVfzgInlT95MzLSs0IIcQSg")

# Global variable to store subscribed users
subscribed_users = set()  # A set to hold unique user IDs

# Function to fetch real-time data (placeholder function)
def fetch_real_time_data():
    # Replace this with the actual API call to fetch real data
    # Here we simulate data for demonstration purposes
    return [round(random.uniform(1.0, 5.0), 2) for _ in range(10)]

# Function to prepare data for machine learning
def prepare_data(data):
    X = np.array(range(len(data))).reshape(-1, 1)  # Round index as feature
    y = np.array(data)  # Multiplier as target
    return X, y

# Function to train a linear regression model
def train_model(data):
    X, y = prepare_data(data)
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Model trained successfully.")
    return model

# Function to analyze signals using the trained model
def analyze_signals(data):
    df = pd.DataFrame(data, columns=['multiplier'])
    
    # Generate signals based on prediction
    model = train_model(df['multiplier'])
    
    # Predict next multiplier
    next_round = np.array([[len(df)]])  # Predicting for the next round
    predicted_multiplier = model.predict(next_round)[0]
    
    # Set thresholds for betting signals
    average_multiplier = df['multiplier'].mean()
    upper_threshold = average_multiplier + df['multiplier'].std()
    lower_threshold = average_multiplier - df['multiplier'].std()

    # Generate signals based on thresholds and predicted value
    signal = "Bet" if predicted_multiplier > upper_threshold else "Don't Bet"
    
    return {
        'predicted_multiplier': predicted_multiplier,
        'signal': signal,
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold,
        'average_multiplier': average_multiplier
    }

# Function to get the latest signal
def get_latest_signal():
    real_data = fetch_real_time_data()  # Fetch real data (simulated)
    signals = analyze_signals(real_data)
    return (
        f"Predicted Multiplier: {signals['predicted_multiplier']:.2f}\n"
        f"Signal: {signals['signal']}\n"
        f"Average Multiplier: {signals['average_multiplier']:.2f}\n"
        f"Upper Threshold: {signals['upper_threshold']:.2f}\n"
        f"Lower Threshold: {signals['lower_threshold']:.2f}"
    )

# Telegram bot commands
@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(message, "Welcome to the 1xBet Aviator Signal Bot! Type /subscribe to receive signals.")

@bot.message_handler(commands=["subscribe"])
def subscribe_user(message):
    subscribed_users.add(message.chat.id)
    bot.reply_to(message, "You have subscribed to receive signals!")

@bot.message_handler(commands=["unsubscribe"])
def unsubscribe_user(message):
    subscribed_users.discard(message.chat.id)
    bot.reply_to(message, "You have unsubscribed from receiving signals.")

@bot.message_handler(commands=["get_signal"])
def signal_button(message):
    markup = types.InlineKeyboardMarkup()
    signal_button = types.InlineKeyboardButton("Get Latest Aviator Signal", callback_data="get_latest_signal")
    markup.add(signal_button)
    bot.send_message(message.chat.id, "Press the button below to get the latest Aviator signal:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data == "get_latest_signal")
def callback_query(call):
    signal = get_latest_signal()
    bot.send_message(call.message.chat.id, signal)

# Function to broadcast the latest signal to all subscribed users
def broadcast_signal():
    signal = get_latest_signal()
    for user_id in subscribed_users:
        try:
            bot.send_message(user_id, signal)
        except Exception as e:
            logger.error(f"Error sending message to {user_id}: {e}")

# Schedule the broadcast (every 60 seconds in this example)
def schedule_broadcast():
    while True:
        broadcast_signal()
        time.sleep(60)  # Change this to the desired interval in seconds

# Start a thread for broadcasting signals
import threading
broadcast_thread = threading.Thread(target=schedule_broadcast)
broadcast_thread.start()

# Run the bot
bot.polling()

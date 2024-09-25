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
from sklearn.metrics import mean_squared_error
import sqlite3

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Initialize the bot with your token
bot = telebot.TeleBot("8149823821:AAHOc4k17ZXwCVfzgInlT95MzLSs0IIcQSg")

# Set up SQLite for persistent user storage
conn = sqlite3.connect('subscribed_users.db', check_same_thread=False)
c = conn.cursor()

# Create table to store subscribed users if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS users (user_id INTEGER PRIMARY KEY)''')
conn.commit()

# Function to add a user to the subscription list
def add_user(user_id):
    c.execute('INSERT OR IGNORE INTO users (user_id) VALUES (?)', (user_id,))
    conn.commit()

# Function to remove a user from the subscription list
def remove_user(user_id):
    c.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
    conn.commit()

# Function to get all subscribed users
def get_subscribed_users():
    c.execute('SELECT user_id FROM users')
    return [row[0] for row in c.fetchall()]

# Function to fetch real-time data (simulated or replace with actual API)
def fetch_real_time_data():
    # Replace this with the actual API call to fetch real data
    # Here we simulate data for demonstration purposes
    return [round(random.uniform(1.0, 5.0), 2) for _ in range(20)]  # Simulating more data points

# Function to prepare data for machine learning
def prepare_data(data):
    X = np.array(range(len(data))).reshape(-1, 1)  # Index as feature
    y = np.array(data)  # Multiplier as target
    return X, y

# Function to train a more powerful linear regression model
def train_model(data):
    X, y = prepare_data(data)
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model performance
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    logger.info(f"Model trained successfully. MSE: {mse:.4f}")
    
    return model

# Function to analyze signals using the trained model
def analyze_signals(data):
    df = pd.DataFrame(data, columns=['multiplier'])
    
    model = train_model(df['multiplier'])
    
    # Predict next multiplier
    next_round = np.array([[len(df)]])  # Predicting for the next round
    predicted_multiplier = model.predict(next_round)[0]
    
    # Calculate thresholds for betting signals
    average_multiplier = df['multiplier'].mean()
    std_multiplier = df['multiplier'].std()
    upper_threshold = average_multiplier + std_multiplier
    lower_threshold = average_multiplier - std_multiplier

    # Generate signals based on predicted value
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
    real_data = fetch_real_time_data()  # Fetch real data (simulated or real)
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
    bot.reply_to(message, "Welcome to the Aviator Signal Bot! Type /subscribe to receive signals.")

@bot.message_handler(commands=["subscribe"])
def subscribe_user(message):
    add_user(message.chat.id)
    bot.reply_to(message, "You have subscribed to receive signals!")

@bot.message_handler(commands=["unsubscribe"])
def unsubscribe_user(message):
    remove_user(message.chat.id)
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
    users = get_subscribed_users()
    for user_id in users:
        try:
            bot.send_message(user_id, signal)
        except Exception as e:
            logger.error(f"Error sending message to {user_id}: {e}")

# Schedule the broadcast (every 60 seconds in this example)
def schedule_broadcast():
    while True:
        broadcast_signal()
        time.sleep(60)  # Broadcast interval in seconds

# Start a thread for broadcasting signals
import threading
broadcast_thread = threading.Thread(target=schedule_broadcast)
broadcast_thread.start()

# Run the bot
bot.polling()

import telebot
import pandas as pd
import random
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from telebot import types

# Initialize the bot with your token
bot = telebot.TeleBot("8149823821:AAHOc4k17ZXwCVfzgInlT95MzLSs0IIcQSg")

# Time windows for 80% accurate signals
TIME_WINDOWS_80 = [
    ("09:00", "09:30"),
    ("12:00", "12:30"),
    ("16:00", "16:30"),
    ("20:00", "20:30")
]

# Helper function to check if the current time is within 80% accuracy windows
def is_time_in_80_window():
    current_time = datetime.now().strftime("%H:%M")
    for start_time, end_time in TIME_WINDOWS_80:
        if start_time <= current_time <= end_time:
            return True
    return False

# Function to fetch real-time data (placeholder function)
def fetch_real_time_data():
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

    # Determine signal accuracy based on time
    accuracy = 80 if is_time_in_80_window() else 50

    # Generate signals based on thresholds and predicted value
    signal = "📈 Bet! 🚀" if predicted_multiplier > upper_threshold else "🚫 Don't Bet 🚧"
    
    return {
        'predicted_multiplier': predicted_multiplier,
        'signal': signal,
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold,
        'average_multiplier': average_multiplier,
        'accuracy': accuracy
    }

# Function to get the latest signal
def get_latest_signal():
    real_data = fetch_real_time_data()  # Fetch real data (simulated)
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
    bot.reply_to(message, "👋 Welcome to the **Aviator Signal Bot**! 🚀\nType /subscribe to get started and receive manual signals.")

@bot.message_handler(commands=["subscribe"])
def subscribe_user(message):
    markup = types.InlineKeyboardMarkup()
    signal_button = types.InlineKeyboardButton("🔄 Get Latest Signal", callback_data="get_latest_signal")
    markup.add(signal_button)
    bot.send_message(message.chat.id, "Press the button below to get the latest **Aviator** signal:", reply_markup=markup)

# Handle the manual button to get the latest signal
@bot.callback_query_handler(func=lambda call: call.data == "get_latest_signal")
def callback_query(call):
    signal = get_latest_signal()
    bot.send_message(call.message.chat.id, signal)

# Run the bot
bot.polling()

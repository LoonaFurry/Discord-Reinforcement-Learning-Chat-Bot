import discord
from discord.ext import commands, tasks
import asyncio
import os
import logging
import google.generativeai as genai, text
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer, util  # Not using embeddings for now
from collections import defaultdict
import aiosqlite
import time
from functools import lru_cache
from prometheus_client import start_http_server, Counter, Histogram, Summary
from google.api_core import exceptions

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration - Use environment variables for security
discord_token = ("discord-bot-token")
gemini_api_key = ("gemini-api-key")
if not discord_token or not gemini_api_key:
    raise ValueError("DISCORD_BOT_TOKEN or GEMINI_API_KEY not set in environment variables")

# Configure the Gemini API
genai.configure(api_key=gemini_api_key)

# Define intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Default response if Gemini API fails
DEFAULT_RESPONSE = "Sorry, I couldn't answer this question."

# Directory and Database File Setup
CODE_DIR = os.path.dirname(__file__)
DB_FILE = os.path.join(CODE_DIR, 'chat_history.db')

# Prometheus Metrics Setup
start_http_server(8000)

# Prometheus metrics
message_counter = Counter('discord_bot_messages_total', 'Total messages processed')
error_counter = Counter('discord_bot_errors_total', 'Total errors')
response_time_histogram = Histogram('discord_bot_response_time_seconds', 'Response times')
response_time_summary = Summary('discord_bot_response_time_summary', 'Summary of response times')

# Context Window Size 
CONTEXT_WINDOW_SIZE = 5000

# User Profile Management 
user_profiles = defaultdict(lambda: {"preferences": {}, "history_summary": ""})

# Initialize SQLite DB
async def init_db():
    """Initializes the SQLite database and creates the 'chat_history' table 
       if it doesn't exist.
    """
    db_exists = os.path.exists(DB_FILE)

    async with aiosqlite.connect(DB_FILE) as db:
        if not db_exists:
            # New database file: Create the table
            logging.info("Database file not found. Creating a new one...")
            await db.execute('''
                CREATE TABLE chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    message TEXT,
                    timestamp TEXT,
                    user_name TEXT,
                    bot_id TEXT,
                    bot_name TEXT
                ) 
            ''')
            await db.commit()
            logging.info("'chat_history' table created and initialized.")

        else:
            # Existing database file: Check for table and create if necessary
            logging.info("Database file found. Connecting...")
            async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_history'") as cursor:
                table_exists = await cursor.fetchone()
                if not table_exists:
                    await db.execute('''
                        CREATE TABLE chat_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id TEXT,
                            message TEXT,
                            timestamp TEXT,
                            user_name TEXT,
                            bot_id TEXT,
                            bot_name TEXT
                        )
                    ''')
                    await db.commit()
                    logging.info("'chat_history' table created.")
                else:
                    logging.info("'chat_history' table already exists.")
                    
# Create a queue for database operations
db_queue = asyncio.Queue()

# Save chat history
async def save_chat_history(user_id, message, user_name, bot_id, bot_name):
    await db_queue.put((user_id, message, user_name, bot_id, bot_name))

# Process database operations 
async def process_db_queue():
    while True:
        user_id, message, user_name, bot_id, bot_name = await db_queue.get()
        try:
            async with aiosqlite.connect(DB_FILE) as db:
                await db.execute(
                    'INSERT INTO chat_history (user_id, message, timestamp, user_name, bot_id, bot_name) VALUES (?, ?, ?, ?, ?, ?)',
                    (user_id, message, datetime.now(timezone.utc).isoformat(), user_name, bot_id, bot_name)
                )
                await db.commit()
        except Exception as e:
            logging.error(f"Error saving to database: {e}")
        finally:
            db_queue.task_done()

# Get relevant chat history
async def get_relevant_history(user_id, current_message):
    history_text = ""
    current_tokens = 0
    messages = []

    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute(
                'SELECT message FROM chat_history WHERE user_id = ? ORDER BY id DESC',
                (user_id,),
        ) as cursor:
            async for row in cursor:
                messages.append(row[0])

    messages.reverse() 

    for message_text in messages:
        message_tokens = len(message_text.split())
        if current_tokens + message_tokens > CONTEXT_WINDOW_SIZE:
            break
        history_text += message_text + "\n"
        current_tokens += message_tokens

    return history_text

# Asynchronous function to ask Gemini API
async def ask_gemini(prompt):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )
    chat_session = model.start_chat(history=[])
    try:
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        logging.error("API exception: %s", e)
        return DEFAULT_RESPONSE

# Event handler when bot is ready
@bot.event
async def on_ready():
    global gemini_model
    gemini_model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    logging.info(f'Logged in as {bot.user}')
    await init_db()
    change_status.start()
    bot.loop.create_task(process_db_queue())  

# Event handler for incoming messages
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    user_id = str(message.author.id)
    user_name = str(message.author)
    bot_id = str(bot.user.id)
    bot_name = str(bot.user.name)

    await save_chat_history(user_id, message.content, user_name, bot_id, bot_name)
    message_counter.inc()

    if bot.user.mentioned_in(message):
        await respond_to_mention(message, user_id)

# Function to respond to mentions
async def respond_to_mention(message, user_id):
    content = message.content
    mention = message.author.mention
    user_profile = user_profiles[user_id]

    relevant_history = await get_relevant_history(user_id, content)
    user_profile["history_summary"] = relevant_history

    prompt = (
        f"You are a Furry Young Protogen, and you're lovely, kind, patient, cute, and understanding. "
        f"Remember all previous chats. Here is the relevant chat history:\n{relevant_history}\n"
        f"Respond to the following message from {mention}: {content}"
    )

    try:
        start_time = time.time()
        response = await ask_gemini(prompt)
        end_time = time.time()
        response_time = end_time - start_time
        response_time_histogram.observe(response_time)
        response_time_summary.observe(response_time)
        await message.channel.send(f"{mention} {response}")
    except Exception as e:
        error_counter.inc()
        logging.error(f"Error processing message: {e}")
        await message.channel.send(f"{mention} Bir hata oluştu. Lütfen daha sonra tekrar deneyin.")

# Call this function during bot initialization
async def main():
    await init_db()
    
# Task to change bot status
@tasks.loop(seconds=60)
async def change_status():
    await bot.change_presence(activity=discord.Game(name="LolbitFurry's Chat Bot"))

# Main entry point
async def main():
    await bot.start(discord_token)

if __name__ == "__main__":
    asyncio.run(main())

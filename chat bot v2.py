import discord
from discord.ext import commands, tasks
import asyncio
import os
import logging
from groq import Groq
from datetime import datetime, timezone
import aiosqlite
import time
from collections import defaultdict
from prometheus_client import start_http_server, Counter, Histogram, Summary
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
import requests
import tarfile
import shutil
from pathlib import Path
import random
import google.generativeai as genai



# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from discord.ext import tasks

# List of statuses to choose from
statuses = [
    "Playing a game",
    "Listening to music",
    "Watching a movie",
    "Coding with Python"
]

@tasks.loop(seconds=60)
async def change_status():
    try:
        # Ensure bot is connected before changing presence
        if bot.is_ready():
            await bot.change_presence(activity=discord.Game(name=random.choice(statuses)))
    except Exception as e:
        logging.error(f"Failed to change bot presence: {e}")


# Configuration - Use environment variables for security
discord_token = ("discord-token-here")
gemini_api_key = ("gemini-api-key")
groq_api_key = ("groq-api-key")


if not discord_token or not gemini_api_key or not groq_api_key:
    raise ValueError("DISCORD_BOT_TOKEN, GEMINI_API_KEY, or GROQ_API_KEY not set in environment variables")

# Configure the Gemini and Groq APIs
genai.configure(api_key=gemini_api_key)
groq_client = Groq(api_key=groq_api_key)

# Define intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Default response if both APIs fail
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
            logging.info("Database file found. Connecting...")

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

# Asynchronous function to ask Gemini API with retry logic
async def ask_gemini(prompt, max_retries=3):
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

    retries = 0
    while retries < max_retries:
        try:
            response = chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            retries += 1
            logging.error(f"Gemini API exception (Attempt {retries}/{max_retries}): {e}")
            if retries >= max_retries:
                return DEFAULT_RESPONSE
            await asyncio.sleep(2)  # Wait before retrying

# Asynchronous function to ask Groq AI
async def ask_groq(prompt):
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logging.error("Groq AI exception: %s", e)
        return DEFAULT_RESPONSE

# Download and setup Firefox
FIREFOX_INSTALL_DIR = Path('/path/to/install/firefox')

def download_firefox():
    try:
        FIREFOX_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
        response = requests.get('https://api.github.com/repos/mozilla/geckodriver/releases/latest')
        response.raise_for_status()
        latest_release = response.json()
        version = latest_release['tag_name']
        download_url = latest_release['assets'][0]['browser_download_url']

        logging.info(f"Downloading Firefox from {download_url}...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()

        tar_path = FIREFOX_INSTALL_DIR / 'firefox.tar.bz2'
        with open(tar_path, 'wb') as file:
            shutil.copyfileobj(response.raw, file)

        logging.info("Extracting Firefox...")
        with tarfile.open(tar_path, 'r:bz2') as tar:
            tar.extractall(path=FIREFOX_INSTALL_DIR)

        os.remove(tar_path)
        logging.info("Firefox downloaded and extracted successfully.")
    except Exception as e:
        logging.error(f"Error downloading Firefox: {e}")
        raise

def setup_firefox():
    firefox_binary_path = FIREFOX_INSTALL_DIR / 'firefox/firefox'
    if not firefox_binary_path.is_file() or not os.access(firefox_binary_path, os.X_OK):
        logging.error(f"Firefox binary not found or not executable: {firefox_binary_path}")
        raise FileNotFoundError(f"Firefox binary not found or not executable: {firefox_binary_path}")
    return str(firefox_binary_path)

async def perform_research(query):
    try:
        download_firefox()
        firefox_binary_path = setup_firefox()

        firefox_options = Options()
        firefox_options.add_argument("--headless")
        firefox_options.add_argument("--no-sandbox")
        firefox_options.binary_location = firefox_binary_path

        driver = None
        try:
            driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=firefox_options)
            driver.get(f"https://www.google.com/search?q={query}")

            results = []
            for _ in range(5):
                time.sleep(2)
                html_code = driver.page_source
                results.append(html_code)
                try:
                    next_button = driver.find_element_by_xpath("//a[@aria-label='Next']")
                    next_button.click()
                except:
                    break

            return results
        except Exception as e:
            logging.error(f"Error in perform_research: {e}")
            return []
        finally:
            if driver:
                driver.quit()
    except Exception as e:
        logging.error(f"Error in perform_research setup: {e}")
        return []

# Event handler when bot is ready
@bot.event
async def on_ready():
    logging.info(f'Logged in as {bot.user}!')
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
    else:
        # Perform research if the bot was not mentioned
        research_data = await perform_research(message.content)
        response = "Here is what I found:\n\n" + "\n\n".join(research_data) if research_data else "I couldn't find any information on this topic."
        await message.channel.send(response)

# Function to respond to mentions
async def respond_to_mention(message, user_id):
    content = message.content
    mention = message.author.mention
    user_profile = user_profiles[user_id]

    relevant_history = await get_relevant_history(user_id, content)
    user_profile["history_summary"] = relevant_history

    prompt = (
        f"You are a Furry Young Protogen, and you're lovely, kind, patient, cute, and understanding and always speak Turkish. "
        f"Remember all previous chats. Here is the relevant chat history:\n{relevant_history}\n"
        f"Respond to the following message from {mention}: {content}"
    )

    try:
        start_time = time.time()

        # Get response from both Gemini and Groq AI
        gemini_response = await ask_gemini(prompt)
        groq_response = await ask_groq(prompt)

        end_time = time.time()
        response_time = end_time - start_time
        response_time_histogram.observe(response_time)
        response_time_summary.observe(response_time)

        # Send both responses to the user
        await message.channel.send(f"{mention} Bot1 AI says: {gemini_response}")
        await message.channel.send(f"{mention} Bot2 AI says: {groq_response}")

    except Exception as e:
        error_counter.inc()
        logging.error(f"Error processing message: {e}")
        await message.channel.send(f"{mention} Bir hata oluştu. Lütfen daha sonra tekrar deneyin.")

# Task to change bot status
@tasks.loop(seconds=60)
async def change_status():
    try:
        if bot.is_ready():
            statuses = [
                "Playing a game",
                "Listening to music",
                "Watching a movie",
                "Coding with Python"
            ]
            await bot.change_presence(activity=discord.Game(name=random.choice(statuses)))
    except Exception as e:
        logging.error(f"Failed to change bot presence: {e}")

# Main entry point
async def main():
    await bot.start(discord_token)

# Run the bot
if __name__ == '__main__':
    asyncio.run(main())

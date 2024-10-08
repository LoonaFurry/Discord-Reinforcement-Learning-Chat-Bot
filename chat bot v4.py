import discord
from discord.ext import commands, tasks
import asyncio
import os
import logging
from dotenv import load_dotenv
import aiosqlite
import time
from collections import defaultdict
from prometheus_client import start_http_server, Counter, Histogram, Summary, Gauge
from duckduckgo_search import AsyncDDGS
import google.generativeai as genai
from datetime import datetime, timezone
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Uncomment this line
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
discord_token = ("discord-bot-token")
gemini_api_key = ("gemini-key-here")

if not discord_token or not gemini_api_key:
    raise ValueError("DISCORD_BOT_TOKEN or GEMINI_API_KEY not set in environment variables")

# Configure Gemini AI
genai.configure(api_key=gemini_api_key)

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-exp-0827",
    generation_config=generation_config,
)

# Discord Bot configuration
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Directory and Database Setup
CODE_DIR = os.path.dirname(__file__)
DB_FILE = os.path.join(CODE_DIR, 'chat_history.db')
USER_PROFILES_FILE = os.path.join(CODE_DIR, "user_profiles.json")

# Prometheus metrics
start_http_server(8000)
message_counter = Counter('discord_bot_messages_total', 'Total messages processed')
error_counter = Counter('discord_bot_errors_total', 'Total errors')
response_time_histogram = Histogram('discord_bot_response_time_seconds', 'Response times')
response_time_summary = Summary('discord_bot_response_time_summary', 'Summary of response times')
active_users = Gauge('discord_bot_active_users', 'Number of active users')
feedback_count = Counter('discord_bot_feedback_count', 'Number of feedback messages received')

# Context window and user profiles
CONTEXT_WINDOW_SIZE = 5000
user_profiles = defaultdict(lambda: {"preferences": {}, "history_summary": ""})
personalities = {
    "friendly": {"greeting": "Hello there!", "farewell": "Have a great day!"},
    "formal": {"greeting": "Greetings!", "farewell": "Best regards."},
    "helpful": {"greeting": "How can I assist you?", "farewell": "I hope this was helpful."}
}

# Initialize SQLite DB
async def init_db():
    db_exists = os.path.exists(DB_FILE)
    async with aiosqlite.connect(DB_FILE) as db:
        if not db_exists:
            logging.info("Creating database...")
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
            await db.execute('''
                CREATE TABLE feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    feedback TEXT,
                    timestamp TEXT
                )
            ''')
            await db.commit()
            logging.info("Database initialized.")
        else:
            # Check if feedback table exists
            async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'") as cursor:
                if not await cursor.fetchone():
                    await db.execute('''
                        CREATE TABLE feedback (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id TEXT,
                            feedback TEXT,
                            timestamp TEXT
                        )
                    ''')
                    await db.commit()
            logging.info("Database found, connecting...")

# Initialize user profiles
def load_user_profiles():
    """Loads user profiles from the JSON file."""
    if os.path.exists(USER_PROFILES_FILE):
        with open(USER_PROFILES_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

def save_user_profiles(profiles):
    """Saves user profiles to the JSON file."""
    with open(USER_PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=4)

# Save chat history to the database
db_queue = asyncio.Queue()

async def save_chat_history(user_id, message, user_name, bot_id, bot_name):
    await db_queue.put((user_id, message, user_name, bot_id, bot_name))

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

# Save feedback to the database
async def save_feedback_to_db(user_id, feedback):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "INSERT INTO feedback (user_id, feedback, timestamp) VALUES (?, ?, ?)",
            (user_id, feedback, datetime.now(timezone.utc).isoformat())
        )
        await db.commit()
    feedback_count.inc()

# Get relevant chat history for user
async def get_relevant_history(user_id, current_message):
    history_text = ""
    current_tokens = 0
    messages = []
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute(
                'SELECT message FROM chat_history WHERE user_id = ? ORDER BY id DESC',
                (user_id,)
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

# Simulate advanced reasoning with Gemini
async def perform_advanced_reasoning(query, relevant_history, summarized_search, user_personality=None):
    prompt = (
        f"You are a Furry Young Protogen who speaks Turkish.  "
        f"Respond thoughtfully, integrating both knowledge from the web and past conversations. "
        f"Here is the relevant chat history:\n{relevant_history}\n"
        f"And here is a summary of web search results:\n{summarized_search}\n"
        f"Now respond to the following message: {query}"
    )

    if user_personality:
        if user_personality == "friendly":
            prompt += "\nRespond in a friendly and approachable manner."
        elif user_personality == "formal":
            prompt += "\nRespond in a formal and professional manner."
        elif user_personality == "helpful":
            prompt += "\nRespond in a helpful and informative manner."

    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini AI reasoning exception: {e}")
        return "An error occurred while processing your request with Gemini AI."

# Asynchronous DuckDuckGo search tool
async def duckduckgotool(query) -> str:
    blob = ''
    try:
        ddg = AsyncDDGS()
        results = ddg.text(query, max_results=100)
        for index, result in enumerate(results[:100]):
            blob += f'[{index}] Title: {result["title"]}\nSnippet: {result["body"]}\n\n'
    except Exception as e:
        blob += f"Search error: {e}\n"
    return blob

# Analyze feedback from database
async def analyze_feedback_from_db():
    try:
        analyzer = SentimentIntensityAnalyzer()
        async with aiosqlite.connect(DB_FILE) as db:
            async with db.execute("SELECT feedback FROM feedback") as cursor:
                async for row in cursor:
                    feedback = row[0]
                    sentiment_score = analyzer.polarity_scores(feedback)['compound']
                    logging.info(f"Feedback: {feedback}, Sentiment Score: {sentiment_score}")
    except Exception as e:
        logging.error(f"Feedback analysis exception: {e}")

@bot.event
async def on_ready():
    logging.info(f"Logged in as {bot.user.name} ({bot.user.id})")
    await init_db()
    bot.loop.create_task(process_db_queue())
    await analyze_feedback_from_db()  # Analyze feedback when the bot starts

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    try:
        # Her mesajı veritabanına kaydet
        user_id = str(message.author.id)
        user_name = message.author.name
        bot_id = str(bot.user.id)
        bot_name = bot.user.name
        content = message.content

        # Kaydetme işlemi için sıraya ekle
        await save_chat_history(user_id, content, user_name, bot_id, bot_name)

        # Eğer bot etiketlenmişse ya da adı geçiyorsa yanıt ver
        if bot.user.mentioned_in(message) or bot.user.name in message.content:
            user_personality = user_profiles[user_id]["preferences"].get("personality", None)

            message_counter.inc()
            start_time = time.time()

            relevant_history = await get_relevant_history(user_id, content)
            summarized_search = await duckduckgotool(content)
            response = await perform_advanced_reasoning(content, relevant_history, summarized_search, user_personality)

            end_time = time.time()
            response_time = end_time - start_time
            response_time_histogram.observe(response_time)
            response_time_summary.observe(response_time)

            # Handle long responses
            if len(response) > 2000:
                # Split the response into chunks of 2000 characters or less
                for i in range(0, len(response), 2000):
                    await message.channel.send(response[i:i + 2000])
            else:
                await message.channel.send(response)

            logging.info(f"Processed message from {user_name} in {response_time:.2f} seconds")

    except Exception as e:
        logging.error(f"An error occurred in on_message: {e}", exc_info=True)


@bot.event
async def on_message_edit(before, after):
    logging.info(f"Message edited: {before.content} -> {after.content}")

@bot.event
async def on_message_delete(message):
    logging.info(f"Message deleted: {message.content}")

@bot.event
async def on_member_join(member):
    logging.info(f"{member.name} has joined the server.")

@bot.event
async def on_member_remove(member):
    logging.info(f"{member.name} has left the server.")

@bot.event
async def on_error(event, *args, **kwargs):
    logging.error(f"An error occurred: {event}")

bot.run(discord_token)

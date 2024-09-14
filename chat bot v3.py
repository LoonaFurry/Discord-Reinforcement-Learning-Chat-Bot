import discord
from discord.ext import commands, tasks
import asyncio
import os
import logging
from dotenv import load_dotenv
import aiosqlite
import time
from collections import defaultdict
from prometheus_client import start_http_server, Counter, Histogram, Summary
from duckduckgo_search import AsyncDDGS
import google.generativeai as genai
from datetime import datetime, timezone


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
discord_token = ("discord-bot-token")
groq_api_key = ("groq-api-key")

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

# Prometheus metrics
start_http_server(8000)
message_counter = Counter('discord_bot_messages_total', 'Total messages processed')
error_counter = Counter('discord_bot_errors_total', 'Total errors')
response_time_histogram = Histogram('discord_bot_response_time_seconds', 'Response times')
response_time_summary = Summary('discord_bot_response_time_summary', 'Summary of response times')

# Context window and user profiles
CONTEXT_WINDOW_SIZE = 5000
user_profiles = defaultdict(lambda: {"preferences": {}, "history_summary": ""})

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
            await db.commit()
            logging.info("Database initialized.")
        else:
            logging.info("Database found, connecting...")

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

# Asynchronous function to ask Gemini AI
async def ask_gemini(prompt):
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        logging.error("Gemini AI exception: %s", e)
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

# Event handler when bot is ready
@bot.event
async def on_ready():
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

# Respond to mentions with history and AI responses
async def respond_to_mention(message, user_id):
    content = message.content
    mention = message.author.mention
    user_profile = user_profiles[user_id]

    # Get chat history for the current user
    relevant_history = await get_relevant_history(user_id, content)
    user_profile["history_summary"] = relevant_history

    # Perform a DuckDuckGo search based on the message content
    search_results = await duckduckgotool(content)

    # Combine search results and user input into a prompt for Gemini AI
    prompt = (
        f"You are a Furry Young Protogen who speaks Turkish. "
        f"Respond thoughtfully, integrating both knowledge from the web and past conversations. "
        f"Here is the relevant chat history:\n{relevant_history}\n"
        f"And here are some web search results:\n{search_results}\n"
        f"Now respond to the following message from {mention}: {content}"
    )

    try:
        start_time = time.time()

        # Get the final response from Gemini AI
        gemini_response = await ask_gemini(prompt)

        end_time = time.time()
        response_time = end_time - start_time
        response_time_histogram.observe(response_time)
        response_time_summary.observe(response_time)

        # Send the combined response to the user
        await message.channel.send(f"{mention} {gemini_response}")

    except Exception as e:
        error_counter.inc()
        logging.error(f"Error processing message: {e}")
        await message.channel.send(f"{mention} Bir hata oluştu. Lütfen daha sonra tekrar deneyin.")

# Task to change bot status
@tasks.loop(seconds=60)
async def change_status():
    await bot.change_presence(activity=discord.Game(name="LolbitFurry's Chat Bot"))

# Main entry point
async def main():
    await bot.start(discord_token)

# Run the bot
if __name__ == '__main__':
    asyncio.run(main())

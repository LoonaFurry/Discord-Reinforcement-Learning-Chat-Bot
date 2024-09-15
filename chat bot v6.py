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
import json
import random
#from transformers import pipeline 


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
CONTEXT_WINDOW_SIZE = 10000  # Increased context window size
user_profiles = defaultdict(lambda: {"preferences": {}, "history_summary": "", "context": [], "personality": None, "dialogue_state": "greeting", "user_name": None})
# Define possible dialogue states
DIALOGUE_STATES = ["greeting", "question_answering", "storytelling", "general_conversation", "farewell"]

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


# Infer user personality using Gemini AI
async def infer_personality(conversation_history):
    """Infers user personality using Gemini AI."""

    prompt = f"""
    Analyze the following conversation and infer the personality traits of the user. 
    Consider factors like their writing style, communication patterns, and expressed emotions.
    Provide a concise description of the personality, including traits like:
    - Extroverted/Introverted
    - Agreeable/Disagreeable
    - Conscientious/Impulsive
    - Neurotic/Stable
    - Open/Closed to Experiences

    Conversation:
    {conversation_history} 
    """

    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        personality_description = response.text
        return personality_description
    except Exception as e:
        logging.error(f"Gemini AI personality inference exception: {e}")
        return "I'm having trouble inferring the user's personality right now." 

# Summarize conversation history using Gemini AI
async def summarize_conversation(conversation_history):
    """Summarizes the conversation history using Gemini AI."""
    prompt = f"""
    Please provide a concise summary of the following conversation:

    Conversation:
    {conversation_history}
    """

    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        summary = response.text
        return summary
    except Exception as e:
        logging.error(f"Gemini AI summarization exception: {e}")
        return "I'm having trouble summarizing the conversation right now."


async def analyze_sentiment(text):
    prompt = f"""
    Analyze the sentiment of the following text:
    "{text}"

    Return the sentiment as one of the following labels:
    - POSITIVE (e.g., happy, joyful, excited)
    - NEGATIVE (e.g., sad, angry, disappointed)
    - NEUTRAL (e.g., factual, objective, neutral)

    Examples:
    "I'm so thrilled about the new update!" -> POSITIVE
    "This is incredibly frustrating." -> NEGATIVE
    "The weather forecast predicts rain tomorrow." -> NEUTRAL
    """

    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        sentiment_label = response.text.strip().upper() 

        # Validate the sentiment label 
        if sentiment_label in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
            return sentiment_label
        else:
            logging.warning(f"Gemini AI returned an invalid sentiment label: {sentiment_label}")
            return "NEUTRAL" # Or handle the invalid label as needed 

    except Exception as e:
        logging.error(f"Error analyzing sentiment with Gemini AI: {e}")
        return None 


# Prompt Engineering Functions 
def get_random_prompt_variation(user_id, query, personality, sentiment):
    """Generates a random prompt variation based on user, personality, and sentiment."""
    prompt_templates = [
        f"You are a friendly Furry Young Protogen who speaks Turkish and has a deep understanding of human emotions and social cues. You're also a great conversationalist who can be {personality} when the context requires it. {get_tone_instruction(sentiment)}. The user, {user_profiles[user_id]['user_name']}, just said: {query}. Now, respond accordingly.",
        f"Imagine you're a {personality} Furry Young Protogen fluent in Turkish and extremely perceptive of social cues. Based on the user's emotional state ({sentiment}), how would you respond to {query}?",
        f"As a helpful and empathetic Furry Young Protogen, how can you best respond to {query} while being {personality} and maintaining a {get_tone_instruction(sentiment)}? Consider {user_profiles[user_id]['user_name']}'s conversation history."
    ]
    return random.choice(prompt_templates)

def get_tone_instruction(sentiment):
    if sentiment == "POSITIVE":
        return "Maintain a cheerful and encouraging tone."
    elif sentiment == "NEGATIVE":
        return "Use an empathetic and comforting tone."
    elif sentiment == "NEUTRAL":
        return "Adopt a neutral and conversational tone."
    else:
        return ""


# Simulate advanced reasoning with Gemini
async def perform_advanced_reasoning(query, relevant_history, summarized_search, user_id):
    # Gemini Prompts for NLP Tasks
    json_example = r'[{"word": "John Doe", "entity": "PERSON"}, {"word": "New York", "entity": "LOCATION"}]'
    json_example_emotion = r'[{"emotion": "joy", "score": 0.8}, {"emotion": "sadness", "score": 0.2}]'
    ner_prompt = f"Extract all named entities (person, organization, location, etc.) from the following text: '{query}'. Return them in a JSON format like this: '{json_example}'."

    sentiment_prompt = f"Analyze the sentiment of the following text: '{query}'. Return the sentiment as a numerical score between 0 (negative) and 5 (positive)."

    emotion_prompt = f"Identify the primary emotions expressed in the following text: '{query}'. Return the emotions in a JSON format like this: '{json_example_emotion}'."


    # Update user context and infer personality dynamically
    if user_id:
        user_profiles[user_id]["context"].append({"query": query})
        if len(user_profiles[user_id]["context"]) > 5:  # Limit context window
            user_profiles[user_id]["context"].pop(0)


    # Personality Inference (Improved)
    if not user_profiles[user_id]["personality"]:  # Only infer if not already set
        try:
            personality_description = await infer_personality(relevant_history)
            user_profiles[user_id]["personality"] = personality_description
        except Exception as e:
            logging.error(f"Error in personality inference: {e}")


    # Sentiment Analysis 
    current_sentiment = await analyze_sentiment(query)
    if current_sentiment:
        logging.info(f"User sentiment: {current_sentiment}")

    # Update Dialogue State based on context
    if "question" in query.lower():
        user_profiles[user_id]["dialogue_state"] = "question_answering"
    elif "story" in query.lower():
        user_profiles[user_id]["dialogue_state"] = "storytelling"
    elif "goodbye" in query.lower() or "bye" in query.lower():
        user_profiles[user_id]["dialogue_state"] = "farewell"
    else:
        user_profiles[user_id]["dialogue_state"] = "general_conversation" 


    context_str = ""
    if user_id and user_profiles[user_id]["context"]:
        context_str = "Here's a summary of the recent conversation:\n"
        for turn in user_profiles[user_id]["context"]:
            context_str += f"User: {turn['query']}\n"


    # Prompt Engineering
    personality = user_profiles[user_id]["personality"] if user_profiles[user_id]["personality"] else "neutral"
    prompt = get_random_prompt_variation(user_id, query, personality, current_sentiment)

    # Add context information to the prompt
    prompt += f"\n\nRelevant Chat History:\n{relevant_history}\n\nSearch Results Summary:\n{summarized_search}\n\nConversation Summary:\n{user_profiles[user_id]['history_summary']}"

    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini AI reasoning exception: {e}")
        return "An error occurred while processing your request with Gemini AI. Please try again later."

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
        async with aiosqlite.connect(DB_FILE) as db:
            async with db.execute("SELECT feedback FROM feedback") as cursor:
                async for row in cursor:
                    feedback = row[0]
                    logging.info(f"Feedback: {feedback}")
    except Exception as e:
        logging.error(f"Feedback analysis exception: {e}")

@bot.event
async def on_ready():
    logging.info(f"Logged in as {bot.user.name} ({bot.user.id})")
    await init_db()
    bot.loop.create_task(process_db_queue())
    await analyze_feedback_from_db() 

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

        # Update user name in profile
        user_profiles[user_id]["user_name"] = user_name

        # Kaydetme işlemi için sıraya ekle
        await save_chat_history(user_id, content, user_name, bot_id, bot_name)

        # Eğer bot etiketlenmişse ya da adı geçiyorsa yanıt ver
        if bot.user.mentioned_in(message) or bot.user.name in message.content:

            message_counter.inc()
            start_time = time.time()

            relevant_history = await get_relevant_history(user_id, content)
            summarized_search = await duckduckgotool(content)

            # Update user profile with conversation summary
            user_profiles[user_id]["history_summary"] = await summarize_conversation(relevant_history)

            response = await perform_advanced_reasoning(content, relevant_history, summarized_search, user_id)

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
        await message.channel.send("I'm experiencing some technical difficulties. Please try again later.")


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

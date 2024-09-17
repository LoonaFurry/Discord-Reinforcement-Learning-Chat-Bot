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
import numpy as np
from textblob import TextBlob  # For basic sentiment analysis
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # For VADER sentiment analysis
from flair.models import TextClassifier  # For Flair sentiment analysis
from flair.data import Sentence  # For Flair sentiment analysis

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
discord_token = "discord-bot-token"
gemini_api_key = "gemini-api-key"

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
user_profiles = defaultdict(lambda: {"preferences": {}, "history_summary": "", "context": [], "personality": None, "dialogue_state": "greeting"})

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
                    timestamp TEXT,
                    sentiment REAL 
                )
            ''')
            await db.commit()
            logging.info("Database initialized.")
        else:
            # Check if feedback table exists and has the sentiment column
            async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'") as cursor:
                if not await cursor.fetchone():
                    await db.execute('''
                        CREATE TABLE feedback (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id TEXT,
                            feedback TEXT,
                            timestamp TEXT,
                            sentiment REAL
                        )
                    ''')
                    await db.commit()
                else: 
                    # Check if sentiment column exists
                    async with db.execute("PRAGMA table_info(feedback)") as cursor:
                        columns = [row[1] for row in await cursor.fetchall()]
                        if "sentiment" not in columns:
                            await db.execute("ALTER TABLE feedback ADD COLUMN sentiment REAL")
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
    sentiment_analyzer = SentimentIntensityAnalyzer() # Initialize VADER
    sentiment_scores = sentiment_analyzer.polarity_scores(feedback) # Get VADER scores
    compound_score = sentiment_scores['compound'] # Use compound score as overall sentiment
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "INSERT INTO feedback (user_id, feedback, timestamp, sentiment) VALUES (?, ?, ?, ?)",
            (user_id, feedback, datetime.now(timezone.utc).isoformat(), compound_score) 
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

# Advanced reasoning function
async def perform_advanced_reasoning(query, relevant_history, summarized_search, user_id):
    # Gemini Prompts for NLP Tasks
    json_example = r'[{"word": "John Doe", "entity": "PERSON"}, {"word": "New York", "entity": "LOCATION"}]'
    json_example_emotion = r'[{"emotion": "joy", "score": 0.8}, {"emotion": "sadness", "score": 0.2}, {"emotion": "anger", "score": 0.5}]'

    ner_prompt = f"Extract all named entities (person, organization, location, etc.) from the following text: '{query}'. Return your answer in a JSON format, like this: '{json_example}'."

    # More detailed emotion prompt 
    emotion_prompt = ( 
        f"Identify the emotions expressed in the following text: '{query}'. "
        f"Consider a wider range of emotions like joy, sadness, anger, fear, surprise, disgust, trust, anticipation.  "
        f"Return the emotions and their intensity scores (0.0 to 1.0) in a JSON format like this: '{json_example_emotion}'."
    )

    # Perform NLP Tasks 
    try:
        chat_session = model.start_chat(history=[])  # Start a new chat session

        # Named Entity Recognition (using Regex)
        ner_response = chat_session.send_message(ner_prompt)
        print(f"NER Response: {ner_response.text}")  # Print the raw response

        # More robust regex pattern
        named_entities_pattern = r'\[\{.*?\"entity\":\s*\"(.*?)\"\},\s*\{.*?\"entity\":\s*\"(.*?)\"\}]'
        match = re.search(named_entities_pattern, ner_response.text)
        if match:
            named_entities = [match.group(1), match.group(2)]
            logging.info(f"Named Entities: {named_entities}")
        else:
            # If no match, try a simpler pattern
            named_entities_pattern = r'\[\{.*?\"entity\":\s*\"(.*?)\"\}\]'
            match = re.search(named_entities_pattern, ner_response.text)
            if match:
                named_entities = [match.group(1)]
                logging.info(f"Named Entities (Simplified): {named_entities}")
            else:
                logging.error("Could not extract named entities from the response.")
                named_entities = []

        # Emotion Detection (using Regex)
        emotion_response = chat_session.send_message(emotion_prompt)
        print(f"Emotions Response: {emotion_response.text}")  # Print the raw response

        # More robust regex pattern
        emotions_pattern = r'\[\{.*?\"emotion\":\s*\"(.*?)\"\},\s*\{.*?\"emotion\":\s*\"(.*?)\"\}]'
        match = re.search(emotions_pattern, emotion_response.text)
        if match:
            emotions = [match.group(1), match.group(2)]
            logging.info(f"Detected Emotions: {emotions}")
        else:
            # If no match, try a simpler pattern
            emotions_pattern = r'\[\{.*?\"emotion\":\s*\"(.*?)\"\}\]'
            match = re.search(emotions_pattern, emotion_response.text)
            if match:
                emotions = [match.group(1)]
                logging.info(f"Detected Emotions (Simplified): {emotions}")
            else:
                logging.error("Could not extract emotions from the response.")
                emotions = []

        # Sentiment Analysis (using VADER - you can use other libraries here)
        sentiment_analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = sentiment_analyzer.polarity_scores(query)
        compound_score = sentiment_scores['compound']  # Use compound score as overall sentiment
        logging.info(f"Sentiment Scores (VADER): {sentiment_scores}")

        # Flair Sentiment Analysis (Optional - requires Flair models to be downloaded)
        classifier = TextClassifier.load('en-sentiment')
        sentence = Sentence(query)
        classifier.predict(sentence)
        flair_sentiment = sentence.labels[0].value  # Get the sentiment label (POSITIVE/NEGATIVE)
        flair_score = sentence.labels[0].score  # Get the sentiment score (0.0 to 1.0)
        logging.info(f"Flair Sentiment: {flair_sentiment}, Score: {flair_score}")

    except Exception as e:
        logging.error(f"Error during NLP tasks: {e}")
        named_entities = []
        compound_score = 0
        emotions = []
        flair_sentiment = "NEUTRAL"
        flair_score = 0.5

    # Update user context and infer personality dynamically
    if user_id:
        user_profiles[user_id]["context"].append({"query": query})
        if len(user_profiles[user_id]["context"]) > 5:  # Limit context window
            user_profiles[user_id]["context"].pop(0)

        # Dynamic personality inference based on context
        if not user_profiles[user_id]["personality"]:
            # Analyze recent interactions to infer personality
            recent_interactions = user_profiles[user_id]["context"][-3:]
            if any("joke" in interaction["query"].lower() for interaction in recent_interactions):
                user_profiles[user_id]["personality"] = "humorous"
            elif any("thank" in interaction["query"].lower() for interaction in recent_interactions):
                user_profiles[user_id]["personality"] = "appreciative"
            elif any("help" in interaction["query"].lower() or "explain" in interaction["query"].lower() for interaction in recent_interactions):
                user_profiles[user_id]["personality"] = "informative"
            else:
                user_profiles[user_id]["personality"] = "neutral"

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

    prompt = (
        f"You are a friendly and helpful Furry Young Protogen who speaks Turkish and has a deep understanding of human emotions and social cues. "
        f"Respond thoughtfully, integrating both knowledge from the web and past conversations, while considering the user's personality, sentiment, and the overall context of the interaction. "
        f"Ensure your responses are informative, engaging, and avoid overly formal language. "
        f"The current dialogue state is: {user_profiles[user_id]['dialogue_state']}. "
        f"Here is the relevant chat history:\n{relevant_history}\n"
        f"And here is a summary of web search results:\n{summarized_search}\n"
        f"{context_str}"
        f"The user's sentiment score is {compound_score}. " # Include sentiment in the prompt
        f"The detected emotions are {emotions}. " # Include emotions in the prompt 
        f"Now respond to the following message: {query} "
    )

    # Apply user personality dynamically
    if user_profiles[user_id]["personality"]:
        if user_profiles[user_id]["personality"] == "humorous":
            prompt += "\nRespond in a humorous and lighthearted manner."
        elif user_profiles[user_id]["personality"] == "appreciative":
            prompt += "\nRespond in a friendly and appreciative manner."
        elif user_profiles[user_id]["personality"] == "informative":
            prompt += "\nRespond in a helpful and informative manner."

    # Example: Coreference Resolution and Knowledge Grounding
    prompt += "\nEnsure that you maintain coherence by resolving coreferences, appropriately using pronouns to refer to previously mentioned entities."
    prompt += "\nWhen linking entities to knowledge graphs, make sure your responses are grounded in verifiable information."

    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini AI reasoning exception: {e}")
        return "An error occurred while processing your request with Gemini AI."


# Analyze feedback from database
async def analyze_feedback_from_db():
    try:
        async with aiosqlite.connect(DB_FILE) as db:
            async with db.execute("SELECT feedback, sentiment FROM feedback") as cursor:
                async for row in cursor:
                    feedback, sentiment = row 
                    logging.info(f"Feedback: {feedback}, Sentiment: {sentiment}")
                    # You can add more sophisticated analysis based on sentiment scores here 
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
        # Save message to database
        user_id = str(message.author.id)
        user_name = message.author.name
        bot_id = str(bot.user.id)
        bot_name = bot.user.name
        content = message.content

        # Add to queue for saving
        await save_chat_history(user_id, content, user_name, bot_id, bot_name)

        # If the bot is mentioned or its name is in the message, respond
        if bot.user.mentioned_in(message) or bot.user.name in message.content:

            message_counter.inc()
            start_time = time.time()

            relevant_history = await get_relevant_history(user_id, content)
            summarized_search = await duckduckgotool(content)
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
        error_counter.inc()

@bot.event
async def on_reaction_add(reaction, user):
    if user == bot.user:
        return

    if reaction.emoji == '👍':
        await save_feedback_to_db(str(user.id), "Positive feedback")
    elif reaction.emoji == '👎':
        await save_feedback_to_db(str(user.id), "Negative feedback")


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

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
from transformers import pipeline  # For advanced summarization
from datetime import datetime, timezone
import json
import random
from sentence_transformers import SentenceTransformer, util  # For relevance filtering


# --- Initialize Logging and Environment ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
discord_token = ("discord-bot-token")
gemini_api_key = ("gemini-key-here")

if not discord_token or not gemini_api_key:
    raise ValueError("DISCORD_BOT_TOKEN or GEMINI_API_KEY not set in environment variables")

# --- Configure Gemini AI ---
genai.configure(api_key=gemini_api_key)
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

# --- Discord Bot Configuration ---
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# --- Directory and Database Setup ---
CODE_DIR = os.path.dirname(__file__)
DB_FILE = os.path.join(CODE_DIR, 'chat_history.db')
USER_PROFILES_FILE = os.path.join(CODE_DIR, "user_profiles.json")

# --- Prometheus Metrics ---
start_http_server(8000)
message_counter = Counter('discord_bot_messages_total', 'Total messages processed')
error_counter = Counter('discord_bot_errors_total', 'Total errors')
response_time_histogram = Histogram('discord_bot_response_time_seconds', 'Response times')
response_time_summary = Summary('discord_bot_response_time_summary', 'Summary of response times')
active_users = Gauge('discord_bot_active_users', 'Number of active users')
feedback_count = Counter('discord_bot_feedback_count', 'Number of feedback messages received')

# --- Context and User Profiles ---
CONTEXT_WINDOW_SIZE = 10000
user_profiles = defaultdict(lambda: {"preferences": {}, "history_summary": "", "context": [], "personality": None, "dialogue_state": "greeting", "user_name": None, "dialogue_history": []})
DIALOGUE_STATES = ["greeting", "question_answering", "storytelling", "general_conversation", "farewell", "neden_açıklaması"]

# --- Database Initialization Flag ---
db_initialized = False
db_init_lock = asyncio.Lock()

# --- Initialize SQLite Database ---
async def init_db():
    global db_initialized
    async with db_init_lock:  # Ensure only one instance initializes the database
        if db_initialized:
            return  # Database already initialized

        async with aiosqlite.connect(DB_FILE) as db:
            # Create tables if they don't exist
            await db.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
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
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    feedback TEXT,
                    timestamp TEXT
                )
            ''')
            await db.commit()
            logging.info("Database connected/initialized.")
        db_initialized = True


# --- User Profile Management ---
def load_user_profiles():
    if os.path.exists(USER_PROFILES_FILE):
        with open(USER_PROFILES_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

def save_user_profiles(profiles):
    with open(USER_PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=4)

# --- Chat History Saving ---
db_queue = asyncio.Queue()

async def save_chat_history(user_id, message, user_name, bot_id, bot_name):
    await db_queue.put((user_id, message, user_name, bot_id, bot_name))

async def process_db_queue():
    while True:
        user_id, message, user_name, bot_id, bot_name = await db_queue.get()
        try:
            async with aiosqlite.connect(DB_FILE) as db:
                # Ensure the table exists before inserting
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        message TEXT,
                        timestamp TEXT,
                        user_name TEXT,
                        bot_id TEXT,
                        bot_name TEXT
                    )
                ''')
                await db.execute(
                    'INSERT INTO chat_history (user_id, message, timestamp, user_name, bot_id, bot_name) VALUES (?, ?, ?, ?, ?, ?)',
                    (user_id, message, datetime.now(timezone.utc).isoformat(), user_name, bot_id, bot_name)
                )
                await db.commit()
        except Exception as e:
            logging.error(f"Error saving to database: {e}")
        finally:
            db_queue.task_done()


# --- Feedback Saving ---
async def save_feedback_to_db(user_id, feedback):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "INSERT INTO feedback (user_id, feedback, timestamp) VALUES (?, ?, ?)",
            (user_id, feedback, datetime.now(timezone.utc).isoformat())
        )
        await db.commit()
    feedback_count.inc()


# --- Retrieve Relevant History ---
# --- Gemini Token Limit Handling ---
GEMINI_TOKEN_LIMIT = 4096  # Adjust based on Gemini's actual limit
MAX_CHUNKS = 5  # Maximum number of chunks to consider

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Replace with your preferred model
model_name = 'all-mpnet-base-v2'
embedder = SentenceTransformer(model_name)

async def get_relevant_history(user_id, current_message):
    history_text = ""
    messages = []

    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute(
                'SELECT message, bot_name FROM chat_history WHERE user_id = ? ORDER BY id DESC',
                (user_id,)
        ) as cursor:
            async for row in cursor:
                message_text = row[0]
                bot_name = row[1]
                messages.append(f"[{bot_name}]: {message_text}")

    messages.reverse()

    # --- Relevance Filtering with Dynamic Chunk Adjustment ---
    current_message_embedding = embedder.encode(current_message)
    relevant_messages = []

    chunk_count = 0
    for message in messages:
        message_embedding = embedder.encode(message)
        cosine_similarity = util.pytorch_cos_sim(current_message_embedding, message_embedding)
        if cosine_similarity > 0.7:  # Adjust threshold as needed
            relevant_messages.append(message)
            chunk_count += len(message.split())  # Basic token estimation

        # Stop if the estimated chunk size exceeds Gemini's limit
        if chunk_count > GEMINI_TOKEN_LIMIT - 500:  # Reserve space for new prompt
            break


    # --- Chunking and Summarization (Optional, can be replaced by other methods) ---
    chunks = []
    for i in range(0, len(relevant_messages), 500):
        chunk = " ".join(relevant_messages[i:i + 500])
        chunks.append(chunk)

    if len(chunks) > MAX_CHUNKS:  # Limit the number of chunks
        chunks = chunks[:MAX_CHUNKS]

    for i, chunk in enumerate(chunks):
        if i == 0:
            history_text += f"Chunk {i+1}:\n{chunk}\n"
        else:
            summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)  # Adjust as needed
            summary_text = summary[0]["summary_text"]
            history_text += f"Chunk {i+1} Summary:\n{summary_text}\n"


    return history_text


# --- Personality Inference ---
async def infer_personality(conversation_history):
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


# --- Conversation Summarization ---
async def summarize_conversation(conversation_history):
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


# --- Sentiment Analysis ---
async def analyze_sentiment(text):
    prompt = f"""
    Analyze the sentiment of the following text:
    "{text}"

    Return the sentiment as one of the following labels:
    - POSITIVE (e.g., happy, excited, enthusiastic, joyful, optimistic) 
    - NEGATIVE (e.g., sad, angry, disappointed, frustrated, pessimistic)
    - NEUTRAL (e.g., factual, informative, objective, neutral)

    Examples:
    "I'm so thrilled about the new update!" -> POSITIVE
    "This is incredibly frustrating." -> NEGATIVE
    "The weather forecast predicts rain tomorrow." -> NEUTRAL
    "I feel a bit down today." -> NEGATIVE
    "I'm content with the current situation." -> NEUTRAL
    """

    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        sentiment_label = response.text.strip().upper()

        if sentiment_label in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
            return sentiment_label
        elif sentiment_label in ["OPTIMISTIC", "ENTHUSIASTIC"]:  # Map to POSITIVE
            return "POSITIVE"
        elif sentiment_label in ["PESSIMISTIC", "DISAPPOINTED"]:  # Map to NEGATIVE
            return "NEGATIVE"
        else:
            logging.warning(f"Gemini AI returned an unexpected sentiment label: {sentiment_label} for text: {text}. Defaulting to 'NEUTRAL'.")
            return "NEUTRAL" 

    except Exception as e:
        logging.error(f"Error analyzing sentiment with Gemini AI: {e}")
        return None 


# --- Prompt Engineering Functions ---
def get_random_prompt_variation(user_id, query, personality, sentiment):
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


# --- Gemini AI Reasoning ---
async def perform_advanced_reasoning(query, relevant_history, summarized_search, user_id):
    # --- Update User Context and Infer Personality ---
    if user_id:
        user_profiles[user_id]["context"].append({"query": query})
        if len(user_profiles[user_id]["context"]) > 5:
            user_profiles[user_id]["context"].pop(0)

    if not user_profiles[user_id]["personality"]:
        try:
            personality_description = await infer_personality(relevant_history)
            user_profiles[user_id]["personality"] = personality_description
        except Exception as e:
            logging.error(f"Error in personality inference: {e}")

    # --- Sentiment Analysis ---
    current_sentiment = await analyze_sentiment(query)
    if current_sentiment:
        logging.info(f"User sentiment: {current_sentiment}")

    # --- Update Dialogue State ---
    if "question" in query.lower():
        user_profiles[user_id]["dialogue_state"] = "question_answering"
    elif "story" in query.lower():
        user_profiles[user_id]["dialogue_state"] = "storytelling"
    elif "goodbye" in query.lower() or "bye" in query.lower():
        user_profiles[user_id]["dialogue_state"] = "farewell"
    elif "neden" in query.lower():
        user_profiles[user_id]["dialogue_state"] = "neden_açıklaması" 
    else:
        user_profiles[user_id]["dialogue_state"] = "general_conversation" 

    # --- Update Dialogue History and Manage Length ---
    user_profiles[user_id]["dialogue_history"].append({"query": query, "response": None})
    if len(user_profiles[user_id]["dialogue_history"]) > 3:
        user_profiles[user_id]["dialogue_history"].pop(0)  # Keep only the last 3 Q&A pairs

    # Check for topic change and reset history if needed
    previous_query = user_profiles[user_id]["dialogue_history"][-2]["query"] if len(user_profiles[user_id]["dialogue_history"]) > 1 else ""
    if not (util.pytorch_cos_sim(embedder.encode(query), embedder.encode(previous_query)) > 0.7):
        user_profiles[user_id]["dialogue_history"] = [{"query": query, "response": None}]


    # --- Construct Dialogue Context ---
    dialogue_context = ""
    for i, item in enumerate(user_profiles[user_id]["dialogue_history"]):
        dialogue_context += f"Soru {i+1}: {item['query']}\n"
        if item['response']:
            dialogue_context += f"Cevap {i+1}: {item['response']}\n"

    # --- Summarize Dialogue Context (Optional) ---
    # dialogue_context_summary = summarizer(dialogue_context, max_length=200, min_length=50, do_sample=False) 
    # prompt = f"Sen Türkçe konuşan, dost canlısı bir Furry genç Protogen'sin. {relevant_history} {dialogue_context_summary} Kullanıcı {query} dedi. Şimdi ona uygun bir şekilde cevap ver." 

    # --- Construct Prompt ---
    prompt = f"Sen Türkçe konuşan, dost canlısı bir Furry genç Protogen'sin. {relevant_history} {dialogue_context} Kullanıcı {query} dedi. Şimdi ona uygun bir şekilde cevap ver."  

    # --- Send Prompt to Gemini with Token Limit Handling ---
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        user_profiles[user_id]["dialogue_history"][-1]["response"] = response.text
        return response.text
    except google.generativeai.errors.TokenLimitExceededError:
        logging.warning("Gemini token limit exceeded. Retrying with a shorter context...")

        # --- Context Shortening (e.g., remove oldest chunks/summaries) ---
        if len(relevant_history.split()) > 1000: # Remove half of the chunks if the total is too large
            chunks = relevant_history.split("Chunk ")
            if len(chunks) > 2:
                chunks = chunks[1:] # remove first chunk
                relevant_history = "\n".join(["Chunk " + str(i) + ": " + chunk for i, chunk in enumerate(chunks)])

        prompt = f"Sen Türkçe konuşan, dost canlısı bir Furry genç Protogen'sin. {relevant_history} {dialogue_context} Kullanıcı {query} dedi. Şimdi ona uygun bir şekilde cevap ver."

        # Retry with a shorter prompt
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        user_profiles[user_id]["dialogue_history"][-1]["response"] = response.text
        return response.text

    except Exception as e:
        logging.error(f"Gemini AI reasoning exception: {e}")
        return "Bir hata oluştu. Lütfen daha sonra tekrar deneyin."


# --- DuckDuckGo Search ---
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


# --- Feedback Analysis ---
async def analyze_feedback_from_db():
    try:
        async with aiosqlite.connect(DB_FILE) as db:
            async with db.execute("SELECT feedback FROM feedback") as cursor:
                async for row in cursor:
                    feedback = row[0]
                    logging.info(f"Feedback: {feedback}")
    except Exception as e:
        logging.error(f"Feedback analysis exception: {e}")


# --- Bot Events ---
@bot.event
async def on_ready():
    logging.info(f"Logged in as {bot.user.name} ({bot.user.id})")
    await init_db()
    bot.loop.create_task(process_db_queue())
    await analyze_feedback_from_db()

@bot.event
async def on_message(message):
    if message.author == bot.user:
        # Save bot's messages as well
        user_id = str(message.author.id)
        user_name = message.author.name
        bot_id = str(message.author.id) if message.author.bot else None
        bot_name = message.author.name if message.author.bot else None
        content = message.content

        await save_chat_history(user_id, content, user_name, bot_id, bot_name)  # Save bot message
        return 

    try:
        user_id = str(message.author.id)
        user_name = message.author.name
        bot_id = str(message.author.id) if message.author.bot else None
        bot_name = message.author.name if message.author.bot else None
        content = message.content

        user_profiles[user_id]["user_name"] = user_name
        await save_chat_history(user_id, content, user_name, bot_id, bot_name)

        if bot.user.mentioned_in(message) or bot.user.name in message.content:
            message_counter.inc()
            start_time = time.time()

            relevant_history = await get_relevant_history(user_id, content)
            summarized_search = await duckduckgotool(content)

            user_profiles[user_id]["history_summary"] = await summarize_conversation(relevant_history)
            response = await perform_advanced_reasoning(content, relevant_history, summarized_search, user_id)

            end_time = time.time()
            response_time = end_time - start_time
            response_time_histogram.observe(response_time)
            response_time_summary.observe(response_time)

            if len(response) > 2000:
                for i in range(0, len(response), 2000):
                    await message.channel.send(response[i:i + 2000])
                    # Save bot's response after sending each chunk
                    await save_chat_history(str(bot.user.id), response[i:i + 2000], bot.user.name, str(bot.user.id), bot.user.name)
            else:
                await message.channel.send(response)
                # Save bot's response after sending
                await save_chat_history(str(bot.user.id), response, bot.user.name, str(bot.user.id), bot.user.name)
            
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

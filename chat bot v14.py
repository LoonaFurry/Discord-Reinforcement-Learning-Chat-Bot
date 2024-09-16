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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline as transformers_pipeline
from spacy import load
import huggingface_hub
from groq import Groq  # Import Groq
from dotenv import load_dotenv
import os

# --- Initialize Logging and Environment ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

discord_token = ("discord-bot-token")
gemini_api_key = ("gemini-api-key")
# Get API key from environment variable
groq_api_key  = ("groq-api-key")

# --- Initialize Groq Client ---
client = Groq(api_key=groq_api_key)
# --- Configure Gemini AI ---
genai.configure(api_key=gemini_api_key)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
gemini_model = genai.GenerativeModel(
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
# CHAT_HISTORY_TXT_FILE = os.path.join(CODE_DIR, "your_corpus.txt")  # Path for the TXT file - Removed

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
user_profiles = defaultdict(lambda: {"preferences": {"favorite_topics": [], "conversation_style": "casual"},
                                     "history_summary": "", "context": [], "personality": None,
                                     "dialogue_state": "greeting", "user_name": None, "dialogue_history": [],
                                     "topic_history": [], "context_stack": ContextStack(),
                                     "current_topic": None, "current_sentiment": 0,
                                     "dialog_stack": DialogStack()})
DIALOGUE_STATES = ["greeting", "question_answering", "storytelling", "general_conversation", "farewell", "neden_açıklaması"]

# --- Database Initialization Flag ---
db_initialized = False
db_init_lock = asyncio.Lock()

# --- Status Messages ---
status_messages = [
    "Derin Düşüncelerle Yolculuktayım...",
    "Lolbitfurry Tarafından Yapıldım ^w^",
    "Sohbet Etmek İçin Hazırım!",
    "Yeni Bilgiler Öğreniyorum...",
    "Sanal Dünyayı Keşfediyorum..."
]

# --- Status Update Task ---
@tasks.loop(seconds=30)
async def update_status():
    await bot.change_presence(activity=discord.Game(random.choice(status_messages)))

# --- Context Stack ---
class ContextStack:
    def __init__(self, max_size=5):
        self.stack = []
        self.max_size = max_size
        self.embedder = SentenceTransformer('all-mpnet-base-v2')  # Using a pre-trained embedding model
        self.tfidf_vectorizer = TfidfVectorizer()

    def add_context(self, context):
        if len(self.stack) >= self.max_size:
            self.stack.pop(0)
        self.stack.append(context)
        self.update_tfidf()
        self.update_embeddings()

    def update_tfidf(self):
        if len(self.stack) > 0:
            texts = [c["query"] for c in self.stack]
            self.tfidf_vectorizer.fit(texts)

    def update_embeddings(self):
        if len(self.stack) > 0:
            self.embeddings = self.embedder.encode([c["query"] for c in self.stack])

    def get_relevant_contexts(self, query):
        if len(self.stack) == 0:
            return []
        query_embedding = self.embedder.encode([query])[0]  # Get embedding for the current query
        similarities = util.pytorch_cos_sim(query_embedding, self.embeddings)
        sorted_contexts = sorted(zip(self.stack, similarities), key=lambda x: x[1], reverse=True)
        return [c[0] for c in sorted_contexts]

class QuestionAnalyzer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()

    async def analyze(self, query):
        # 1. Question Type Classification
        question_type = await self.classify_question_groq(query)

        # 2. Entity Recognition (NER) - Assuming Groq can handle this or you might need to use another service if needed
        entities = await self.extract_entities_groq(query)

        # 3. Keyword Extraction (TF-IDF)
        keywords = self.extract_keywords_tfidf(query)

        # 4. Semantic Role Labeling (SRL) - Assuming Groq can handle this or you might need to use another service if needed
        srl_data = await self.extract_srl_groq(query)

        # 5. Dependency Parsing (Parse Tree) - Assuming Groq can handle this or you might need to use another service if needed
        dependency_tree = await self.extract_dependency_groq(query)

        return question_type, entities  # Return only question_type and entities

    async def classify_question_groq(self, question):
        prompt = f"""
        Classify the following question into one of the predefined categories:

        Question:
        {question}

        Categories:
        - General
        - Technical
        - Personal
        - Other
        """
        return await groq_ai_function(prompt)

    async def extract_entities_groq(self, text):
        prompt = f"""
        Extract entities from the following text and categorize them:

        Text:
        {text}
        """
        return await groq_ai_function(prompt)

    def extract_keywords_tfidf(self, query):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([query])  # You might want to pre-train on a larger corpus
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        return [feature_names[i] for i in tfidf_matrix.nonzero()[1]]  # Extract top keywords

    async def extract_srl_groq(self, text):
        prompt = f"""
        Analyze the semantic roles in the following text and provide a summary:

        Text:
        {text}
        """
        return await groq_ai_function(prompt)

    async def extract_dependency_groq(self, text):
        prompt = f"""
        Provide the dependency parsing details for the following text:

        Text:
        {text}
        """
        return await groq_ai_function(prompt)

# --- Groq AI Functions (Replace Gemini Functions) ---
async def groq_ai_function(prompt, model_name="llama-3.1-70b-versatile"):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    return response

async def infer_personality_groq(conversation_history):
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
    return await groq_ai_function(prompt)

async def summarize_conversation_groq(conversation_history):
    prompt = f"""
    Please provide a concise summary of the following conversation:

    Conversation:
    {conversation_history}
    """
    return await groq_ai_function(prompt)

async def analyze_sentiment_groq(text):
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
    response = await groq_ai_function(prompt)
    sentiment_label = response.strip().upper()

    if sentiment_label in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
        return sentiment_label
    elif sentiment_label in ["OPTIMISTIC", "ENTHUSIASTIC"]:  # Map to POSITIVE
        return "POSITIVE"
    elif sentiment_label in ["PESSIMISTIC", "DISAPPOINTED"]:  # Map to NEGATIVE
        return "NEGATIVE"
    else:
        logging.warning(f"Groq AI returned an unexpected sentiment label: {sentiment_label} for text: {text}. Defaulting to 'NEUTRAL'.")
        return "NEUTRAL"

class GeminiInteraction:
    def __init__(self):
        self.q_table = {}  # Initialize Q-table
        self.conversation_styles = {"formal": 0, "casual": 1, "friendly": 2}  # Example styles
        self.exploration_rate = 1.0  # Example starting exploration rate
        self.learning_rate = 0.1  # Example learning rate
        self.discount_factor = 0.9  # Example discount factor

    async def send_to_gemini(self, prompt, state):
        # Ensure the state exists in the Q-table
        if state not in self.q_table:
            self.q_table[state] = {style: 0 for style in self.conversation_styles.keys()}  # Initialize with default values

        # Choose conversation style based on reinforcement learning
        if random.random() < self.exploration_rate:
            action = random.choice(list(self.conversation_styles.keys()))  # Explore randomly
        else:
            if self.q_table[state]:  # Check if there are actions available
                action = max(self.q_table[state], key=self.q_table[state].get)  # Exploit best action
            else:
                action = self.default_action()  # Default action if no actions available

        # Modify prompt with selected style
        prompt = f"Respond to the following in a {action} style: {prompt}"

        try:
            chat_session = gemini_model.start_chat(history=[])
            response = chat_session.send_message(prompt)
            return response.text, action
        except Exception as e:
            logging.error(f"Error sending to Gemini: {e}")
            return "Bir hata oluştu. Lütfen daha sonra tekrar deneyin.", action

    def generate_prompt(self, query, context_stack, question_type, entities, current_sentiment, user_preferences, chat_history):
        # Construct prompt for Gemini based on context and question analysis, including chat history
        prompt = f"""
Sen Türkçe konuşan, dost canlısı bir Furry genç Protogen'sin. 
Kullanıcının önceki konuşmaları: {chat_history}.  # Include remembered chats
Kullanıcının son konuşmaları: {context_stack}.
Kullanıcının sorduğu soru: {query}.
Bu sorunun türü: {question_type}.
Soruda geçen önemli ifadeler: {entities}.
Kullanıcının hissi: {current_sentiment}.
Kullanıcının tercihleri: {user_preferences}.

Şimdi kullanıcıya uygun bir şekilde cevap ver.
"""
        return prompt

    def learn(self, state, action, reward, next_state):
        # Q-learning update rule
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state].values(), default=0)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state][action] = new_q

    def update_exploration_rate(self, episode):
        # Decay exploration rate over time
        self.exploration_rate = max(0.01, self.exploration_rate * 0.99)

    def default_action(self):
        # Define a default action to take when no actions are present
        return random.choice(list(self.conversation_styles.keys()))  # Return a random style

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
    
    # Removed - TXT file saving

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

# --- Summarizer using Groq AI ---
async def summarize_text_groq(text):
    prompt = f"""
    Provide a concise summary of the following text:

    Text:
    {text}
    """
    return await groq_ai_function(prompt)

async def get_embeddings_groq(text):
    prompt = f"""
    Generate embeddings for the following text. Provide a numerical vector representation:

    Text:
    {text}
    """
    response = await groq_ai_function(prompt)
    # Parse the response to extract embeddings
    embeddings = parse_embeddings_from_response(response)
    return embeddings

def parse_embeddings_from_response(response):
    # Burada yanıtı ayrıştırmak için uygun kodu ekleyin.
    import json
    try:
        data = json.loads(response)
        embeddings = data.get('embeddings', [])
        return embeddings
    except json.JSONDecodeError:
        logging.error("Failed to parse embeddings from response.")
        return []

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
    embedder = SentenceTransformer('all-mpnet-base-v2') # Initialize embedder here
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

# --- Dialog Management ---
class DialogStack:
    def __init__(self, max_size=10):
        self.stack = []
        self.max_size = max_size

    def push(self, dialog_state):
        if len(self.stack) >= self.max_size:
            self.stack.pop(0)
        self.stack.append(dialog_state)

    def peek(self):
        if self.stack:
            return self.stack[-1]
        else:
            return None

    def pop(self):
        if self.stack:
            return self.stack.pop()
        else:
            return None

    def clear(self):
        self.stack = []

# --- Bot Events ---
@bot.event
async def on_ready():
    logging.info(f"Logged in as {bot.user.name} ({bot.user.id})")
    await init_db()  # Ensure database is initialized before starting other tasks
    bot.loop.create_task(process_db_queue())
    await analyze_feedback_from_db()
    update_status.start()  # Start the task after the bot is ready

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

            # --- Load Past Chat History for Context ---
            past_messages = []
            async with aiosqlite.connect(DB_FILE) as db:
                async with db.execute(
                        'SELECT message FROM chat_history WHERE user_id = ? ORDER BY id DESC',
                        (user_id,)
                ) as cursor:
                    async for row in cursor:
                        past_messages.append(row[0])
            past_messages.reverse()

            # --- Dialog Management ---
            dialog_stack = user_profiles[user_id]["dialog_stack"]
            current_dialog_state = dialog_stack.peek()

            if current_dialog_state is None:
                # If the conversation is new, start with the greeting state
                current_dialog_state = "greeting"
                dialog_stack.push(current_dialog_state)
            else:
                # Check if the user's message is related to the current topic
                if user_profiles[user_id]["current_topic"] is not None and not user_profiles[user_id]["current_topic"] == 'None':
                    # If there is a current topic and the user's message is related to it
                    pass  # Do nothing, continue with the current dialog state
                else:
                    # If the user's message is about a new topic or not related to the current topic
                    # Transition to the "question_answering" state
                    current_dialog_state = "question_answering"
                    dialog_stack.push(current_dialog_state)

            # --- Topic Prediction ---
            # Removed - No need for topic prediction without corpus

            # --- Construct Context ---
            relevant_history = await get_relevant_history(user_id, content)
            summarized_search = await duckduckgotool(content)
            user_profiles[user_id]["history_summary"] = await summarize_conversation_groq(relevant_history)  # Use Groq summarization
            context_stack = user_profiles[user_id]["context_stack"]
            context_stack.add_context({"query": content, "response": None, "topic": None})  # Removed topic field
            relevant_contexts = context_stack.get_relevant_contexts(content) 

            # --- Analyze Question ---
            question_analyzer = QuestionAnalyzer()
            question_type, entities = await question_analyzer.analyze(content)

            # --- Sentiment Analysis ---
            user_profiles[user_id]["current_sentiment"] = await analyze_sentiment_groq(content)

            # --- Generate Prompt for Gemini ---
            # Create instance of GeminiInteraction here
            gemini_interaction = GeminiInteraction()
            prompt = gemini_interaction.generate_prompt(content, relevant_contexts,
                                                      question_type, entities,
                                                      user_profiles[user_id]["current_sentiment"],
                                                      user_profiles[user_id]["preferences"],
                                                      past_messages)

            # --- Call Gemini to Generate Response ---
            state = (current_dialog_state, user_profiles[user_id]["personality"], user_profiles[user_id]["current_sentiment"])
            response, chosen_style = await gemini_interaction.send_to_gemini(content, state)  # Pass content to send_to_gemini

            # --- Handle Response (Chunking if needed) ---
            if len(response) > 2000:
                for i in range(0, len(response), 2000):
                    await message.channel.send(response[i:i + 2000])
                    # Save bot's response after sending each chunk
                    await save_chat_history(str(bot.user.id), response[i:i + 2000], bot.user.name, str(bot.user.id), bot.user.name)
            else:
                await message.channel.send(response)
                # Save bot's response after sending
                await save_chat_history(str(bot.user.id), response, bot.user.name, str(bot.user.id), bot.user.name)

            # --- Reward Signal (Example) ---
            feedback = user_profiles[user_id].get("feedback", "neutral")  # Default to "neutral"
            reward = 1 if feedback == "positive" else -1 if feedback == "negative" else 0

            # --- Update RL Agent ---
            next_state = (current_dialog_state, user_profiles[user_id]["personality"], user_profiles[user_id]["current_sentiment"])
            gemini_interaction.learn(state, chosen_style, reward, next_state)

            # --- Update Exploration Rate ---
            gemini_interaction.update_exploration_rate(len(user_profiles[user_id].get("dialogue_history", [])))

            end_time = time.time()
            response_time = end_time - start_time
            response_time_histogram.observe(response_time)
            response_time_summary.observe(response_time)

            logging.info(f"Processed message from {user_name} in {response_time:.2f} seconds")

    except Exception as e:
        logging.error(f"An error occurred in on_message: {e}", exc_info=True)
        await message.channel.send("I'm experiencing some technical difficulties. Please try again later.")

async def collect_feedback():
    # Placeholder function to collect feedback
    # Implement this based on how you gather feedback from users
    return "positive"  # Example feedback, replace with actual feedback collection logic

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

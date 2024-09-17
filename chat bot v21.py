import discord
from discord.ext import commands, tasks
import asyncio
import os
import logging
from dotenv import load_dotenv
import aiosqlite
import time
from collections import defaultdict, deque
from prometheus_client import start_http_server, Counter, Histogram, Summary, Gauge
from duckduckgo_search import AsyncDDGS
import google.generativeai as genai
from datetime import datetime, timezone
import json
import numpy as np
import random
import nltk
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import transformers
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys
import io

# Force UTF-8 encoding for standard output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('vader_lexicon')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("error.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Create the bot instance with intents ---
intents = discord.Intents.all()
intents.message_content = True
intents.members = True

# Load environment variables
load_dotenv()
discord_token = ("discord-bot-token") 
gemini_api_key = ("gemini-api-key") 


if not discord_token or not gemini_api_key:
    raise ValueError("DISCORD_BOT_TOKEN or GEMINI_API_KEY not set in environment variables")

# Configure Gemini AI
genai.configure(api_key=gemini_api_key)

# Create the model
generation_config = {
    "temperature": 0.7,
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
CONTEXT_WINDOW_SIZE = 10000
user_profiles = defaultdict(lambda: {"preferences": {}, "history_summary": "", "context": deque(maxlen=10),
                                       "personality": None, "dialogue_state": "greeting", "long_term_memory": [],
                                       "last_bot_action": None})
DIALOGUE_STATES = ["greeting", "question_answering", "storytelling", "general_conversation", "farewell"]
BOT_ACTIONS = ["factual_response", "creative_response", "clarifying_question",
              "change_dialogue_state"]  # Define possible bot actions

# --- Initialize Sentiment Analyzer ---
sentiment_analyzer = SentimentIntensityAnalyzer()

# --- Initialize TF-IDF Vectorizer for Semantic Similarity ---
tfidf_vectorizer = TfidfVectorizer()

# --- Initialize a pre-trained language model for embeddings (e.g., Sentence-BERT) ---
embedding_model = transformers.AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = transformers.AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# --- Function to get Sentence Embeddings ---
def get_embeddings(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings

# Database Ready Flag and Lock
db_ready = False
db_lock = asyncio.Lock()

# Create chat history table 
async def create_chat_history_table():
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
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

# Initialize SQLite DB
async def init_db():
    global db_ready
    async with db_lock:
        await create_chat_history_table()  # Create the table if it doesn't exist
        db_ready = True

# Initialize user profiles
def load_user_profiles():
    if os.path.exists(USER_PROFILES_FILE):
        with open(USER_PROFILES_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

def save_user_profiles(profiles):
    with open(USER_PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=4)

# Save chat history to the database
db_queue = asyncio.Queue()

async def save_chat_history(user_id, message, user_name, bot_id, bot_name):
    async with db_lock:
        await db_queue.put((user_id, message, user_name, bot_id, bot_name))

async def process_db_queue():
    while True:
        # Wait for the database to be ready
        while not db_ready:
            await asyncio.sleep(1)

        user_id, message, user_name, bot_id, bot_name = await db_queue.get()
        try:
            async with db_lock:
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
    async with db_lock:
        async with aiosqlite.connect(DB_FILE) as db:
            await db.execute(
                "INSERT INTO feedback (user_id, feedback, timestamp) VALUES (?, ?, ?)",
                (user_id, feedback, datetime.now(timezone.utc).isoformat())
            )
            await db.commit()
    feedback_count.inc()

# Get relevant chat history for user (using TF-IDF for semantic relevance)
async def get_relevant_history(user_id, current_message):
    async with db_lock:
        history_text = ""
        messages = []
        async with aiosqlite.connect(DB_FILE) as db:
            async with db.execute(
                    'SELECT message FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?',
                    (user_id, 50)
            ) as cursor:
                async for row in cursor:
                    messages.append(row[0])

        messages.reverse()

        if not messages:
            return ""

        tfidf_matrix = tfidf_vectorizer.fit_transform(messages + [current_message])
        current_message_vector = tfidf_matrix[-1]

        similarities = cosine_similarity(current_message_vector, tfidf_matrix[:-1]).flatten()

        most_similar_indices = np.argsort(similarities)[-3:]

        for index in most_similar_indices:
            history_text += messages[index] + "\n"

        return history_text

# --- Function to Update Long-Term Memory ---
async def update_long_term_memory(user_id, message, embeddings):
    user_profiles[user_id]["long_term_memory"].append({"message": message, "embeddings": embeddings.tolist()})

# --- Function to Retrieve from Long-Term Memory ---
async def retrieve_from_long_term_memory(user_id, query_embeddings):
    most_relevant_memories = []

    for memory in user_profiles[user_id]["long_term_memory"]:
        similarity = cosine_similarity(query_embeddings, np.array(memory["embeddings"]))[0][0]
        if similarity > 0.7:
            most_relevant_memories.append(memory["message"])

    return most_relevant_memories

# --- DQN Implementation ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural network to approximate Q-values
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)) 
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1)) 
        return np.argmax(act_values[0]) 

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            # Prioritized Experience Replay (PER) 
            if len(self.memory) > 0:  
                td_errors = [abs(experience[2] + self.gamma * np.amax(self.model.predict(experience[3][:self.state_size].reshape(1, -1))[0]) - self.model.predict(experience[0][:self.state_size].reshape(1, -1))[0][experience[1]]) for experience in self.memory]
                probabilities = td_errors / np.sum(td_errors) 
                minibatch_indices = np.random.choice(len(self.memory), size=len(self.memory), replace=False, p=probabilities)
                minibatch = [self.memory[i] for i in minibatch_indices]
            else:
                return 
        else:
            minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state[:self.state_size].reshape(1, -1))[0]))
            target_f = self.model.predict(state[:self.state_size].reshape(1, -1))
            target_f[0][action] = target
            self.model.fit(state[:self.state_size].reshape(1, -1), target_f, epochs=1, verbose=0)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- Gemini AI Rate Limit Handling ---
RATE_LIMIT_PER_MINUTE = 60  # Adjust based on Gemini's API limits
RATE_LIMIT_WINDOW = 60 # Seconds
user_last_request_time = defaultdict(lambda: 0)

async def generate_response_with_rate_limit(prompt, user_id):
    current_time = time.time()
    time_since_last_request = current_time - user_last_request_time[user_id]

    if time_since_last_request < RATE_LIMIT_WINDOW / RATE_LIMIT_PER_MINUTE:
        await asyncio.sleep(RATE_LIMIT_WINDOW / RATE_LIMIT_PER_MINUTE - time_since_last_request)

    user_last_request_time[user_id] = time.time()

    response = model.generate_content(prompt) 
    return response.text


# --- Simulate advanced reasoning with Gemini ---
async def perform_advanced_reasoning(query, relevant_history, summarized_search, user_id):
    # --- Sentiment Analysis with NLTK VADER ---
    sentiment = sentiment_analyzer.polarity_scores(query)['compound']

    # --- Dynamically Infer User Personality ---
    if user_id:
        if not user_profiles[user_id]["personality"]:
            recent_sentiments = [sentiment_analyzer.polarity_scores(turn["query"])['compound'] for turn in user_profiles[user_id]["context"]]
            avg_sentiment = np.mean(recent_sentiments) if recent_sentiments else 0

            if avg_sentiment > 0.5:
                user_profiles[user_id]["personality"] = "positive"
            elif avg_sentiment < -0.5:
                user_profiles[user_id]["personality"] = "negative"
            else:
                user_profiles[user_id]["personality"] = "neutral"

        # --- Update Dialogue State based on Context and User Input ---
        if "question" in query.lower() or "?" in query:
            user_profiles[user_id]["dialogue_state"] = "question_answering"
        elif "story" in query.lower() or "tell me a story" in query.lower():
            user_profiles[user_id]["dialogue_state"] = "storytelling"
        elif "goodbye" in query.lower() or "bye" in query.lower():
            user_profiles[user_id]["dialogue_state"] = "farewell"
        else:
            user_profiles[user_id]["dialogue_state"] = "general_conversation"

    # --- Construct Context String ---
    context_str = ""
    if user_id and user_profiles[user_id]["context"]:
        context_str = "Here's a summary of the recent conversation:\n"
        for turn in user_profiles[user_id]["context"]:
            context_str += f"User: {turn['query']}\n"

    # --- Construct Prompt for Gemini ---
    prompt = (
        f"You are a friendly and helpful AI assistant who has a deep understanding of human emotions and social cues. "
        f"Respond thoughtfully, integrating both knowledge from the web and past conversations, while considering the user's personality, sentiment, and the overall context of the interaction. "
        f"Ensure your responses are informative, engaging, and avoid overly formal language. "
        f"The current dialogue state is: {user_profiles[user_id]['dialogue_state']}. "
        f"Here is the relevant chat history:\n{relevant_history}\n"
        f"And here is a summary of web search results:\n{summarized_search}\n"
        f"{context_str} "
        f"The user's query is: {query} "
    )

    # --- Add relevant long-term memories to the prompt ---
    query_embeddings = get_embeddings(query)
    retrieved_memories = await retrieve_from_long_term_memory(user_id, query_embeddings)
    if retrieved_memories:
        prompt += f"\n\nHere are some potentially relevant things the user has mentioned before: \n"
        for memory in retrieved_memories:
            prompt += f"- {memory}\n"

    # --- Apply User Personality Dynamically ---
    if user_profiles[user_id]["personality"]:
        if user_profiles[user_id]["personality"] == "humorous":
            prompt += "\nTry to respond in a humorous and lighthearted manner."
        elif user_profiles[user_id]["personality"] == "positive":
            prompt += "\nRespond in a positive and encouraging manner."
        elif user_profiles[user_id]["personality"] == "negative":
            prompt += "\nRespond in a careful and understanding manner."


    # --- DQN Integration ---
    # 1. State Representation 
    dialogue_state_onehot = [0] * len(DIALOGUE_STATES)
    dialogue_state_onehot[DIALOGUE_STATES.index(user_profiles[user_id]["dialogue_state"])] = 1

    user_engagement_score = calculate_user_engagement(user_id)
    topic_similarity_score = calculate_topic_similarity(query, user_id)
    last_bot_action_encoded = one_hot_encode_last_action(user_id)

    state = np.array([
        sentiment,
        len(query.split()),
        *dialogue_state_onehot,
        user_engagement_score,
        topic_similarity_score,
        *last_bot_action_encoded
    ])

    # 2. Choose an action using the DQN agent
    chosen_action = agent.act(state)
    user_profiles[user_id]["last_bot_action"] = BOT_ACTIONS[chosen_action] # Store the action taken

    # 3. Modify the prompt based on the chosen action 
    if chosen_action == 0:  # Factual response
        prompt += "\nProvide a concise and factual response to the user's query, focusing on accuracy and informativeness."
    elif chosen_action == 1:  # Creative response
        prompt += "\nRespond to the user's query in a creative and engaging way, potentially using humor, storytelling, or analogies. Prioritize entertainment and engagement."
    elif chosen_action == 2:  # Ask clarifying question
        prompt += "\nInstead of directly answering, ask a clarifying question to better understand the user's needs or the context of their query."
    elif chosen_action == 3:  # Change dialogue state (example)
        if user_profiles[user_id]["dialogue_state"] != "storytelling":
            prompt += "\nTry to gently guide the conversation towards storytelling mode. You could ask the user if they'd like to hear a story or start telling a story related to their query."

    # --- Generate Response with Gemini ---
    response_text = await generate_response_with_rate_limit(prompt, user_id)

    # --- DQN: Observe Next State and Reward ---
    next_state = calculate_next_state(state, response_text, user_id)
    reward = calculate_reward(response_text, user_id)

    # --- DQN: Remember and Replay ---
    done = False  # You might need to define when an episode is considered "done"
    agent.remember(state, chosen_action, reward, next_state, done)
    agent.replay(32)  # You can adjust the batch size
    agent.update_epsilon()

    return response_text, sentiment


# --- Helper Functions for DQN ---
def calculate_user_engagement(user_id):
    # Example: Calculate engagement based on message frequency and length
    engagement_score = 0
    if user_id in user_profiles:
        context = user_profiles[user_id]["context"]
        if context:
            message_count = len(context)
            avg_message_length = np.mean([len(turn["query"].split()) for turn in context])
            engagement_score = message_count * avg_message_length / 10  # Scale as needed
    return engagement_score


def calculate_topic_similarity(query, user_id):
    # Example: Calculate similarity between current query and previous topics
    similarity_score = 0
    if user_id in user_profiles and user_profiles[user_id]["long_term_memory"]:
        query_embeddings = get_embeddings(query)
        memory_embeddings = np.array([memory["embeddings"] for memory in user_profiles[user_id]["long_term_memory"]])
        
        # Reshape memory_embeddings to 2D:
        memory_embeddings = memory_embeddings.reshape(memory_embeddings.shape[0], -1) 

        similarities = cosine_similarity(query_embeddings, memory_embeddings).flatten()
        similarity_score = np.mean(similarities) if similarities.size else 0
    return similarity_score


def one_hot_encode_last_action(user_id):
    encoded_action = [0] * len(BOT_ACTIONS)
    if user_id in user_profiles and user_profiles[user_id]["last_bot_action"]:
        action_index = BOT_ACTIONS.index(user_profiles[user_id]["last_bot_action"])
        encoded_action[action_index] = 1
    return encoded_action


def calculate_next_state(state, response_text, user_id):
    # Example: Update sentiment and dialogue state in the next state
    next_state = state.copy()
    next_state[0] = sentiment_analyzer.polarity_scores(response_text)['compound']

    # Correctly update the one-hot encoded dialogue state
    next_state[2:2+len(DIALOGUE_STATES)] = [0] * len(DIALOGUE_STATES)  # Reset the one-hot encoding
    if "question" in response_text.lower() or "?" in response_text:
        next_state[2 + DIALOGUE_STATES.index("question_answering")] = 1 
    # ... (Add more logic here to update dialogue state using the same one-hot encoding approach) ...

    return next_state


def calculate_reward(response_text, user_id):
    # Example: Reward positive sentiment, longer conversations, and task completion
    reward = 0
    sentiment = sentiment_analyzer.polarity_scores(response_text)['compound']
    if sentiment > 0.5:
        reward += 1
    if len(user_profiles[user_id]["context"]) > 5:  # Encourage longer conversations
        reward += 0.5
    # ... (Add more reward logic based on task completion or user feedback) ...
    return reward

# Initialize DQN agent (with CORRECT state_size)
state_size = 2 + len(DIALOGUE_STATES) + len(BOT_ACTIONS)  # 2 + 5 + 4 = 11
# Uncomment the line below and retrain if you want to use 13 features
# state_size = 2 + len(DIALOGUE_STATES) + len(BOT_ACTIONS) + 2 # Add 2 for the extra features
action_size = len(BOT_ACTIONS) 
agent = DQNAgent(state_size, action_size)

# Asynchronous DuckDuckGo search tool
async def duckduckgotool(query) -> str:
    blob = ''
    try:
        ddg = AsyncDDGS()
        results = await ddg.text(query, max_results=100) 
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

    # Start the queue processing task only after the database is ready
    bot.loop.create_task(process_db_queue()) 
    
    await analyze_feedback_from_db()


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    try:
        # Wait until the database is ready
        while not db_ready:
            await asyncio.sleep(1) 

        # Save every message to the database 
        user_id = str(message.author.id)
        user_name = message.author.name
        bot_id = str(bot.user.id)
        bot_name = bot.user.name
        content = message.content

        # Add to the queue for database saving
        await save_chat_history(user_id, content, user_name, bot_id, bot_name)

        # If bot is mentioned or its name is in the message, respond
        if bot.user.mentioned_in(message) or bot.user.name in message.content:
            message_counter.inc()
            start_time = time.time()

            relevant_history = await get_relevant_history(user_id, content)
            summarized_search = await duckduckgotool(content)
            response, sentiment = await perform_advanced_reasoning( 
                content, relevant_history, summarized_search, user_id
            )

            end_time = time.time()
            response_time = end_time - start_time
            response_time_histogram.observe(response_time)
            response_time_summary.observe(response_time)

            # Handle long responses
            if len(response) > 2000:
                for i in range(0, len(response), 2000):
                    await message.channel.send(response[i:i + 2000])
            else:
                await message.channel.send(response)

            logging.info(f"Processed message from {user_name} in {response_time:.2f} seconds")

            # --- Update User Context ---
            user_profiles[user_id]["context"].append({"query": content, "response": response})

            # --- Update Long-Term Memory ---
            message_embeddings = get_embeddings(content)
            await update_long_term_memory(user_id, content, message_embeddings)

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

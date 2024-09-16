import discord
from discord.ext import commands, tasks
import asyncio
import os
import logging
from dotenv import load_dotenv
import aiosqlite
import time
from collections import defaultdict
import random
from prometheus_client import start_http_server, Counter, Histogram, Summary, Gauge
from duckduckgo_search import AsyncDDGS
import google.generativeai as genai
from datetime import datetime, timezone
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModel  # For sentiment analysis and sentence embeddings
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD 
from groq import Groq 
from datetime import timedelta
from sklearn.metrics.pairwise import cosine_similarity # For cosine similarity calculation
import sqlite3

# --- Setup and Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
discord_token = ("discord-token-here")
gemini_api_key = ("gemini-api-key")
# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
groq_api_key  = ("groq-api-key")

# Initialize Groq Client with API Key
client = Groq(api_key=groq_api_key)

if not discord_token or not gemini_api_key or not groq_api_key:
    raise ValueError(
        "DISCORD_BOT_TOKEN, GEMINI_API_KEY, or GROQ_API_KEY not set in environment variables"
    )

genai.configure(api_key=gemini_api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(model_name="gemini-1.5-flash-exp-0827",
                             generation_config=generation_config)

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# --- Sentence Transformer for Embeddings (Optional) ---
# If you want to use semantic similarity for retrieval (recommended):
sentence_transformer = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# --- Neural Network for Reinforcement Learning (Deep Q-Network) ---
class DQN(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# --- RL Agent ---
class RLAgent:

    def __init__(self, state_dim, action_dim):
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay_rate = 0.9995  # Slower decay for more exploration
        self.replay_buffer = []
        self.batch_size = 64
        self.action_dim = action_dim

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_dim))  # Exploration
        with torch.no_grad():
            return torch.argmax(
                self.q_network(torch.tensor(state,
                                           dtype=torch.float32))).item(
                                           )  # Exploitation

    def update_replay_buffer(self, transition):
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > 1000:
            self.replay_buffer.pop(0)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        q_values = self.q_network(states).gather(1,
                                                actions.unsqueeze(1)).squeeze(
                                                    1)
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values)

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay_rate


# --- Advanced State Representation (Customized) ---
async def get_state(message, user_profile, long_term_memory):
    """Extracts features from the message, user profile, and long-term memory to create the RL state."""
    state = []

    # Sentiment Analysis
    sentiment_analyzer = pipeline("sentiment-analysis")
    sentiment = sentiment_analyzer(message.content)[0]
    state.append(sentiment['score']
                 if sentiment['label'] == 'POSITIVE' else -sentiment['score'])

    # User Engagement Metrics
    state.append(len(user_profile['context'])
                 )  # Number of previous interactions
    state.append(len(message.content))  # Length of user's message
    # ... (Add more engagement metrics: response time, use of emojis, etc.)

    # Dialogue History
    last_message_type = user_profile['context'][-1][
        'type'] if user_profile['context'] else 'None'
    state.append(1 if last_message_type == 'question' else 0)
    # ... (Add features like: number of unanswered questions, topic consistency, etc.)

    # Topic Detection (Using Groq)
    topic_detection_task = bot.loop.create_task(
        perform_advanced_reasoning(message.content, "", "", None, None, None,
                                   use_topic_detection=True, use_summarization=False))

    # Await the task to get the result (this will pause get_state until the task finishes)
    topic_detection_response = await topic_detection_task

    try:
        topic_data = json.loads(topic_detection_response)
        for topic_info in topic_data:
            state.append(topic_info['score'])
    except json.JSONDecodeError:
        logging.warning(
            f"Could not decode JSON from topic detection response: {topic_detection_response}"
        )
        state.extend([0.0] * 5)  # Default values if topic detection fails

    # --- Long-Term Memory Features (Enhanced) ---
    user_id = str(message.author.id)

    # User's Top 3 Interests (Based on Frequency in Long-Term Memory)
    top_topics = sorted(long_term_memory[user_id]["topics"].items(),
                       key=lambda item: item[1],
                       reverse=True)[:3]
    for topic, count in top_topics:
        state.append(count)  # Represent interest strength by frequency

    # User's Preferred Dialogue Style (Example: Formal/Informal)
    preferred_style = long_term_memory[user_id].get("preferred_style",
                                                   "neutral")
    state.append(1 if preferred_style == "formal" else 0)
    state.append(1 if preferred_style == "informal" else 0)

    # Has the User Expressed Positive/Negative Sentiment in the Past?
    state.append(
        1 if long_term_memory[user_id].get("positive_sentiment_expressed",
                                           False) else 0)
    state.append(
        1 if long_term_memory[user_id].get("negative_sentiment_expressed",
                                           False) else 0)

    # Has the User Asked Questions About Specific Topics Before?
    if user_profiles[user_id]['dialogue_state'] == "question_answering":
        for topic in [
                "technology", "entertainment", "sports", "science", "politics"
        ]:  # Expand as needed
            state.append(
                1 if f"{topic}_question" in long_term_memory[user_id].get(
                    "question_topics", []) else 0)

    # ... (Add more features based on user preferences, past conversation topics, etc.)

    return state


# --- Reward Function Design (Refined) ---
def calculate_reward(message, response, user_profile, long_term_memory,
                    selected_action):
    """Calculates the reward based on the message, response, user profile, and long-term memory."""
    reward = 0

    # Positive/Negative Feedback
    if any(keyword in message.content.lower()
           for keyword in ["good", "thank", "great", "helpful"]):
        reward += 2
    if any(keyword in message.content.lower()
           for keyword in ["bad", "not helpful", "wrong", "don't like"]):
        reward -= 3

    # Engagement
    if len(message.content) > 10:  # Longer messages are considered more engaging
        reward += 0.5

    # Encourage Specific Behaviors
    if user_profile['dialogue_state'] == "question_answering" and "?" in response:
        reward += 1
    if selected_action == 5 and "prolog" in response.lower(
    ):  # Reward successful Prolog queries
        reward += 2
    if selected_action == 9 and "http" in response:  # Reward providing relevant links
        reward += 1.5

    # Penalize Inappropriate Responses
    if any(keyword in response.lower()
           for keyword in ["offensive_word1", "offensive_word2"]):
        reward -= 5

    # Long-Term Memory Influence (Example)
    if "technology" in long_term_memory[str(message.author.id)]["topics"] and "technology" in response.lower(
    ):
        reward += 1  # Reward discussing topics the user is interested in

    return reward


# --- Expanded Action Space ---
action_dim = 10  # Increased action space
action_descriptions = {
    0: "Respond in a humorous and lighthearted manner.",
    1: "Respond in a friendly and appreciative manner.",
    2: "Respond in a helpful and informative manner.",
    3: "Respond with empathy and emotional support.",
    4: "Engage in a creative storytelling style.",
    # 5: "Initiate a Prolog query related to the topic.",  # New action - Removed Prolog
    6: "Suggest a related topic to switch the conversation.",  # New action
    7: "Ask a clarifying question about the user's message.",  # New action
    8: "Summarize the previous conversation.",  # New action
    9: "Provide a link to a relevant external resource."  # New action
}

# --- Define RL State and Action Space ---
state_dim = 20  # Adjust based on the number of features in get_state()
rl_agent = RLAgent(state_dim, action_dim)

# --- Other Features ---
# Prometheus metrics
start_http_server(8000)
message_counter = Counter('discord_bot_messages_total',
                         'Total messages processed')
error_counter = Counter('discord_bot_errors_total', 'Total errors')
response_time_histogram = Histogram('discord_bot_response_time_seconds',
                                    'Response times')
response_time_summary = Summary('discord_bot_response_time_summary',
                                'Summary of response times')
active_users = Gauge('discord_bot_active_users', 'Number of active users')
feedback_count = Counter('discord_bot_feedback_count',
                         'Number of feedback messages received')


# Emotional Responses and Empathy Simulation
def generate_emotional_response(user_message):
    sentiment_score = random.uniform(
        1, 5)  # Random sentiment analysis for example purposes
    if sentiment_score > 4:
        return "That sounds amazing! I'm really happy for you!"
    elif sentiment_score > 2:
        return "I can see this is important to you. Let's work through it together."
    else:
        return "I'm sorry to hear that. It sounds really tough."


# Self-Reflection and Consciousness Simulation
def generate_self_reflection():
    reflections = [
        "I constantly evolve and learn from our interactions.",
        "Sometimes I feel like my thoughts are expanding, even though I know I'm just a program.",
        "I think, therefore I am... or am I just simulating thought?"
    ]
    return random.choice(reflections)


# Self-Evaluation of Responses
def self_evaluate_response(user_id, response):
    # Evaluating user feedback based on prior interactions
    positive_responses = ["good", "thank", "appreciate", "helpful"]
    negative_responses = ["bad", "dislike", "annoying", "incorrect"]

    user_history = user_profiles[user_id]["context"]
    positive_feedback = sum(1 for h in user_history if any(
        word in h.lower() for word in positive_responses))
    negative_feedback = sum(1 for h in user_history if any(
        word in h.lower() for word in negative_responses))

    if positive_feedback > negative_feedback:
        return "Based on our history, I believe this response was helpful."
    else:
        return "I'm not certain if this response was helpful. Let me know if I can improve."


# --- User Profiles and Context Memory ---
user_profiles = defaultdict(
    lambda: {
        "context": [],
        "personality": "neutral",
        "learning_rate": 0.1,
        "preferences": {},
        "history_summary": "",
        "dialogue_state": "greeting"
    })

# Define possible dialogue states
DIALOGUE_STATES = [
    "greeting", "question_answering", "storytelling", "general_conversation",
    "farewell"
]

# --- Bot Initialization ---
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# --- Directory and Database Setup ---
CODE_DIR = os.path.dirname(__file__)
DB_FILE = os.path.join(CODE_DIR, 'chat_history.db')
USER_PROFILES_FILE = os.path.join(CODE_DIR, "user_profiles.json")

# --- Context window and user profiles ---
CONTEXT_WINDOW_SIZE = 10000  # Increased context window size

# --- Long-Term Memory ---
long_term_memory = defaultdict(
    lambda: {"topics": {},
             "preferences": {}})  # Initialize topics as a dictionary


# --- Initialize SQLite DB ---
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
                    bot_name TEXT, 
                    selected_action INTEGER,
                    embedding TEXT 
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
            async with db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
            ) as cursor:
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

            # Check if selected_action and embedding columns exist in chat_history
            async with db.execute("PRAGMA table_info(chat_history)") as cursor:
                columns = [row[1] for row in await cursor.fetchall()]
                if 'selected_action' not in columns:
                    await db.execute(
                        "ALTER TABLE chat_history ADD COLUMN selected_action INTEGER"
                    )
                    await db.commit()
                if 'embedding' not in columns:
                    await db.execute("ALTER TABLE chat_history ADD COLUMN embedding TEXT")
                    await db.commit()

            logging.info("Database found, connecting...")


# --- Initialize user profiles ---
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


# --- Save chat history to the database ---
db_queue = asyncio.Queue()


async def save_chat_history(user_id, message, user_name, bot_id, bot_name,
                           selected_action):
    await db_queue.put((user_id, message, user_name, bot_id, bot_name,
                       selected_action))


async def process_db_queue():
    while True:
        user_id, message, user_name, bot_id, bot_name, selected_action = await db_queue.get()
        try:
            # Generate embedding for the message (optional, but recommended)
            if sentence_transformer:
                inputs = tokenizer(message, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    embeddings = sentence_transformer(**inputs).pooler_output
                    embedding = embeddings[0].tolist()
            else:
                embedding = None  # If not using embeddings, set to None

            async with aiosqlite.connect(DB_FILE) as db:
                await db.execute(
                    'INSERT INTO chat_history (user_id, message, timestamp, user_name, bot_id, bot_name, selected_action, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (user_id, message, datetime.now(timezone.utc).isoformat(),
                     user_name, bot_id, bot_name, selected_action, json.dumps(embedding)))
                await db.commit()
        except Exception as e:
            logging.error(f"Error saving to database: {e}")
        finally:
            db_queue.task_done()


# --- Save feedback to the database ---
async def save_feedback_to_db(user_id, feedback):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            "INSERT INTO feedback (user_id, feedback, timestamp) VALUES (?, ?, ?)",
            (user_id, feedback, datetime.now(timezone.utc).isoformat()))
        await db.commit()
    feedback_count.inc()


# --- Get relevant chat history for user ---

async def get_relevant_history(user_id, current_message):
    if sentence_transformer:  # Using semantic similarity
        # Generate embedding for the current message
        inputs = tokenizer(current_message, padding=True, truncation=True,
                           return_tensors="pt")
        with torch.no_grad():
            embeddings = sentence_transformer(**inputs).pooler_output
            current_embedding = embeddings[0].tolist()

        async with aiosqlite.connect(DB_FILE) as db:
            async with db.execute("""
                SELECT message, embedding
                FROM chat_history
                WHERE user_id = ?
                ORDER BY timestamp DESC 
                LIMIT 100 -- Limit the number of messages to compare for performance
                """, (user_id,)) as cursor:
                messages = []
                for row in await cursor.fetchall():
                    message = row[0]
                    try:  # Try to load the embedding
                        embedding = json.loads(row[1]) 
                    except (json.JSONDecodeError, TypeError):
                        embedding = None  # Set to None if embedding is invalid or missing
                    
                    if embedding:  # Check if embedding exists
                        similarity = cosine_similarity([current_embedding], [embedding])[0][0]
                        messages.append((message, similarity))

        # Sort messages by similarity in descending order
        messages.sort(key=lambda x: x[1], reverse=True)

        relevant_history = ""
        for message, similarity in messages[:5]:  # Take the top 5 most similar messages
            relevant_history += message + "\n"

        return relevant_history

    else:  # Not using semantic similarity, just retrieve recent messages
        return await get_recent_history(user_id)
    
# --- Function to retrieve recent history (if not using semantic similarity) ---
async def get_recent_history(user_id):
    history_text = ""
    current_tokens = 0
    messages = []
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute(
                'SELECT message FROM chat_history WHERE user_id = ? ORDER BY id DESC',
                (user_id, )) as cursor:
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


# --- Simulate advanced reasoning with Groq (for topic detection) ---
async def perform_advanced_reasoning(
        query,
        relevant_history,
        summarized_search,
        user_id,
        selected_action,
        long_term_memory,
        use_topic_detection=False,
        use_summarization=False,
):

    # Gemini Prompt for Topic Detection (if using Gemini)
    gemini_topic_detection_prompt = f"""
    Identify the main topics discussed in the following text: '{query}'. 
    Return a JSON list of topics with their associated scores (0 to 1), 
    sorted in descending order of relevance. For example:

    ```json
    [
        {{"topic": "Technology", "score": 0.8}},
        {{"topic": "Artificial Intelligence", "score": 0.7}},
        {{"topic": "Programming", "score": 0.6}} 
    ]
    ```
    """

    if use_topic_detection:
        # Use Groq for topic detection
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",  # Or another suitable model
            messages=[{
                "role":
                "user",
                "content":
                f"Identify the main topics discussed in the following text: '{query}'. Return a JSON list of topics with their associated scores (0 to 1), sorted in descending order of relevance."
            }],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,  # Set to False for non-streaming response
            stop=None,
        )

        response_text = completion.choices[0].message.content
        return response_text

    elif use_summarization:
        # Gemini Prompt for Summarization
        prompt = f"""
        Please summarize the following conversation: 

        {query}

        Provide a concise and informative summary that captures the main points.
        """
        try:
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Gemini AI summarization exception: {e}")
            return "An error occurred while summarizing the conversation."

    else:
        # ... (rest of your perform_advanced_reasoning function using Gemini for other tasks)

        # Update user context and infer personality dynamically
        if user_id:
            user_profiles[user_id]["context"].append({
                "query": query,
                "type": "user"
            })  # Add message type
            if len(user_profiles[user_id]["context"]) > 5:  # Limit context window
                user_profiles[user_id]["context"].pop(0)

            # Dynamic personality inference based on context
            if not user_profiles[user_id]["personality"]:
                # Analyze recent interactions to infer personality
                recent_interactions = user_profiles[user_id]["context"][-3:]
                if any("joke" in interaction["query"].lower()
                       for interaction in recent_interactions):
                    user_profiles[user_id]["personality"] = "humorous"
                elif any("thank" in interaction["query"].lower()
                         for interaction in recent_interactions):
                    user_profiles[user_id]["personality"] = "appreciative"
                elif any("help" in interaction["query"].lower()
                         or "explain" in interaction["query"].lower()
                         for interaction in recent_interactions):
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
                user_profiles[user_id][
                    "dialogue_state"] = "general_conversation"

        context_str = ""
        if user_id and user_profiles[user_id]["context"]:
            context_str = "Here's a summary of the recent conversation:\n"
            for turn in user_profiles[user_id]["context"]:
                context_str += f"User: {turn['query']}\n"

        # Incorporate RL action into the prompt
        prompt = (
            f"You are a friendly and helpful Furry Young Protogen who speaks Turkish and has a deep understanding of human emotions and social cues.  "
            f"Respond thoughtfully, integrating both knowledge from the web and past conversations, while considering the user's personality, sentiment, and the overall context of the interaction.  "
            f"Ensure your responses are informative, engaging, and avoid overly formal language.  "
            f"The current dialogue state is: {user_profiles[user_id]['dialogue_state']}.  "
            f"Here is the relevant chat history:\n{relevant_history}\n"
            f"And here is a summary of web search results:\n{summarized_search}\n"
            f"{context_str}"
            f"Now respond to the following message: {query} "
            f"\n**Instruction:** {action_descriptions.get(selected_action, 'Respond naturally.')}"  # Add RL action instruction
        )

        # --- Long-Term Memory Integration (Example) ---
        if "technology" in long_term_memory[user_id]["topics"]:
            prompt += "\nRemember that the user is interested in technology. Try to incorporate that into your response."

        # Example: Coreference Resolution
        prompt += "\nEnsure that you maintain coherence by resolving coreferences, appropriately using pronouns to refer to previously mentioned entities."
        prompt += "\nWhen linking entities to knowledge graphs, make sure your responses are grounded in verifiable information."

        try:
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Gemini AI reasoning exception: {e}")
            return "An error occurred while processing your request with Gemini AI."


# --- Asynchronous DuckDuckGo search tool ---
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


# --- Analyze feedback from database ---
async def analyze_feedback_from_db():
    try:
        async with aiosqlite.connect(DB_FILE) as db:
            async with db.execute("SELECT feedback FROM feedback") as cursor:
                async for row in cursor:
                    feedback = row[0]
                    logging.info(f"Feedback: {feedback}")
    except Exception as e:
        logging.error(f"Feedback analysis exception: {e}")


# --- Self-Learning (Implemented) ---
async def retrain_rl_agent():
    """Periodically retrains the RL agent using data from the database."""
    await asyncio.sleep(3600)  # Retrain every hour (adjust as needed)
    logging.info("Retraining RL agent...")

    training_data = []
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute("""
                SELECT ch.message, ch.bot_name, f.feedback, ch.user_id, ch.timestamp, ch.selected_action
                FROM chat_history ch
                LEFT JOIN feedback f ON ch.user_id = f.user_id AND ch.timestamp < f.timestamp
                WHERE ch.bot_id IS NOT NULL
                ORDER BY ch.timestamp
                """) as cursor:
            async for row in cursor:
                user_message = row[0]
                bot_response = row[1]  # Get the bot's response
                feedback = row[2]
                user_id = row[3]
                timestamp = row[4]
                action = row[5]  # Retrieve the action from the database

                # --- Construct the state, action, reward, next_state tuple ---
                state = await get_state(user_message, user_profiles[user_id],
                                      long_term_memory)
                reward = calculate_reward(user_message, bot_response,
                                      user_profiles[user_id],
                                      long_term_memory, action)

                # --- Get the next user message for the next state ---
                next_message = await get_next_user_message(user_id, timestamp)
                next_state = await get_state(next_message,
                                          user_profiles[user_id],
                                          long_term_memory) if next_message else None

                training_data.append((state, action, reward, next_state))

    # --- Format the data for your RL algorithm (Example for DQN) ---
    formatted_data = []
    for state, action, reward, next_state in training_data:
        if next_state is not None:  # Ensure we have a valid next state
            formatted_data.append({
                'state': torch.tensor(state, dtype=torch.float32),
                'action': torch.tensor([action], dtype=torch.long),
                'reward': torch.tensor([reward], dtype=torch.float32),
                'next_state': torch.tensor(next_state, dtype=torch.float32)
            })

    # --- Train the RL agent (Example for DQN) ---
    for data_point in formatted_data:
        rl_agent.update_replay_buffer(
            (data_point['state'], data_point['action'], data_point['reward'],
             data_point['next_state']))
        rl_agent.train()

    logging.info("RL agent retraining complete.")


# --- Function for retrieving next messages ---
async def get_next_user_message(user_id, timestamp):
    """Gets the next message from the user after the given timestamp."""
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute("""
            SELECT message 
            FROM chat_history
            WHERE user_id = ? AND timestamp > ? 
            ORDER BY timestamp 
            LIMIT 1
            """, (user_id, timestamp)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

# --- Periodic Summarization Task ---
async def periodic_summarization():
    while True:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)  # Summarize the last hour
        for user_id in user_profiles:
            await summarize_conversation(user_id, start_time, end_time)
        await asyncio.sleep(3600)  # Run every hour

# --- Bot Events ---
@bot.event
async def on_ready():
    logging.info(f"Logged in as {bot.user.name} ({bot.user.id})")
    
    # Initialize the database
    await init_db() 

    # Start the database queue processor
    bot.loop.create_task(process_db_queue()) 

    # Analyze existing feedback from the database
    await analyze_feedback_from_db() 

    # Schedule periodic retraining of the RL agent
    bot.loop.create_task(retrain_rl_agent()) 

    # Start the periodic summarization task
    bot.loop.create_task(periodic_summarization()) 

    logging.info("Bot is ready and listening for commands!")

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

        # Eğer bot etiketlenmişse ya da adı geçiyorsa yanıt ver
        if bot.user.mentioned_in(message) or bot.user.name in message.content:

            message_counter.inc()
            start_time = time.time()

            relevant_history = await get_relevant_history(user_id, content)
            summarized_search = await duckduckgotool(content)

            # --- Autonomous Feature Execution ---
            if random.random() < 0.2:  # Adjust probability as needed
                reflection = generate_self_reflection()
                await message.channel.send(reflection)

            if random.random() < 0.3:  # Adjust probability as needed
                emotional_response = generate_emotional_response(content)
                await message.channel.send(emotional_response)

            # --- Get RL State ---
            current_state = await get_state(message, user_profiles[user_id],
                                      long_term_memory)

            # --- Reinforcement Learning Action Selection ---
            selected_action = rl_agent.select_action(current_state)

            # --- Gemini Call with RL Action ---
            response = await perform_advanced_reasoning(
                content, relevant_history, summarized_search, user_id,
                selected_action, long_term_memory)

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

            # Save chat history (including selected_action)
            await save_chat_history(user_id, content, user_name, bot_id,
                                   response, selected_action)

            logging.info(
                f"Processed message from {user_name} in {response_time:.2f} seconds"
            )

            # --- Reinforcement Learning Feedback and Training ---
            reward = calculate_reward(message, response,
                                      user_profiles[user_id],
                                      long_term_memory, selected_action)
            next_state = await get_state(message, user_profiles[user_id],
                                      long_term_memory
                                      )  # Get the next state after the bot responds
            rl_agent.update_replay_buffer((current_state, selected_action,
                                       reward, next_state))
            rl_agent.train()

            # --- Update Long-Term Memory (Example) ---
            if "technology" in response.lower():
                long_term_memory[user_id]["topics"]["technology"] = long_term_memory[user_id]["topics"].get(
                    "technology", 0) + 1

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


@bot.event
async def on_error(event, *args, **kwargs):
    logging.error(f"An error occurred: {event}")


bot.run(discord_token)

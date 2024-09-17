import asyncio
import json
import logging
import os
import random
import time
import urllib.parse
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

import aiosqlite
import discord
import google.generativeai as genai
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from discord.ext import commands
from duckduckgo_search import AsyncDDGS
import groq
from prometheus_client import start_http_server, Counter, Histogram, Summary, Gauge
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModel

# --- Setup and Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

discord_token = ("discord-bot-token")  # Replace with your actual token
gemini_api_key = ("gemini-api-key")  # Replace with your actual API key
groq_api_key = ("groq-api-key")  # Replace with your actual API key
genai.configure(api_key=gemini_api_key)

# Ensure that the environment variables are set
if not discord_token or not gemini_api_key or not groq_api_key:
    raise ValueError(
        "DISCORD_BOT_TOKEN, GEMINI_API_KEY, or GROQ_API_KEY not set in environment variables"
    )

# Initialize Groq client
groq_client = groq.Client(api_key=groq_api_key)

last_message_time = {}

CODE_DIR = os.path.dirname(__file__)
DB_FILE = os.path.join(CODE_DIR, "chat_history.db")
USER_PROFILES_FILE = os.path.join(CODE_DIR, "user_profiles.json")

start_http_server(8000)
message_counter = Counter("discord_bot_messages_total", "Total messages processed")
error_counter = Counter("discord_bot_errors_total", "Total errors")
response_time_histogram = Histogram(
    "discord_bot_response_time_seconds", "Response times"
)
response_time_summary = Summary(
    "discord_bot_response_time_summary", "Summary of response times"
)
active_users = Gauge("discord_bot_active_users", "Number of active users")
feedback_count = Counter(
    "discord_bot_feedback_count", "Number of feedback messages received"
)

CONTEXT_WINDOW_SIZE = 10000
user_profiles = defaultdict(
    lambda: {
        "preferences": {},
        "history_summary": "",
        "context": [],
        "personality": None,
        "dialogue_state": "greeting",
    }
)
DIALOGUE_STATES = [
    "greeting",
    "question_answering",
    "storytelling",
    "general_conversation",
    "farewell",
]

# --- Functions to calculate response time and time lapse ---


def calculate_response_time(start_time):
    end_time = time.time()
    return end_time - start_time


def calculate_time_lapse(user_id):
    current_time = time.time()
    if user_id in last_message_time:
        time_lapse = current_time - last_message_time[user_id]
    else:
        time_lapse = None  # First time interaction with this user
    last_message_time[user_id] = current_time
    return time_lapse


# --- Safe message sending with rate limit handling ---


async def safe_send(channel, message_content):
    while True:
        try:
            await channel.send(message_content)
            break
        except discord.errors.HTTPException as e:
            if e.status == 429:  # Rate limited
                retry_after = e.retry_after
                logging.warning(f"Rate limited. Retrying after {retry_after} seconds.")
                await asyncio.sleep(retry_after)
            else:
                logging.error(f"Discord API error: {e}")
                return


# --- Generative AI setup ---
generation_config = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-exp-0827", generation_config=generation_config
)

# --- Sentence Transformer for Embeddings ---
sentence_transformer = AutoModel.from_pretrained(
    "sentence-transformers/all-mpnet-base-v2"
)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# --- Memory Components ---


class ShortTermMemory:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.messages = deque(maxlen=capacity)

    def add_message(self, message):
        self.messages.append(message)

    def get_messages(self):
        return list(self.messages)


class EpisodicMemory:
    def __init__(self, max_turns=5, inactivity_timeout=60):
        self.episodes = {}
        self.current_episode_id = None
        self.max_turns = max_turns
        self.inactivity_timeout = inactivity_timeout
        self.last_interaction_time = None

    def start_episode(self, episode_id):
        self.episodes[episode_id] = []
        self.current_episode_id = episode_id
        self.last_interaction_time = time.time()

    def add_message_to_episode(self, message):
        if self.current_episode_id:
            self.episodes[self.current_episode_id].append(message)
            self.last_interaction_time = time.time()

    def get_episode(self, episode_id):
        return self.episodes.get(episode_id, [])

    def should_end_episode(self):
        if not self.current_episode_id:
            return False
        current_time = time.time()
        if (
            len(self.episodes[self.current_episode_id]) >= self.max_turns
            or (current_time - self.last_interaction_time) > self.inactivity_timeout
        ):
            return True
        return False

    async def end_episode(self):
        if self.current_episode_id:
            await self.summarize_episode(self.current_episode_id)
            self.current_episode_id = None

    async def summarize_episode(self, episode_id):
        messages = self.get_episode(episode_id)
        if messages:
            summary = await summarize_conversation("".join(messages))
            self.episodes[episode_id] = [{"type": "summary", "content": summary}]


class LongTermMemory:
    def __init__(self):
        self.knowledge_base = {}

    def add_knowledge(self, key, value):
        self.knowledge_base[key] = value

    def get_knowledge(self, key):
        return self.knowledge_base.get(key)


# Initialize Memory Components
short_term_memory = ShortTermMemory()
episodic_memory = EpisodicMemory()
long_term_memory = LongTermMemory()

# --- Vector Database for Efficient Retrieval ---


class VectorDatabase:
    def __init__(self):
        self.embeddings = []
        self.messages = []

    def add_message(self, message, embedding):
        self.embeddings.append(embedding)
        self.messages.append(message)

    def search(self, query_embedding, top_k=5):
        similarities = cosine_similarity([query_embedding], self.embeddings)
        top_indices = np.argsort(similarities[0])[::-1][:top_k]
        return [self.messages[i] for i in top_indices]


# Initialize Vector Database
vector_db = VectorDatabase()

# --- Summarization and Advanced Reasoning ---


async def summarize_conversation(conversation):
    """Summarizes a conversation using Gemini."""
    prompt = f"Please summarize the following conversation:\n\n{conversation}"
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini AI summarization exception: {e}")
        return "An error occurred while summarizing the conversation."


async def analyze_user_intent(user_message):
    """Analyzes the user's intent using Groq."""
    groq_prompt = f"""
    You are an AI assistant designed to understand user intent. 
    Classify the intent of the following user message: '{user_message}' 
    Choose from the following intent categories: 
    - Greeting
    - Question
    - Request (for a joke, story, etc.) 
    - Information (seeking information about something)
    - Other 

    Provide your answer as a JSON object in the format: 
    {{"intent": "intent_category", "confidence": confidence_score}}
    """
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": groq_prompt}],
            temperature=0.5,
            max_tokens=256,
            top_p=1,
        )
        response_text = completion.choices[0].message.content
        intent_data = json.loads(response_text)
        return intent_data["intent"], intent_data["confidence"]
    except Exception as e:
        logging.error(f"Groq intent analysis error: {e}")
        return "Other", 0.0


# --- Sentiment Analysis and State Representation ---
sentiment_analysis_pipeline = pipeline("sentiment-analysis")


async def get_state(message, user_profile, long_term_memory):
    state = []
    state.append(len(message.content))
    state.append(1 if "?" in message.content else 0)

    try:
        sentiment_result = sentiment_analysis_pipeline(message.content)[0]
        sentiment_score = sentiment_result["score"]
        if sentiment_result["label"] == "POSITIVE":
            state.append(sentiment_score)
            state.append(0)
        else:
            state.append(0)
            state.append(sentiment_score)
    except Exception as e:
        logging.error(f"Sentiment analysis error: {e}")
        state.extend([0, 0])

    keywords = ["technology", "help", "joke", "story"]
    for keyword in keywords:
        state.append(1 if keyword in message.content.lower() else 0)

    state.append(user_profile.get("dialogue_state", 0))
    if "preferred_topics" in user_profile.get("preferences", {}):
        for topic in keywords:
            state.append(
                1 if topic in user_profile["preferences"]["preferred_topics"] else 0
            )
    else:
        state.extend([0] * len(keywords))

    user_data = long_term_memory.get_knowledge(str(message.author.id))
    if user_data:
        state.append(user_data.get("topics", {}).get("technology", 0))

    time_lapse = calculate_time_lapse(str(message.author.id))
    state.append(time_lapse or 0)

    intent, confidence = await analyze_user_intent(message.content)
    intent_mapping = {
        "Greeting": 0,
        "Question": 1,
        "Request": 2,
        "Information": 3,
        "Other": 4,
    }
    state.append(intent_mapping.get(intent, 4))
    state.append(confidence)

    return state


# --- RL Agent with DQN ---


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


class RLAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay_rate = 0.9995
        self.replay_buffer = []
        self.batch_size = 64
        self.action_dim = action_dim

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_dim))
        with torch.no_grad():
            return torch.argmax(self.q_network(state)).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_transition(self, transition):
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > 10000:
            self.replay_buffer.pop(0)


# Initialize RL Agent
state_dim = 20
action_dim = 5
rl_agent = RLAgent(state_dim, action_dim, learning_rate=0.002)

# --- Actions and Responses ---


async def generate_greeting():
    greetings = [
        "Hello! How can I assist you today?",
        "Hi there! What can I do for you?",
        "Greetings! How may I help you?",
    ]
    return random.choice(greetings)


async def answer_question(question):
    prompt = f"Please answer the following question:\n\n{question}"
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini AI answer generation error: {e}")
        return "I'm sorry, I'm having trouble answering that question right now."


async def tell_joke():
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "Parallel lines have so much in common. It's a shame they'll never meet.",
        "Why don't they play poker in the jungle? Too many cheetahs.",
    ]
    return random.choice(jokes)


async def discuss_technology(message):
    prompt = f"Let's discuss technology. I saw you mentioned something about it. Can you tell me more about what you're interested in regarding technology, based on this message: '{message}'?"
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini AI technology discussion error: {e}")
        return "I'm sorry, I'm having trouble discussing technology right now."


async def search_web(query):
    async with AsyncDDGS() as ddgs:
        try:
            results = await ddgs.search(query)
            if results:
                return f"Here's what I found on the web about '{query}': {results[0].url}"
            else:
                return f"I couldn't find any information about '{query}' on the web."
        except Exception as e:
            logging.error(f"Web search error: {e}")
            return "I'm sorry, I'm having trouble searching the web right now."


# --- Initialize SQLite DB ---
async def init_db():
    db_exists = os.path.exists(DB_FILE)
    async with aiosqlite.connect(DB_FILE) as db:
        if not db_exists:
            logging.info("Creating database...")
            await db.execute(
                """
                CREATE TABLE chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    message TEXT,
                    timestamp TEXT,
                    user_name TEXT,
                    bot_id TEXT,
                    bot_name TEXT
                )
            """
            )
            await db.execute(
                """
                CREATE TABLE feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    feedback TEXT,
                    timestamp TEXT
                )
            """
            )
            await db.commit()
            logging.info("Database initialized.")
        else:
            # Check if feedback table exists
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
            ) as cursor:
                if not await cursor.fetchone():
                    await db.execute(
                        """
                        CREATE TABLE feedback (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id TEXT,
                            feedback TEXT,
                            timestamp TEXT
                        )
                    """
                    )
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


async def save_chat_history(user_id, message, user_name, bot_id, bot_name):
    await db_queue.put((user_id, message, user_name, bot_id, bot_name))


async def process_db_queue():
    while True:
        user_id, message, user_name, bot_id, bot_name = await db_queue.get()
        try:
            async with aiosqlite.connect(DB_FILE) as db:
                await db.execute(
                    "INSERT INTO chat_history (user_id, message, timestamp, user_name, bot_id, bot_name) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        user_id,
                        message,
                        datetime.now(timezone.utc).isoformat(),
                        user_name,
                        bot_id,
                        bot_name,
                    ),
                )
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
            (user_id, feedback, datetime.now(timezone.utc).isoformat()),
        )
        await db.commit()
    feedback_count.inc()


# --- Get relevant chat history for user ---
async def get_relevant_history(user_id, current_message):
    history_text = ""
    current_tokens = 0
    messages = []
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute(
            "SELECT message FROM chat_history WHERE user_id = ? ORDER BY id DESC",
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


# --- Simulate advanced reasoning with Gemini ---


async def perform_advanced_reasoning(
    query, relevant_history, summarized_search, user_id
):
    # Gemini Prompts for NLP Tasks
    json_example = (
        r'[{"word": "John Doe", "entity": "PERSON"}, {"word": "New York", "entity": "LOCATION"}]'
    )
    json_example_emotion = (
        r'[{"emotion": "joy", "score": 0.8}, {"emotion": "sadness", "score": 0.2}]'
    )
    ner_prompt = f"Extract all named entities (person, organization, location, etc.) from the following text: '{query}'. Return them in a JSON format like this: '{json_example}'."

    sentiment_prompt = f"Analyze the sentiment of the following text: '{query}'. Return the sentiment as a numerical score between 0 (negative) and 5 (positive)."

    emotion_prompt = f"Identify the primary emotions expressed in the following text: '{query}'. Return the emotions in a JSON format like this: '{json_example_emotion}'."

    # Update user context and infer personality dynamically
    if user_id:
        user_profiles[user_id]["context"].append({"query": query})
        if len(user_profiles[user_id]["context"]) > 5:  # Limit context window
            user_profiles[user_id]["context"].pop(0)

        # Dynamic personality inference based on context
        if not user_profiles[user_id]["personality"]:
            # Analyze recent interactions to infer personality
            recent_interactions = user_profiles[user_id]["context"][-3:]
            if any(
                "joke" in interaction["query"].lower()
                for interaction in recent_interactions
            ):
                user_profiles[user_id]["personality"] = "humorous"
            elif any(
                "thank" in interaction["query"].lower()
                for interaction in recent_interactions
            ):
                user_profiles[user_id]["personality"] = "appreciative"
            elif any(
                "help" in interaction["query"].lower()
                or "explain" in interaction["query"].lower()
                for interaction in recent_interactions
            ):
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

    # --- Memory Network Integration ---
    attended_memories = ""
    if user_id:
        # 1. Tokenize the query and relevant history
        query_tokens = tokenizer.encode(query, add_special_tokens=True)
        history_tokens = tokenizer.encode(relevant_history, add_special_tokens=True)

        # 2. Pass through Memory Network
        output, attention_weights = memory_network(
            torch.tensor([query_tokens]), torch.tensor([history_tokens])
        )

        # 3. Get top attended tokens from history
        top_attended_indices = torch.topk(attention_weights, k=5)[1].squeeze()
        top_attended_tokens = tokenizer.convert_ids_to_tokens(
            history_tokens[top_attended_indices]
        )

        # 4. Construct a string of attended memories
        attended_memories = "Attended Memories:\n"
        for token in top_attended_tokens:
            attended_memories += f"- {token}\n"

    prompt = (
        f"You are a friendly and helpful Furry Young Protogen who speaks Turkish and has a deep understanding of human emotions and social cues.  "
        f"Respond thoughtfully, integrating both knowledge from the web and past conversations, while considering the user's personality, sentiment, and the overall context of the interaction.  "
        f"Ensure your responses are informative, engaging, and avoid overly formal language.  "
        f"The current dialogue state is: {user_profiles[user_id]['dialogue_state']}.  "
        f"Here is the relevant chat history:\n{relevant_history}\n"
        f"And here is a summary of web search results:\n{summarized_search}\n"
        f"{context_str}"
        f"{attended_memories}"  # Include attended memories
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

    # Example: Coreference Resolution
    prompt += (
        "\nEnsure that you maintain coherence by resolving coreferences, appropriately using pronouns to refer to previously mentioned entities."
    )
    prompt += (
        "\nWhen linking entities to knowledge graphs, make sure your responses are grounded in verifiable information."
    )
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini AI reasoning exception: {e}")
        return "An error occurred while processing your request with Gemini AI."


# --- Asynchronous DuckDuckGo search tool ---
async def duckduckgotool(query) -> str:
    blob = ""
    try:
        ddg = AsyncDDGS()
        results = ddg.text(query, max_results=100)
        for index, result in enumerate(results[:100]):
            blob += f"[{index}] Title: {result['title']}\nSnippet: {result['body']}\n\n"
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


# --- Memory Network and Attention Mechanism ---

# Define vocabulary size (replace with your actual vocabulary size)
vocab_size = len(tokenizer)  # Using tokenizer's vocabulary size


class MemoryNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(MemoryNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.I = nn.Linear(embedding_dim, hidden_dim)
        self.G = nn.Linear(hidden_dim, hidden_dim)
        self.O = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, query, memories):
        # 1. Embed query and memories
        query_embedding = self.embedding(query)
        memory_embeddings = self.embedding(memories)

        # 2. Calculate attention weights
        attention_weights = self.softmax(
            torch.matmul(query_embedding, memory_embeddings.T)
        )

        # 3. Weighted sum of memories
        weighted_memories = torch.matmul(attention_weights, memory_embeddings)

        # 4. Update hidden state
        hidden_state = self.G(self.I(query_embedding) + weighted_memories)

        # 5. Output
        output = self.O(hidden_state)
        return output, attention_weights


# Initialize Memory Network
embedding_dim = 128
hidden_dim = 256
memory_network = MemoryNetwork(embedding_dim, hidden_dim)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(memory_network.parameters())

# --- Training the Memory Network ---
# (You'll need to replace this with your actual training data and logic)
# Example training data format:
# training_data = [
#     ("what is the capital of France?", "Paris is the capital of France.", "Paris"),
#     # ... more examples
# ]

async def train_memory_network(training_data):
    for epoch in range(5):  # Example: Train for 5 epochs
        for query, memories, target in training_data:
            # 1. Tokenize query, memories, and target
            query_tokens = tokenizer.encode(query, add_special_tokens=True)
            memory_tokens = tokenizer.encode(memories, add_special_tokens=True)
            target_tokens = tokenizer.encode(target, add_special_tokens=True)

            # 2. Pass through Memory Network
            output, _ = memory_network(
                torch.tensor([query_tokens]), torch.tensor([memory_tokens])
            )

            # 3. Calculate loss
            loss = loss_fn(output, torch.tensor([target_tokens]))

            # 4. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# --- Bot Event Handling ---
@bot.event
async def on_ready():
    logging.info(f"Bot logged in as {bot.user}")
    await init_db()
    bot.loop.create_task(process_db_queue())
    await analyze_feedback_from_db()
    # Train the memory network
    await train_memory_network(training_data)
    
    # Load training data (replace with your actual training data)
    training_data = [
        # Add your training data here
    ]

    # Train the memory network
async def train_memory_network(training_data):
    for epoch in range(5):  # Example: Train for 5 epochs
        for query, memories, target in training_data:
            # 1. Tokenize query, memories, and target
            query_tokens = await tokenizer.encode(query, add_special_tokens=True) 
            memory_tokens = await tokenizer.encode(memories, add_special_tokens=True)
            target_tokens = await tokenizer.encode(target, add_special_tokens=True)

            # 2. Pass through Memory Network
            output, _ = memory_network(
                torch.tensor([query_tokens]), torch.tensor([memory_tokens])
            )

            # 3. Calculate loss
            loss = loss_fn(output, torch.tensor([target_tokens]))

            # 4. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

@bot.event
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    user_id = str(message.author.id)
    user_profile = long_term_memory.get_knowledge(user_id) or {
        "dialogue_state": 0,
        "preferences": {},
        "topics": defaultdict(int),
        "last_interaction": time.time(),
        "engagement_score": 0,
    }

    if not episodic_memory.current_episode_id:
        episodic_memory.start_episode(user_id + "_" + str(time.time()))
    episodic_memory.add_message_to_episode(message.content)

    try:
        # Save chat history to the database
        user_name = message.author.name
        bot_id = str(bot.user.id)
        bot_name = bot.user.name
        content = message.content

        await save_chat_history(user_id, content, user_name, bot_id, bot_name) 

        # Respond if the bot is mentioned or its name is in the message
        if bot.user.mentioned_in(message) or bot.user.name in message.content:
            message_counter.inc()
            start_time = time.time()

            state = await get_state(message, user_profile, long_term_memory)
            state_tensor = torch.tensor([state], dtype=torch.float32)

            selected_action = rl_agent.select_action(state_tensor)

            response = ""

            relevant_history = await get_relevant_history(user_id, content)
            summarized_search = await duckduckgotool(content)
            response = await perform_advanced_reasoning(
                content, relevant_history, summarized_search, user_id
            )

            # Add message and embedding to the vector database
            message_embedding = sentence_transformer.encode(
                content, convert_to_tensor=True
            )
            vector_db.add_message(content, message_embedding)

            # Handle long responses
            if len(response) > 2000:
                # Split the response into chunks of 2000 characters or less
                for i in range(0, len(response), 2000):
                    await message.channel.send(response[i : i + 2000])
            else:
                await message.channel.send(response)

            end_time = time.time()
            response_time = end_time - start_time
            response_time_histogram.observe(response_time)
            response_time_summary.observe(response_time)
            logging.info(
                f"Processed message from {user_name} in {response_time:.2f} seconds"
            )

            # --- Reward Calculation (Adjust as Needed) ---
            reward = 0

            if selected_action == 0:  # Greeting
                if user_profile.get("dialogue_state") == 0:
                    reward = 1
            elif selected_action == 1:  # Question Answering
                if "?" in message.content:
                    if len(response) > 20:
                        reward = 0.5
            # ... (Add reward logic for other actions) 

            if len(message.content) > 50:
                user_profile["engagement_score"] += 0.1
            try:
                sentiment_result = sentiment_analysis_pipeline(message.content)[0]
                if sentiment_result["label"] == "POSITIVE":
                    user_profile["engagement_score"] += sentiment_result["score"] * 0.2
            except Exception as e:
                logging.error(f"Sentiment analysis error: {e}")

            if user_profile["engagement_score"] > 2:
                reward += 0.5
                user_profile["engagement_score"] = 0

            response_time = calculate_response_time(user_profile["last_interaction"])
            if response_time > 5: 
                reward -= 0.1

            # --- Update User Profile --- 
            user_profile["dialogue_state"] = 1 
            if "technology" in message.content.lower():
                user_profile["topics"]["technology"] += 1
            user_profile["last_interaction"] = time.time()
            long_term_memory.add_knowledge(user_id, user_profile)

            if episodic_memory.should_end_episode():
                await episodic_memory.end_episode()

            next_state = await get_state(message, user_profile, long_term_memory)
            done = episodic_memory.should_end_episode()
            rl_agent.store_transition((state, selected_action, reward, next_state, done))

            rl_agent.train()
            rl_agent.update_epsilon()
            rl_agent.update_target_network()

            logging.info(
                f"Transition: State: {state}, Action: {selected_action}, Reward: {reward}, Next State: {next_state}, Done: {done}"
            )

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

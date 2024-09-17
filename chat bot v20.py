import asyncio
import json
import logging
import os
import random
import time
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Tokens and API keys
discord_token = ("discord-bot-token")  # Replace with your actual token
gemini_api_key = ("gemini-api-key")  # Replace with your actual API key
groq_api_key = "groq-api-key"  # Replace with your actual API key

if not discord_token or not gemini_api_key or not groq_api_key:
    raise ValueError("DISCORD_BOT_TOKEN, GEMINI_API_KEY, or GROQ_API_KEY not set in environment variables")

genai.configure(api_key=gemini_api_key)
groq_client = groq.Client(api_key=groq_api_key)

# Initialize Prometheus metrics
start_http_server(8000)
message_counter = Counter("discord_bot_messages_total", "Total messages processed")
error_counter = Counter("discord_bot_errors_total", "Total errors")
response_time_histogram = Histogram("discord_bot_response_time_seconds", "Response times")
response_time_summary = Summary("discord_bot_response_time_summary", "Summary of response times")
active_users = Gauge("discord_bot_active_users", "Number of active users")
feedback_count = Counter("discord_bot_feedback_count", "Number of feedback messages received")

CONTEXT_WINDOW_SIZE = 10000
user_profiles = defaultdict(lambda: {
    "preferences": {},
    "history_summary": "",
    "context": [],
    "personality": None,
    "dialogue_state": "greeting",
})
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
sentence_transformer = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
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
        if (len(self.episodes[self.current_episode_id]) >= self.max_turns or 
            (current_time - self.last_interaction_time) > self.inactivity_timeout):
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
    prompt = f"Please summarize the following conversation:\n\n{conversation}"
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini AI summarization exception: {e}")
        return "An error occurred while summarizing the conversation."

async def analyze_user_intent(user_message):
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

    state.append(user_profile.get("dialogue_state", "greeting"))
    return np.array(state)

# --- Reinforcement Learning Agent ---
class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.alpha = 0.001
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        model.compile(optimizer=optim.Adam(model.parameters(), lr=self.alpha), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.FloatTensor(state).unsqueeze(0))
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(torch.FloatTensor(next_state).unsqueeze(0))[0])
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0))
            target_f[0][action] = target
            self.model.fit(torch.FloatTensor(state).unsqueeze(0), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Initialize RL Agent
action_size = len(DIALOGUE_STATES)
state_size = 15  # Adjust based on the state representation
rl_agent = RLAgent(state_size, action_size)

# --- Main Bot Events ---
@bot.event
async def on_ready():
    logging.info(f"Logged in as {bot.user.name}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    user_id = str(message.author.id)
    user_profile = user_profiles[user_id]
    start_time = time.time()

    # Update user profile with the latest message
    user_profile["context"].append(message.content)
    if len(user_profile["context"]) > CONTEXT_WINDOW_SIZE:
        user_profile["context"].pop(0)

    # Check for episode ending
    if episodic_memory.should_end_episode():
        await episodic_memory.end_episode()

    # Get user intent and state
    intent, _ = await analyze_user_intent(message.content)
    state = await get_state(message, user_profile, long_term_memory)

    # Choose action based on RL agent
    action = rl_agent.act(state)
    user_profile["dialogue_state"] = DIALOGUE_STATES[action]

    # Generate response
    prompt = f"User message: {message.content}\nBot personality: {user_profile['personality']}\nBot response:"
    response = await generate_response(prompt)

    # Send response
    await safe_send(message.channel, response)
    message_counter.inc()

    # Log response time
    response_time = calculate_response_time(start_time)
    response_time_summary.observe(response_time)

    # Store message in memory
    short_term_memory.add_message(message.content)
    episodic_memory.add_message_to_episode(message.content)
    vector_db.add_message(message.content, state)

@bot.command(name='feedback')
async def feedback(ctx, *, feedback_message):
    feedback_count.inc()
    logging.info(f"Feedback received: {feedback_message}")
    await ctx.send("Thank you for your feedback!")

# --- Response Generation ---
async def generate_response(prompt):
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini AI generation error: {e}")
        return "Sorry, I couldn't generate a response."

# Run the bot
bot.run(discord_token)

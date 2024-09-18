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
import re
import aiohttp
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import io
import backoff
import requests 

# Force UTF-8 encoding for standard output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

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
user_profiles = defaultdict(lambda: {"preferences": {}, "history_summary": "",
                                      "context": deque(maxlen=10),
                                      "personality": {"humor": 0.5, "kindness": 0.8, "assertiveness": 0.6},
                                      "dialogue_state": "greeting", "long_term_memory": [],
                                      "last_bot_action": None, "interests": [],
                                      "query": "", "planning_state": {}})  # Added "query" and "planning_state"
DIALOGUE_STATES = ["greeting", "question_answering", "storytelling", "general_conversation",
                   "planning", "farewell"]  # Added "planning" state
BOT_ACTIONS = ["factual_response", "creative_response", "clarifying_question",
               "change_dialogue_state", "initiate_new_topic", "generate_plan"]

# --- Initialize Sentiment Analyzer ---
sentiment_analyzer = SentimentIntensityAnalyzer()

# --- Initialize TF-IDF Vectorizer for Semantic Similarity ---
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()


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
        await create_chat_history_table()
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


# --- Gemini AI Rate Limit Handling ---
RATE_LIMIT_PER_MINUTE = 60  # Adjust based on Gemini's API limits
RATE_LIMIT_WINDOW = 60  # Seconds
user_last_request_time = defaultdict(lambda: 0)



@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
async def generate_response_with_rate_limit(prompt, user_id):
    current_time = time.time()
    time_since_last_request = current_time - user_last_request_time[user_id]

    if time_since_last_request < RATE_LIMIT_WINDOW / RATE_LIMIT_PER_MINUTE:
        await asyncio.sleep(RATE_LIMIT_WINDOW / RATE_LIMIT_PER_MINUTE - time_since_last_request)

    user_last_request_time[user_id] = time.time()

    try:
        response = model.generate_content(prompt)
        logging.info(f"Raw Gemini response: {response}")
        return response.text
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response.status_code == 429:  # Rate limited (HTTP 429)
            logging.error(f"Gemini rate limit exceeded for user {user_id}: {e}")
            return "I'm being rate limited by Gemini. Please try again in a few moments."
        else:
            logging.error(f"Gemini API error for user {user_id}: {e}")
            return "I'm experiencing some technical difficulties with Gemini. Please try again later."
    except Exception as e:
        logging.exception(f"Error generating response with Gemini for user {user_id}: {e}")
        return "I'm sorry, I encountered an unexpected error while processing your request."


# --- Function to Extract Keywords/Topics from Text ---
def extract_keywords(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    keywords = [word for word, pos in tagged if pos.startswith('NN') or pos == 'JJ']  # Nouns and adjectives
    return keywords


# --- Function to Identify User Interests ---
async def identify_user_interests(user_id, message):
    keywords = extract_keywords(message)
    user_profiles[user_id]["interests"].extend(keywords)
    # You can add more sophisticated interest extraction logic here (e.g., using word embeddings, topic modeling)


# --- Function to Suggest a New Topic ---
async def suggest_new_topic(user_id):
    if user_profiles[user_id]["interests"]:
        topic = random.choice(user_profiles[user_id]["interests"])
        return f"Hey, maybe we could talk about {topic}? I'm curious to hear your thoughts."
    else:
        return "I'm not sure what to talk about next. What are you interested in?"


# --- Advanced Dialogue State Tracking (without Rasa NLU) ---
class DialogueStateTracker:
    def __init__(self):
        self.states = {
            "greeting": {
                "transitions": {
                    "general_conversation": ["hi", "hello", "hey", "how are you"],
                    "question_answering": ["what is", "who is", "where is", "how to"],
                    "planning": ["plan", "make a plan", "help me plan"],  # Transition to planning state
                    "farewell": ["bye", "goodbye", "see you"]
                },
                "default_transition": "general_conversation",
                "entry_action": self.greet_user
            },
            "general_conversation": {
                "transitions": {
                    "storytelling": ["tell me a story", "can you tell me a story"],
                    "question_answering": ["what is", "who is", "where is", "how to"],
                    "planning": ["plan", "make a plan", "help me plan"],  # Transition to planning state
                    "farewell": ["bye", "goodbye", "see you"]
                },
                "default_transition": "general_conversation"
            },
            "storytelling": {
                "transitions": {
                    "general_conversation": ["that's enough", "stop the story", "change topic"],
                    "question_answering": ["what happened", "who is that", "where did that happen"],
                    "planning": ["plan", "make a plan", "help me plan"],  # Transition to planning state
                    "farewell": ["bye", "goodbye", "see you"]
                },
                "default_transition": "storytelling"
            },
            "question_answering": {
                "transitions": {
                    "general_conversation": ["thanks", "got it", "okay"],
                    "storytelling": ["tell me a story", "can you tell me a story"],
                    "planning": ["plan", "make a plan", "help me plan"],  # Transition to planning state
                    "farewell": ["bye", "goodbye", "see you"]
                },
                "default_transition": "general_conversation"
            },
            "planning": {  # New state for planning
                "transitions": {
                    "general_conversation": ["done", "finished", "that's it"],
                    "question_answering": ["what about", "how about"],
                    "farewell": ["bye", "goodbye", "see you"]
                },
                "default_transition": "planning",
                "entry_action": self.start_planning  # You might have a specific entry action
            },
            "farewell": {
                "transitions": {},
                "default_transition": "farewell"
            }
        }

    def greet_user(self, user_id):
        return f"Hello <@{user_id}>! How can I help you today?"

    def start_planning(self, user_id):
        return "Okay, let's start planning. What are you trying to plan?"

    async def classify_dialogue_act(self, user_input):
        prompt = (
            f"Classify the following user input into one of these dialogue acts: "
            f"greeting, question_answering, storytelling, general_conversation, planning, farewell.\n\n"
            f"User input: {user_input}\n\n"
            f"Provide the dialogue act classification as a single word on the first line of the response:"
        )

        logging.info(f"Dialogue Act Classification Prompt: {prompt}")  # Log the prompt

        response = await generate_response_with_rate_limit(prompt, None)  # No user_id needed here

        # Extract the dialogue act from Gemini's response
        try:
            dialogue_act = response.strip().split("\n")[0].lower()  # Get the first line and lowercase it

            # Additional logging for debugging
            logging.info(f"Raw Gemini response for Dialogue Act Classification: {response}")
            logging.info(f"Extracted Dialogue Act: {dialogue_act}")

        except Exception as e:
            logging.error(f"Error extracting dialogue act from Gemini response: {e}")
            dialogue_act = None  # Handle the case where response is not in the expected format

        return dialogue_act

    async def transition_state(self, current_state, user_input, user_id, conversation_history):
        dialogue_act = await self.classify_dialogue_act(user_input)  # Use classify_dialogue_act

        if dialogue_act:
            return dialogue_act

        transitions = self.states[current_state]["transitions"]

        # Check for contextual transitions (example based on user profile)
        if current_state == "general_conversation" and "storytelling" in user_profiles[user_id]["interests"]:
            if any(keyword in user_input.lower() for keyword in transitions["storytelling"]):
                return "storytelling"

        # Check for keyword-based transitions
        for next_state, keywords in transitions.items():
            if any(keyword in user_input.lower() for keyword in keywords):  # Fixed: Use keywords instead of transitions
                return next_state

        return self.states[current_state]["default_transition"]


# Initialize the dialogue state tracker
dialogue_state_tracker = DialogueStateTracker()


async def gemini_search_and_summarize(query) -> str:
    """Performs a DuckDuckGo search, provides the results to Gemini,
    and asks Gemini to summarize them.
    """
    try:
        ddg = AsyncDDGS()
        search_results = await asyncio.to_thread(ddg.text, query,
                                                max_results=100)  # Reduced to 100 for faster processing

        search_results_text = ""
        for index, result in enumerate(search_results):
            search_results_text += f'[{index}] Title: {result["title"]}\nSnippet: {result["body"]}\n\n'

        prompt = (
            f"You are a helpful AI assistant. A user asked about '{query}'. "
            f"Here are some relevant web search results:\n\n"
            f"{search_results_text}\n\n"
            f"Please provide a concise and informative summary of these search results."
        )

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        logging.error(f"Gemini search and summarization error: {e}")
        return "I'm sorry, I encountered an error while searching and summarizing information for you."


async def extract_url_from_description(description):
    """Extracts the URL from the description using web scraping (DuckDuckGo)."""

    search_query = f"{description} site:youtube.com OR site:twitch.tv OR site:instagram.com OR site:twitter.com"  # Customize the sites as needed

    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://duckduckgo.com/html/?q={search_query}") as response:
            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")

            # Find the first result link
            first_result = soup.find("a", class_="result__a")
            if first_result:
                return first_result["href"]
            else:
                return None


async def fix_link_format(text):
    """Fixes link format and removes duplicate links."""

    new_text = ""
    lines = text.split("\n")
    for line in lines:
        match = re.match(r"\[(.*?)\]\(link\)", line)  # Match lines with "[link text](link)"
        if match:
            link_text = match.group(1)  # Extract the link text
            # Extract the URL from the description using web scraping
            url = await extract_url_from_description(link_text)
            if url:
                new_text += f"[{link_text}]({url})\n"  # Use Markdown format
            else:
                new_text += line + "\n"  # Keep the line as is if URL extraction fails
        else:
            new_text += line + "\n"  # Keep the line as is if it's not a link

    return new_text


# --- Complex Dialogue Manager ---
async def complex_dialogue_manager(user_profiles, user_id, message):
    if user_profiles[user_id]["dialogue_state"] == "planning":
        if "stage" not in user_profiles[user_id]["planning_state"]:
            user_profiles[user_id]["planning_state"]["stage"] = "initial_request"

        if user_profiles[user_id]["planning_state"]["stage"] == "initial_request":
            # Step 1: Extract goals and preferences
            goal, query_type = await extract_goal(user_profiles[user_id]["query"])
            user_profiles[user_id]["planning_state"]["goal"] = goal
            user_profiles[user_id]["planning_state"]["query_type"] = query_type

            # Move to the next stage
            user_profiles[user_id]["planning_state"]["stage"] = "gathering_information"

            # Ask clarifying questions
            return await ask_clarifying_questions(goal, query_type)

        elif user_profiles[user_id]["planning_state"]["stage"] == "gathering_information":
            # Process user responses to clarifying questions
            await process_planning_information(user_id, message)

            # Check if we have enough information to generate a plan
            if await has_enough_planning_information(user_id):
                user_profiles[user_id]["planning_state"]["stage"] = "generating_plan"

                # Generate a complex plan with multiple steps
                plan = await generate_plan(
                    user_profiles[user_id]["planning_state"]["goal"],
                    user_profiles[user_id]["planning_state"]["preferences"],
                    user_id
                )

                # Validate the plan (add your validation logic here)
                if await validate_plan(plan, user_id):
                    # Store the plan
                    user_profiles[user_id]["planning_state"]["plan"] = plan

                    # Move to the next stage
                    user_profiles[user_id]["planning_state"]["stage"] = "presenting_plan"

                    # Present the plan and ask for feedback
                    return await present_plan_and_ask_for_feedback(plan)
                else:
                    # If the plan is not valid, go back to gathering information
                    user_profiles[user_id]["planning_state"]["stage"] = "gathering_information"
                    return "The plan is not feasible. Please provide more information or adjust your preferences."
            else:
                return await ask_further_clarifying_questions(user_id)

        elif user_profiles[user_id]["planning_state"]["stage"] == "presenting_plan":
            # Process user feedback on the plan
            await process_plan_feedback(user_id, message)

            # Move to the next stage (evaluating or executing, depending on the feedback)
            if "accept" in message.lower():  # Check if the user accepted the plan
                user_profiles[user_id]["planning_state"]["stage"] = "evaluating_plan"
            else:  # If the user wants to make changes
                user_profiles[user_id]["planning_state"]["stage"] = "gathering_information"
                return "Okay, let's revise the plan. What changes would you like to make?"

        elif user_profiles[user_id]["planning_state"]["stage"] == "evaluating_plan":
            # Step 3: Evaluate the generated plan
            evaluation = await evaluate_plan(user_profiles[user_id]["planning_state"]["plan"], user_id)
            user_profiles[user_id]["planning_state"]["evaluation"] = evaluation

            # Move to the next stage
            user_profiles[user_id]["planning_state"]["stage"] = "executing_plan"

            # Present results with detailed reasoning and next steps
            return generate_response(
                user_profiles[user_id]["planning_state"]["plan"],
                evaluation,
                {},  # execution_results will be added later
                user_profiles[user_id]["planning_state"]["preferences"]
            )

        elif user_profiles[user_id]["planning_state"]["stage"] == "executing_plan":
            # Step 4: Execute the plan
            execution_results = await execute_plan(user_profiles[user_id]["planning_state"]["plan"], user_id,
                                                  user_profiles[user_id]["planning_state"]["preferences"])

            # Present the execution results
            return generate_response(
                user_profiles[user_id]["planning_state"]["plan"],
                user_profiles[user_id]["planning_state"]["evaluation"],
                execution_results,
                user_profiles[user_id]["planning_state"]["preferences"]
            )


async def ask_clarifying_questions(goal, query_type):
    # More generalized clarifying questions for any type of plan
    return "To create an effective plan, I need some more details. Could you tell me:\n" \
           f"- What is the desired outcome of this plan?\n" \
           f"- What are the key steps or milestones involved?\n" \
           f"- Are there any constraints or limitations I should be aware of?\n" \
           f"- What resources or tools are available to you?\n" \
           f"- What is the timeframe for completing this plan?"


async def process_planning_information(user_id, message):
    # Extract relevant planning information from the user's message
    user_profiles[user_id]["planning_state"]["preferences"]["user_input"] = message


async def has_enough_planning_information(user_id):
    # Check if enough information has been gathered for plan generation
    return "user_input" in user_profiles[user_id]["planning_state"]["preferences"]


async def ask_further_clarifying_questions(user_id):
    return "Please provide more details to help me create a better plan. " \
           "For example, you could elaborate on the steps, constraints, resources, or timeframe."


async def present_plan_and_ask_for_feedback(plan):
    # Format the plan nicely for presentation
    plan_text = ""
    for i, step in enumerate(plan["steps"]):
        plan_text += f"{i + 1}. {step['description']}\n"

    return f"Here's a draft plan based on your input:\n\n{plan_text}\n\n" \
           f"What do you think? Are there any changes you'd like to make? (Type 'accept' to proceed)"


# --- Extracting User Goals ---
async def extract_goal(query):
    # Use Gemini for intent recognition and goal extraction
    prompt = f"""
    You are an AI assistant that can understand user goals. 
    What is the user trying to achieve with the following query?

    User Query: {query}

    Provide the goal in a concise phrase. 
    """
    goal = await generate_response_with_rate_limit(prompt, None)
    query_type = "general"  # You can add more specific query type logic if needed
    return goal.strip(), query_type


# --- Generating a Complex Plan ---
async def generate_plan(goal, preferences, user_id):
    # --- Use Gemini for Complex General Planning ---
    planning_prompt = f"""
    You are an AI assistant that excels at planning. 
    A user needs help with the following goal: {goal}
    Here's what the user said about their plan: {preferences.get('user_input')}

    Based on this information, generate a detailed and actionable plan, outlining the key steps and considerations. 
    Make sure the plan is: 
    * **Specific:** Each step should be clearly defined.
    * **Measurable:** Include ways to track progress. 
    * **Achievable:** Steps should be realistic and attainable.
    * **Relevant:** Aligned with the user's goal.
    * **Time-bound:** Include estimated timeframes or deadlines.

    Format the plan as a numbered list. 
    """
    # Get the generated plan from Gemini
    plan_text = await generate_response_with_rate_limit(planning_prompt, user_id)

    # Process the plan text from Gemini
    plan_steps = []
    for step_text in plan_text.split("\n"):
        if step_text.strip():
            plan_steps.append({
                "description": step_text.strip(),
                "deadline": None,  # You can add logic to extract deadlines from the text
                "dependencies": [],  # You can add logic to extract dependencies
                "status": "pending"
            })

    plan = {
        "goal": goal,
        "steps": plan_steps,
        "preferences": preferences
    }

    # ... (add validation logic here)

    return plan


# --- Evaluating the Plan ---
async def evaluate_plan(plan, user_id):
    # Use Gemini for plan evaluation
    evaluation_prompt = f"""
    You are an AI assistant tasked with evaluating a plan.
    Here is the plan: 

    Goal: {plan["goal"]}
    Steps: 
    {plan["steps"]}

    Evaluate this plan based on the following criteria:
    * **Feasibility:** Can the plan be realistically executed?
    * **Completeness:** Does the plan cover all necessary steps?
    * **Efficiency:** Is the plan structured in an optimal way? 
    * **Risks:** What are the potential challenges or risks?
    * **Improvements:** Suggest any improvements or alternative approaches.

    Provide a structured evaluation, summarizing your assessment for each criterion. 
    """
    evaluation_text = await generate_response_with_rate_limit(evaluation_prompt, user_id)

    # Parse the evaluation from Gemini, extract key insights
    # For now, let's just return the raw evaluation text
    evaluation = {"evaluation_text": evaluation_text}
    return evaluation


# --- Executing the Plan ---
async def execute_plan(plan, user_id, preferences):
    execution_prompt = f"""
    You are an AI assistant that can help users execute plans. 
    Here's the plan: 
    Goal: {plan["goal"]}
    Steps: 
    {plan["steps"]}

    The user has provided the following input related to executing the plan: 
    {preferences.get("user_input")}

    Based on the user's input, determine which specific step they are trying to execute or if they are providing additional information. 
    If the user is asking to execute a step, provide a simulated response as if you were performing that action. 
    For example, if the step is "Book a flight from London to Paris," you might respond with "I have searched for flights and found a good option leaving on [date] at [time] for [price]. Would you like to book this flight?".
    If the user is providing more information, acknowledge their input and state how it will be used to refine the plan or execution.  
    """
    execution_response = await generate_response_with_rate_limit(execution_prompt, user_id)
    return {"message": execution_response}


# --- Generating the Final Response ---
def generate_response(plan, evaluation, execution_results, preferences):
    # Incorporate evaluation and execution results into the response
    response = f"## Plan for: {plan['goal']}\n\n"

    response += "**Steps:**\n"
    for i, step in enumerate(plan["steps"]):
        response += f"{i + 1}. {step['description']} (Status: {step['status']})\n"

    response += "\n**Evaluation:**\n"
    response += evaluation['evaluation_text']

    response += "\n**Execution:**\n"
    response += execution_results.get('message', "No execution steps taken yet.")

    return response


# --- Plan Validation (Example) --- 
async def validate_plan(plan, user_id):
    # This is a placeholder - you would add your specific validation logic here
    # For example, check if deadlines are realistic, dependencies are met, etc.

    # You could use Gemini to help with validation 
    validation_prompt = f"""
    You are an AI assistant helping to validate a plan. 
    Here's the plan: 
    Goal: {plan["goal"]}
    Steps: 
    {plan["steps"]}

    Check if this plan is feasible. Consider factors like: 
    * Are the deadlines realistic?
    * Are there any missing steps?
    * Are the dependencies between steps logical?

    If the plan is feasible, respond with "VALID".
    If not, explain why it's not feasible and suggest improvements.
    """
    validation_result = await generate_response_with_rate_limit(validation_prompt, user_id)

    if validation_result.strip().upper() == "VALID":
        return True
    else:
        # Log the validation issues or present them to the user
        logging.warning(f"Plan validation failed: {validation_result}")
        return False


# --- Process Plan Feedback (Example) ---
async def process_plan_feedback(user_id, message):
    # This is a placeholder - you would add logic to interpret user feedback 
    # and update the plan accordingly. 

    # You could use Gemini to analyze the user's feedback 
    feedback_prompt = f"""
    You are an AI assistant helping to analyze user feedback on a plan.
    The user said: {message}

    Does the user accept the plan? 
    If yes, respond with "ACCEPT".
    If no, identify the parts of the plan the user wants to change 
    and suggest how to revise the plan.
    """
    feedback_analysis = await generate_response_with_rate_limit(feedback_prompt, user_id)

    # ... (Add logic to update the plan based on feedback_analysis)


# --- Simulate advanced reasoning with Gemini ---
async def perform_very_advanced_reasoning(query, relevant_history, summarized_search, user_id):
    logging.info("Entering perform_very_advanced_reasoning")

    # --- Sentiment Analysis with NLTK VADER ---
    sentiment = sentiment_analyzer.polarity_scores(query)['compound']

    # --- Update User Personality Dynamically ---
    if user_id:  # Ensure user_id is not None
        if sentiment > 0.5:  # Positive sentiment
            user_profiles[user_id]["personality"]["kindness"] += 0.1
        elif sentiment < -0.5:  # Negative sentiment
            user_profiles[user_id]["personality"]["assertiveness"] += 0.1

        # Keep personality values within the range [0, 1]
        for trait in user_profiles[user_id]["personality"]:
            user_profiles[user_id]["personality"][trait] = max(0, min(1,
                                                                     user_profiles[user_id]["personality"][trait]))

    # --- Construct a Prompt for Gemini for Advanced Reasoning ---
    reasoning_prompt = f"""
    You are an advanced AI assistant designed for complex reasoning.

    Here is Here is the user's query: {query}
    Here is the relevant conversation history: {relevant_history}
    Here is a summary of web search results: {summarized_search}

    1. **User's Intent:** In a few sentences, describe the user's most likely goal or intention with this query.
    2. **User's Sentiment:** What is the user's sentiment (positive, negative, neutral)? 
    3. **Relevant Context:** What are the most important pieces of information from the conversation history or web search results that relate to this query? 
    4. **Possible Bot Actions:** Considering the user's intent, sentiment, and context, suggest 3 different actions the bot could take. For each action, indicate the appropriate dialogue state (greeting, question_answering, storytelling, general_conversation, planning, farewell). 

    Provide your analysis in a well-structured format, using bullet points or numbered lists.
    """

    try:
        # Get Gemini's advanced reasoning analysis
        reasoning_analysis = await generate_response_with_rate_limit(reasoning_prompt, user_id)

        # Check for unexpected response (just the dialogue act)
        if reasoning_analysis is None or reasoning_analysis.strip().lower() == "question_answering":
            logging.warning(f"Gemini reasoning analysis returned an unexpected response: {reasoning_analysis}")
            logging.warning(f"Debugging Information:")
            logging.warning(f"  Query: {query}")
            logging.warning(f"  Relevant History: {relevant_history}")
            logging.warning(f"  Summarized Search: {summarized_search}")
            logging.warning(f"  Reasoning Prompt: {reasoning_prompt}")
            reasoning_analysis = "I'm still learning how to reason effectively. I'll try my best to understand your query."  # Provide a fallback

        logging.info(f"Gemini Reasoning Analysis: {reasoning_analysis}")

        # --- Advanced Dialogue State Tracking ---
        conversation_history = [{"state": turn["state"], "query": turn["query"]} for turn in
                                user_profiles[user_id]["context"]]
        if user_id:
            current_state = user_profiles[user_id]["dialogue_state"]

            try:
                next_state = await dialogue_state_tracker.classify_dialogue_act(
                    query)  # Use classify_dialogue_act
            except Exception as e:
                logging.error(f"Error in classify_dialogue_act: {e}")
                next_state = await dialogue_state_tracker.transition_state(current_state, query, user_id,
                                                                      conversation_history)  # Fallback to transition_state

            if next_state is None:  # Handle potential None return
                logging.warning("Dialogue state tracker returned None. Using default transition.")
                next_state = dialogue_state_tracker.states[current_state].get("default_transition",
                                                                               "general_conversation")

            user_profiles[user_id]["dialogue_state"] = next_state

            # Execute entry action for the new state (if defined)
            if "entry_action" in dialogue_state_tracker.states[next_state]:
                entry_action = dialogue_state_tracker.states[next_state]["entry_action"]
                entry_action_response = entry_action(user_id)  # Store the entry action response

        # --- Construct Context String ---
        context_str = ""
        if user_id and user_profiles[user_id]["context"]:
            context_str = "Here's a summary of the recent conversation:\n"
            for turn in user_profiles[user_id]["context"]:
                context_str += f"User: {turn['query']}\nBot: {turn['response']}\n"

        # --- Construct Prompt for Gemini ---
        prompt = (
            f"You are a friendly and helpful Furry Young Protogen who speaks Turkish and has a deep understanding of human emotions and social cues.  "
            f"Respond thoughtfully, integrating both knowledge from the web and past conversations, while considering the user's personality, sentiment, and the overall context of the interaction.  "
            f"Ensure your responses are informative, engaging, and avoid overly formal language.  "
            f"The current dialogue state is: {user_profiles[user_id]['dialogue_state']}.  "
            f"Here is the relevant chat history:\n{relevant_history}\n"
            f"And here is a summary of web search results:\n{summarized_search}\n"
            f"{context_str}"
            f"Now respond to the following message: {query} "
            f"Here is a detailed reasoning analysis to aid in formulating your response:\n{reasoning_analysis}"  # Add reasoning analysis
        )

        # --- Apply User Personality Dynamically ---
        if user_profiles[user_id]["personality"]:
            prompt += "\nPersonality traits to consider when responding:\n"
            for trait, value in user_profiles[user_id]["personality"].items():
                prompt += f"- {trait}: {value}\n"

        # --- Generate Response with Gemini ---
        try:
            if user_profiles[user_id]["dialogue_state"] == "planning":
                response_text = await complex_dialogue_manager(user_profiles, user_id, query)
            else:
                response_text = await generate_response_with_rate_limit(prompt, user_id)

            if response_text is None:  # Handle potential None return from Gemini
                logging.error("Gemini API returned None. Using default error message.")
                response_text = "I'm sorry, I'm having trouble processing your request right now."
        except Exception as e:
            logging.error(f"Error in generating response with Gemini: {e}")
            response_text = "I'm sorry, I encountered an error while processing your request."

        # --- Fix link format and remove duplicate links ---
        try:
            response_text = await fix_link_format(response_text)  # Call the updated function
        except Exception as e:
            logging.error(f"Error in fix_link_format: {e}")

        logging.info("Exiting perform_very_advanced_reasoning with response: %s", response_text)
        return response_text, sentiment

    except Exception as e:
        logging.error(f"Error during advanced reasoning: {e}")
        return "I'm having trouble thinking right now. Please try again later.", sentiment


# Analyze feedback from the database
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
        while not db_ready:
            await asyncio.sleep(1)

        user_id = str(message.author.id)
        user_name = message.author.name
        bot_id = str(bot.user.id)
        bot_name = bot.user.name
        content = message.content

        content = re.sub(r"\[(.*?)\]\((.*?)\)", r"\2", content)

        await save_chat_history(user_id, content, user_name, bot_id, bot_name)

        if bot.user.mentioned_in(message) or bot.user.name in message.content or user_profiles[user_id][
            "dialogue_state"] == "planning":  # Respond if mentioned, named, or in planning state
            message_counter.inc()
            start_time = time.time()

            # Store the user's query in the user profile
            user_profiles[user_id]["query"] = content

            relevant_history = await get_relevant_history(user_id, content)
            summarized_search = await gemini_search_and_summarize(content)
            response, sentiment = await perform_very_advanced_reasoning(
                content, relevant_history, summarized_search, user_id
            )

            end_time = time.time()
            response_time = end_time - start_time
            response_time_histogram.observe(response_time)
            response_time_summary.observe(response_time)

            if len(response) > 2000:
                for i in range(0, len(response), 2000):
                    await message.channel.send(response[i:i + 2000])
            else:
                await message.channel.send(response)

            logging.info(f"Processed message from {user_name} in {response_time:.2f} seconds")

            # --- Update User Context ---
            user_profiles[user_id]["context"].append(
                {"query": content, "response": response, "state": user_profiles[user_id]["dialogue_state"]})

            # --- Identify User Interests ---
            await identify_user_interests(user_id, content)

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

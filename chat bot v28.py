import discord
from discord.ext import tasks
import asyncio
import os
import logging
import json
import time
import aiosqlite
from collections import defaultdict, deque
from prometheus_client import start_http_server, Counter, Histogram, Summary, Gauge
from duckduckgo_search import AsyncDDGS
import google.generativeai as genai
from datetime import datetime, timezone
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transitions import Machine
import pickle
from transformers import pipeline

# Force UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# NLTK Downloads (Ensure these are downloaded)
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("error.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Bot Instance and Environment Variables ---
intents = discord.Intents.all()
intents.message_content = True
intents.members = True

discord_token = "discord-bot-token"  # Replace with your actual API key
gemini_api_key = "gemini-api-key"  # Replace with your actual API key

# --- Gemini AI Configuration ---
genai.configure(api_key=gemini_api_key)
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

# --- Discord Bot Configuration ---
bot = discord.Client(intents=intents)

# --- Directory and Database Setup ---
CODE_DIR = os.path.dirname(__file__)
DB_FILE = os.path.join(CODE_DIR, 'chat_history.db')
USER_PROFILES_FILE = os.path.join(CODE_DIR, "user_profiles.json")
KNOWLEDGE_GRAPH_FILE = os.path.join(CODE_DIR, "knowledge_graph.pkl")

# --- Prometheus Metrics ---
start_http_server(8000)
message_counter = Counter('discord_bot_messages_total', 'Total messages processed')
error_counter = Counter('discord_bot_errors_total', 'Total errors')
response_time_histogram = Histogram('discord_bot_response_time_seconds', 'Response times')
response_time_summary = Summary('discord_bot_response_time_summary', 'Summary of response times')
active_users = Gauge('discord_bot_active_users', 'Number of active users')
feedback_count = Counter('discord_bot_feedback_count', 'Number of feedback messages received')

# --- Context Window and User Profiles ---
CONTEXT_WINDOW_SIZE = 10000
user_profiles = defaultdict(lambda: {
    "preferences": {"communication_style": "friendly", "topics_of_interest": []},
    "demographics": {"age": None, "location": None},
    "history_summary": "",
    "context": deque(maxlen=CONTEXT_WINDOW_SIZE),
    "personality": {"humor": 0.5, "kindness": 0.8, "assertiveness": 0.6},
    "dialogue_state": "greeting",
    "long_term_memory": [],
    "last_bot_action": None,
    "interests": [],
    "query": "",
    "planning_state": {},
    "interaction_history": []
})

# --- Expanded Dialogue States and Bot Actions ---
DIALOGUE_STATES = [
    "greeting",
    "question_answering",
    "storytelling",
    "general_conversation",
    "planning",
    "farewell",
    "seeking_clarification",
    "providing_information",
    "expressing_emotion",
    "making_request",
    "confirming_understanding",
    "negotiating",
    "persuading",
    "arguing",
    "expressing_opinion",
    "sharing_experience",
    "joking"
]

BOT_ACTIONS = [
    "factual_response",
    "creative_response",
    "clarifying_question",
    "change_dialogue_state",
    "initiate_new_topic",
    "generate_plan",
    "execute_plan",
    "provide_summary",
    "express_empathy",
    "fulfill_request",
    "confirm_back",
    "propose_compromise",
    "present_evidence",
    "refute_argument",
    "acknowledge_opinion",
    "relate_to_experience",
    "tell_joke"
]

# --- Initialize Sentiment Analyzer, TF-IDF Vectorizer, Sentence Transformer, and Summarizer ---
sentiment_analyzer = SentimentIntensityAnalyzer()
tfidf_vectorizer = TfidfVectorizer()
sentence_transformer = SentenceTransformer('all-mpnet-base-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# --- Long-Term Memory (Knowledge Graph with Semantic Search) ---
class KnowledgeGraph:
    def __init__(self):
        self.graph = {}
        self.embedding_cache = {}
        self.node_id_counter = 0

    def _generate_node_id(self):
        self.node_id_counter += 1
        return str(self.node_id_counter)

    def add_node(self, node_type, node_id=None, data=None):
        if node_id is None:
            node_id = self._generate_node_id()
        if node_type not in self.graph:
            self.graph[node_type] = {}
        self.graph[node_type][node_id] = data if data is not None else {}
        self.embedding_cache[node_id] = sentence_transformer.encode(str(data))

    def get_node(self, node_type, node_id):
        return self.graph.get(node_type, {}).get(node_id)

    def add_edge(self, source_type, source_id, relation, target_type, target_id, properties=None):
        source_node = self.get_node(source_type, source_id)
        if source_node is not None:
            if "edges" not in source_node:
                source_node["edges"] = []
            source_node["edges"].append({
                "relation": relation,
                "target_type": target_type,
                "target_id": target_id,
                "properties": properties if properties is not None else {}
            })

    def get_related_nodes(self, node_type, node_id, relation=None, direction="outgoing"):
        node = self.get_node(node_type, node_id)
        if node is not None and "edges" in node:
            related_nodes = []
            for edge in node["edges"]:
                if relation is None or edge["relation"] == relation:
                    if direction == "outgoing" or direction == "both":
                        related_nodes.append(self.get_node(edge["target_type"], edge["target_id"]))
                    if direction == "incoming" or direction == "both":
                        # For incoming edges, implement reverse lookup
                        for nt, nodes in self.graph.items():
                            for nid, n in nodes.items():
                                if "edges" in n:
                                    for e in n["edges"]:
                                        if e["target_id"] == node_id and e["relation"] == relation:
                                            related_nodes.append(n)
            return related_nodes
        return []

    def search_nodes(self, query, top_k=3, node_type=None):
        query_embedding = sentence_transformer.encode(query)
        results = []
        for current_node_type, nodes in self.graph.items():
            if node_type is None or current_node_type == node_type:
                for node_id, node_data in nodes.items():
                    node_embedding = self.embedding_cache.get(node_id)
                    if node_embedding is not None:
                        similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
                        results.append((current_node_type, node_id, node_data, similarity))

        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]

    def update_node(self, node_type, node_id, new_data):
        node = self.get_node(node_type, node_id)
        if node is not None:
            self.graph[node_type][node_id].update(new_data)
            self.embedding_cache[node_id] = sentence_transformer.encode(str(new_data))

    def delete_node(self, node_type, node_id):
        if node_type in self.graph and node_id in self.graph[node_type]:
            del self.graph[node_type][node_id]
            del self.embedding_cache[node_id]

            # Remove edges connected to the deleted node
            for nt, nodes in self.graph.items():
                for nid, node in nodes.items():
                    if "edges" in node:
                        node["edges"] = [edge for edge in node["edges"] if edge["target_id"] != node_id]

    def save_to_file(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_file(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)


# Create/Load Knowledge Graph
knowledge_graph = KnowledgeGraph()
if os.path.exists(KNOWLEDGE_GRAPH_FILE):
    knowledge_graph.load_from_file(KNOWLEDGE_GRAPH_FILE)

# --- Long-Term Memory Interaction ---
async def store_long_term_memory(user_id, information_type, information):
    knowledge_graph.add_node(information_type, data={"user_id": user_id, "information": information})
    knowledge_graph.add_edge("user", user_id, "has_" + information_type, information_type,
                             knowledge_graph.node_id_counter - 1)
    knowledge_graph.save_to_file(KNOWLEDGE_GRAPH_FILE)


async def retrieve_long_term_memory(user_id, information_type, query=None, top_k=3):
    if query:
        search_results = knowledge_graph.search_nodes(query, top_k=top_k, node_type=information_type)
        return [(node_type, node_id, node_data) for node_type, node_id, node_data, score in search_results]
    else:
        related_nodes = knowledge_graph.get_related_nodes("user", user_id, "has_" + information_type)
        return related_nodes

# --- Plan Execution and Monitoring ---
async def execute_plan_step(plan, step_index, user_id, message):
    step = plan["steps"][step_index]
    execution_prompt = f"""
    You are an AI assistant helping a user execute a plan. 
    Here's the plan step: {step["description"]}
    The user said: {message.content}

    If the user's message indicates they are ready to proceed with this step, provide a simulated response as if you were completing the step.
    If the user is asking for clarification or modification, acknowledge their request and provide helpful information or guidance. 
    Be specific and relevant to the plan step. 
    """
    try:
        execution_response = await generate_response_with_rate_limit(execution_prompt, user_id)
    except Exception as e:
        logging.error(f"Error executing plan step: {e}")
        return "I encountered an error while trying to execute this step. Please try again later."

    step["status"] = "in_progress"
    await store_long_term_memory(user_id, "plan_execution_result", {
        "step_description": step["description"],
        "result": "in_progress",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    return execution_response


async def monitor_plan_execution(plan, user_id, message):
    current_step_index = next((i for i, step in enumerate(plan["steps"]) if step["status"] == "in_progress"), None)

    if current_step_index is not None:
        if "done" in message.content.lower() or "completed" in message.content.lower():
            plan["steps"][current_step_index]["status"] = "completed"
            await message.channel.send(f"Great! Step {current_step_index + 1} is complete. ")
            if current_step_index + 1 < len(plan["steps"]):
                next_step_response = await execute_plan_step(plan, current_step_index + 1, user_id, message)
                return f"Moving on to the next step: {next_step_response}"
            else:
                return "Congratulations! You have completed all the steps in the plan."
        else:
            return await execute_plan_step(plan, current_step_index, user_id, message)


async def generate_plan(goal, preferences, user_id, message):
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

    For each step, also analyze potential risks and dependencies.
    
    Format the plan as a JSON object with the following structure:
    ```json
    {{
      "goal": "The user's goal",
      "steps": [
        {{
          "description": "Description of the step",
          "deadline": "Optional deadline for the step",
          "dependencies": ["List of dependencies (other step descriptions)"], 
          "risks": ["List of potential risks"],
          "status": "pending" 
        }},
        // ... more steps
      ],
      "preferences": {{
        // User preferences related to the plan
      }}
    }}
    ```
    """
    try:
        plan_text = await generate_response_with_rate_limit(planning_prompt, user_id)
        plan = json.loads(plan_text)
    except (json.JSONDecodeError, Exception) as e:
        logging.error(f"Error parsing plan from JSON or generating plan: {e}")
        return {"goal": goal, "steps": [], "preferences": preferences}

    await store_long_term_memory(user_id, "plan", plan)
    return plan


async def evaluate_plan(plan, user_id):
    evaluation_prompt = f"""
    You are an AI assistant tasked with evaluating a plan, including its potential risks and dependencies.
    Here is the plan: 

    Goal: {plan["goal"]}
    Steps: 
    {json.dumps(plan["steps"], indent=2)} 

    Evaluate this plan based on the following criteria:
    * **Feasibility:** Can the plan be realistically executed?
    * **Completeness:** Does the plan cover all necessary steps?
    * **Efficiency:** Is the plan structured in an optimal way? Are there any redundant or unnecessary steps?
    * **Risks:** Analyze the risks identified for each step. Are they significant? How can they be mitigated?
    * **Dependencies:** Are the dependencies between steps logical and well-defined? Are there any potential conflicts or bottlenecks?
    * **Improvements:** Suggest any improvements or alternative approaches, considering the risks and dependencies.

    Provide a structured evaluation, summarizing your assessment for each criterion. Be as specific as possible in your analysis.
    """
    try:
        evaluation_text = await generate_response_with_rate_limit(evaluation_prompt, user_id)
    except Exception as e:
        logging.error(f"Error evaluating plan: {e}")
        return {"evaluation_text": "I encountered an error while evaluating the plan. Please try again later."}

    await store_long_term_memory(user_id, "plan_evaluation", evaluation_text)
    evaluation = {"evaluation_text": evaluation_text}
    return evaluation


async def validate_plan(plan, user_id):
    validation_prompt = f""" 
    You are a meticulous AI assistant, expert in evaluating the feasibility and safety of plans.
    Carefully analyze the following plan and identify any potential issues, flaws, or missing information that 
    could lead to failure or undesirable outcomes.

    Goal: {plan["goal"]}
    Steps: 
    {json.dumps(plan["steps"], indent=2)} 

    Consider the following aspects:
    * **Clarity and Specificity:** Are the steps clearly defined and specific enough to be actionable?
    * **Realism and Feasibility:** Are the steps realistic and achievable given the user's context and resources?
    * **Dependencies:** Are dependencies between steps clearly stated and logical? Are there any circular dependencies?
    * **Time Constraints:** Are the deadlines realistic and achievable? Are there any potential time conflicts?
    * **Resource Availability:** Are the necessary resources available for each step? 
    * **Risk Assessment:** Have potential risks been adequately identified and analyzed? Are mitigation strategies in place?
    * **Safety and Ethics:** Does the plan adhere to safety and ethical guidelines? Are there any potential negative consequences?

    Provide a detailed analysis of the plan, highlighting any weaknesses or areas for improvement. 
    If the plan is sound and well-structured, state "The plan appears to be valid." 
    Otherwise, provide specific suggestions for making the plan more robust and effective. 
    """

    try:
        validation_result = await generate_response_with_rate_limit(validation_prompt, user_id)
    except Exception as e:
        logging.error(f"Error validating plan: {e}")
        return False, "I encountered an error while validating the plan. Please try again later."

    logging.info(f"Plan validation result: {validation_result}")

    if "valid" in validation_result.lower():
        return True, validation_result
    else:
        return False, validation_result


async def process_plan_feedback(user_id, message):
    feedback_prompt = f"""
    You are an AI assistant helping to analyze user feedback on a plan.
    The user said: {message}

    Does the user accept the plan? 
    If yes, respond with "ACCEPT".
    If no, identify the parts of the plan the user wants to change 
    and suggest how to revise the plan.
    """
    try:
        feedback_analysis = await generate_response_with_rate_limit(feedback_prompt, user_id)
        if "accept" in feedback_analysis.lower():
            return "accept"
        else:
            return feedback_analysis  # Return suggestions for revision
    except Exception as e:
        logging.error(f"Error processing plan feedback: {e}")
        return "I encountered an error while processing your feedback. Please try again later."

# --- User Interest Identification (Word Embeddings & Topic Modeling) ---
user_message_buffer = defaultdict(list)

async def identify_user_interests(user_id, message):
    user_message_buffer[user_id].append(message)
    if len(user_message_buffer[user_id]) >= 5:  # Process every 5 messages
        messages = user_message_buffer[user_id]
        user_message_buffer[user_id] = []  # Clear the buffer
        embeddings = sentence_transformer.encode(messages)
        num_topics = 3  # You can adjust the number of topics 
        kmeans = KMeans(n_clusters=num_topics, random_state=0)
        kmeans.fit(embeddings)
        topic_labels = kmeans.labels_

        for i, message in enumerate(messages):
            user_profiles[user_id]["interests"].append({
                "message": message,
                "embedding": embeddings[i].tolist(), 
                "topic": topic_labels[i]
            })
        save_user_profiles()

async def suggest_new_topic(user_id):
    if user_profiles[user_id]["interests"]:
        interests = user_profiles[user_id]["interests"]
        topic_counts = defaultdict(int)
        for interest in interests:
            topic_counts[interest["topic"]] += 1
        most_frequent_topic = max(topic_counts, key=topic_counts.get)
        suggested_interest = random.choice(
            [interest for interest in interests if interest["topic"] == most_frequent_topic]
        )
        return f"Hey, maybe we could talk more about '{suggested_interest['message']}'? I'm curious to hear your thoughts."
    else:
        return "I'm not sure what to talk about next. What are you interested in?"

# --- Advanced Dialogue State Tracking (with transitions library) ---
class DialogueStateTracker:
    states = DIALOGUE_STATES

    def __init__(self):
        self.machine = Machine(model=self, states=self.states, initial='greeting')

        # --- Define Transitions with Conditions ---
        self.machine.add_transition('greet', 'greeting', 'general_conversation', conditions=['user_says_hello'])
        self.machine.add_transition('ask_question', '*', 'question_answering', conditions=['user_asks_question'])
        self.machine.add_transition('tell_story', '*', 'storytelling', conditions=['user_requests_story'])
        self.machine.add_transition('plan', '*', 'planning', conditions=['user_requests_plan'])
        self.machine.add_transition('farewell', '*', 'farewell', conditions=['user_says_goodbye'])
        self.machine.add_transition('clarify', '*', 'seeking_clarification', conditions=['user_seeks_clarification'])
        self.machine.add_transition('express_emotion', '*', 'expressing_emotion', conditions=['user_expresses_emotion'])
        self.machine.add_transition('make_request', '*', 'making_request', conditions=['user_makes_request'])
        self.machine.add_transition('confirm', '*', 'confirming_understanding', conditions=['user_confirms'])
        self.machine.add_transition('express_opinion', '*', 'expressing_opinion', conditions=['user_expresses_opinion'])
        self.machine.add_transition('share_experience', '*', 'sharing_experience', conditions=['user_shares_experience'])
        self.machine.add_transition('joke', '*', 'joking', conditions=['user_tells_joke'])
        # ... (Add more transitions for other dialogue acts as needed) ...

    # --- Conditions for Transitions ---
    def user_says_hello(self, user_input):
        return any(greeting in user_input.lower() for greeting in ["hi", "hello", "hey"])

    def user_asks_question(self, user_input):
        return any(
            question_word in user_input.lower() for question_word in ["what", "who", "where", "when", "how", "why"]
        )

    def user_requests_story(self, user_input):
        return any(
            story_keyword in user_input.lower()
            for story_keyword in ["tell me a story", "tell a story", "story time"]
        )

    def user_requests_plan(self, user_input):
        return any(
            plan_keyword in user_input.lower() for plan_keyword in ["make a plan", "plan something", "help me plan"]
        )

    def user_says_goodbye(self, user_input):
        return any(goodbye in user_input.lower() for goodbye in ["bye", "goodbye", "see you later"])

    def user_seeks_clarification(self, user_input):
        return any(
            clarification_phrase in user_input.lower()
            for clarification_phrase in ["what do you mean", "can you explain", "i don't understand", "clarify"]
        )

    def user_expresses_emotion(self, user_input):
        sentiment = sentiment_analyzer.polarity_scores(user_input)['compound']
        if sentiment > 0.5 or sentiment < -0.5:
            return True
        emotion_keywords = ["happy", "sad", "angry", "excited", "frustrated", "worried"]
        return any(keyword in user_input.lower() for keyword in emotion_keywords)

    def user_makes_request(self, user_input):
        request_keywords = ["can you", "could you", "would you", "please", "i need"]
        return any(keyword in user_input.lower() for keyword in request_keywords)

    def user_confirms(self, user_input):
        confirmation_keywords = ["yes", "okay", "sure", "correct", "right", "yeah", "do it"]
        return any(keyword in user_input.lower() for keyword in confirmation_keywords)

    def user_expresses_opinion(self, user_input):
        opinion_keywords = ["i think", "in my opinion", "i believe", "it seems to me"]
        return any(keyword in user_input.lower() for keyword in opinion_keywords)

    def user_shares_experience(self, user_input):
        experience_keywords = ["i remember", "one time", "i experienced", "i went through"]
        return any(keyword in user_input.lower() for keyword in experience_keywords)

    def user_tells_joke(self, user_input):
        laughter_indicators = ["haha", "lol", "lmao", "😂", "🤣"]
        return any(indicator in user_input.lower() for indicator in laughter_indicators)

    # State-specific entry actions (Add more as needed)
    def greet_user(self, user_id):
        greetings = [
            f"Hello <@{user_id}>! How can I help you today?",
            f"Hi <@{user_id}>, what's on your mind?",
            f"Hey <@{user_id}>! What can I do for you?"
        ]
        return random.choice(greetings)

    def start_planning(self, user_id):
        user_profiles[user_id]["planning_state"]["preferences"] = {}
        return "Okay, let's start planning. What are you trying to plan?"

    def say_goodbye(self, user_id):
        goodbyes = [
            f"Goodbye, <@{user_id}>! Have a great day!",
            f"See you later, <@{user_id}>!",
            f"Talk to you soon, <@{user_id}>!"
        ]
        return random.choice(goodbyes)

    async def classify_dialogue_act(self, user_input):
        prompt = (
            f"Classify the following user input into one of these dialogue acts: "
            f"{', '.join(DIALOGUE_STATES)}.\n\n"
            f"User input: {user_input}\n\n"
            f"Provide the dialogue act classification as a single word on the first line of the response:"
        )
        logging.info(f"Dialogue Act Classification Prompt: {prompt}")
        try:
            response = await generate_response_with_rate_limit(prompt, None)
            dialogue_act = response.strip().split("\n")[0].lower()
            logging.info(f"Raw Gemini response for Dialogue Act Classification: {response}")
            logging.info(f"Extracted Dialogue Act: {dialogue_act}")
        except Exception as e:
            logging.error(f"Error extracting dialogue act from Gemini response: {e}")
            dialogue_act = None
        return dialogue_act

    async def transition_state(self, current_state, user_input, user_id, conversation_history):
        # Example transitions (add more based on your dialogue flow):
        if self.machine.is_state("greeting") and await self.classify_dialogue_act(user_input) == "general_conversation":
            if self.machine.trigger('greet', user_input=user_input):
                return self.state
        if await self.classify_dialogue_act(user_input) == "question_answering":
            if self.machine.trigger('ask_question', user_input=user_input):
                return self.state
        if await self.classify_dialogue_act(user_input) == "storytelling":
            if self.machine.trigger('tell_story', user_input=user_input):
                return self.state
        if await self.classify_dialogue_act(user_input) == "planning":
            if self.machine.trigger('plan', user_input=user_input):
                return self.state
        if await self.classify_dialogue_act(user_input) == "farewell":
            if self.machine.trigger('farewell', user_input=user_input):
                return self.state

        # ... (Add more transitions for other dialogue acts) ...

        return "general_conversation"  # Default fallback state

# Initialize Dialogue State Tracker
dialogue_state_tracker = DialogueStateTracker()

# --- Gemini AI Rate Limit Handling ---
RATE_LIMIT_PER_MINUTE = 60  # Adjust as needed based on your Gemini API limits
RATE_LIMIT_WINDOW = 60
user_last_request_time = defaultdict(lambda: 0)
global_last_request_time = 0
global_request_count = 0

@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
async def generate_response_with_rate_limit(prompt, user_id=None):
    global global_last_request_time, global_request_count
    current_time = time.time()

    # Global rate limit
    if current_time - global_last_request_time < RATE_LIMIT_WINDOW:
        global_request_count += 1
        if global_request_count > RATE_LIMIT_PER_MINUTE:
            sleep_time = RATE_LIMIT_WINDOW - (current_time - global_last_request_time)
            await asyncio.sleep(sleep_time)
            global_request_count = 0
    else:
        global_request_count = 0
    global_last_request_time = current_time

    # Per-user rate limit (if user_id is provided)
    if user_id:
        time_since_last_request = current_time - user_last_request_time[user_id]
        if time_since_last_request < RATE_LIMIT_WINDOW / RATE_LIMIT_PER_MINUTE:
            sleep_time = RATE_LIMIT_WINDOW / RATE_LIMIT_PER_MINUTE - time_since_last_request
            await asyncio.sleep(sleep_time)
        user_last_request_time[user_id] = time.time()

    for attempt in range(3):  # Retry up to 3 times
        try:
            response = model.generate_content(prompt)
            logging.info(f"Raw Gemini response: {response}")
            return response.text
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limited
                logging.error(f"Gemini rate limit exceeded for user {user_id}: {e}")
                await asyncio.sleep(5)  # Wait a bit longer before retrying
                continue
            elif e.response.status_code == 500:  # Server error
                logging.error(f"Gemini server error: {e}")
                return "I'm experiencing some technical difficulties with Gemini. Please try again later."
            else:
                raise  # Reraise other HTTP errors
        except Exception as e:
            logging.exception(f"Error generating response with Gemini for user {user_id}: {e}")
            if attempt < 2:
                await asyncio.sleep(2)
                continue
            else:
                return "I'm sorry, I encountered an unexpected error while processing your request."

# --- Gemini Search and Summarization ---
async def gemini_search_and_summarize(query) -> str:
    try:
        ddg = AsyncDDGS()
        search_results = await asyncio.to_thread(ddg.text, query, max_results=100) 

        search_results_text = ""
        for index, result in enumerate(search_results):
            search_results_text += f'[{index}] Title: {result["title"]}\nSnippet: {result["body"]}\n\n'

        prompt = (
            f"You are a helpful AI assistant. A user asked about '{query}'. "
            f"Here are some relevant web search results:\n\n"
            f"{search_results_text}\n\n"
            f"Please provide a concise and informative summary of these search results."
        )

        # Dynamically set max_length based on prompt length
        max_summary_length = min(8192, len(prompt) // 2) 
        summarization_config = generation_config.copy()
        summarization_config["max_output_tokens"] = max_summary_length
        summarizer_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-exp-0827",
            generation_config=summarization_config,
        )
        response = summarizer_model.generate_content(prompt)

        return response.text

    except Exception as e:
        logging.error(f"Gemini search and summarization error: {e}")
        return "I'm sorry, I encountered an error while searching and summarizing information for you."

# --- URL Extraction from Description ---
async def extract_url_from_description(description):
    search_query = f"{description} site:youtube.com OR site:twitch.tv OR site:instagram.com OR site:twitter.com"

    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://duckduckgo.com/html/?q={search_query}") as response:
            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")

            first_result = soup.find("a", class_="result__a")
            if first_result:
                return first_result["href"]
            else:
                return None

# --- Link Format Fixing ---
async def fix_link_format(text):
    new_text = ""
    lines = text.split("\n")
    for line in lines:
        match = re.match(r"\[(.*?)\]\(link\)", line)
        if match:
            link_text = match.group(1)
            url = await extract_url_from_description(link_text)
            if url:
                new_text += f"[{link_text}]({url})\n"
            else:
                new_text += line + "\n"
        else:
            new_text += line + "\n"

    return new_text

# --- Complex Dialogue Manager ---
async def complex_dialogue_manager(user_profiles, user_id, message):
    planning_state = user_profiles[user_id].setdefault("planning_state", {})
    
    if user_profiles[user_id]["dialogue_state"] == "planning":
        planning_state.setdefault("stage", "initial_request")

        if planning_state["stage"] == "initial_request":
            goal, query_type = await extract_goal(user_profiles[user_id]["query"])
            planning_state["goal"] = goal
            planning_state["query_type"] = query_type
            planning_state["stage"] = "gathering_information"
            return await ask_clarifying_questions(goal, query_type)

        elif planning_state["stage"] == "gathering_information":
            await process_planning_information(user_id, message)
            if await has_enough_planning_information(user_id):
                planning_state["stage"] = "generating_plan"
                plan = await generate_plan(
                    planning_state["goal"],
                    planning_state["preferences"],
                    user_id,
                    message
                )
                is_valid, validation_result = await validate_plan(plan, user_id)
                if is_valid:
                    planning_state["plan"] = plan
                    planning_state["stage"] = "presenting_plan"
                    return await present_plan_and_ask_for_feedback(plan)
                else:
                    planning_state["stage"] = "gathering_information"
                    return f"The plan has some issues: {validation_result} Please provide more information or adjust your preferences."
            else:
                return await ask_further_clarifying_questions(user_id)

        elif planning_state["stage"] == "presenting_plan":
            feedback_result = await process_plan_feedback(user_id, message.content)
            if feedback_result == "accept":
                planning_state["stage"] = "evaluating_plan"
                evaluation = await evaluate_plan(planning_state["plan"], user_id)
                planning_state["evaluation"] = evaluation
                planning_state["stage"] = "executing_plan"
                initial_execution_message = await execute_plan_step(
                    planning_state["plan"], 0, user_id, message
                )
                return (
                    generate_response(
                        planning_state["plan"],
                        evaluation,
                        {},
                        planning_state["preferences"] 
                    )
                    + "\n\n"
                    + initial_execution_message
                )
            else:
                planning_state["stage"] = "gathering_information"
                return f"Okay, let's revise the plan. Here are some suggestions: {feedback_result} What changes would you like to make?"

        elif planning_state["stage"] == "executing_plan":
            execution_result = await monitor_plan_execution(
                planning_state["plan"], user_id, message
            )
            return execution_result

# --- Planning Helper Functions ---
async def ask_clarifying_questions(goal, query_type):
    return "To create an effective plan, I need some more details. Could you tell me:\n" \
           f"- What is the desired outcome of this plan?\n" \
           f"- What are the key steps or milestones involved?\n" \
           f"- Are there any constraints or limitations I should be aware of?\n" \
           f"- What resources or tools are available to you?\n" \
           f"- What is the timeframe for completing this plan?"


async def process_planning_information(user_id, message):
    user_profiles[user_id]["planning_state"]["preferences"]["user_input"] = message.content


async def has_enough_planning_information(user_id):
    return "user_input" in user_profiles[user_id]["planning_state"]["preferences"]


async def ask_further_clarifying_questions(user_id):
    return "Please provide more details to help me create a better plan. " \
           "For example, you could elaborate on the steps, constraints, resources, or timeframe."


async def present_plan_and_ask_for_feedback(plan):
    plan_text = ""
    for i, step in enumerate(plan["steps"]):
        plan_text += f"{i + 1}. {step['description']}\n"
    return f"Here's a draft plan based on your input:\n\n{plan_text}\n\n" \
           f"What do you think? Are there any changes you'd like to make? (Type 'accept' to proceed)"

# --- Goal Extraction ---
async def extract_goal(query):
    prompt = f"""
    You are an AI assistant that can understand user goals. 
    What is the user trying to achieve with the following query?

    User Query: {query}

    Provide the goal in a concise phrase. 
    """
    try:
        goal = await generate_response_with_rate_limit(prompt, None)
    except Exception as e:
        logging.error(f"Error extracting goal: {e}")
        return "I couldn't understand your goal. Please try rephrasing.", "general"
    return goal.strip(), "general"

# --- Response Generation ---
def generate_response(plan, evaluation, execution_results, preferences):
    response = f"## Plan for: {plan['goal']}\n\n"
    response += "**Steps:**\n"
    for i, step in enumerate(plan["steps"]):
        response += f"{i + 1}. {step['description']} (Status: {step['status']})\n"
    response += "\n**Evaluation:**\n"
    response += evaluation['evaluation_text']
    response += "\n**Execution:**\n"
    response += execution_results.get('message', "No execution steps taken yet.")
    return response

# --- Advanced Reasoning and Response Generation ---
async def perform_very_advanced_reasoning(query, relevant_history, summarized_search, user_id, message, content):
    logging.info("Entering perform_very_advanced_reasoning")

    # Check if user profile exists before accessing personality
    if user_id and user_id in user_profiles:
        sentiment = sentiment_analyzer.polarity_scores(query)['compound']

        # Dynamically adjust personality traits
        if sentiment > 0.5:
            user_profiles[user_id]["personality"]["kindness"] += 0.1
            user_profiles[user_id]["personality"]["humor"] += 0.05
        elif sentiment < -0.5:
            user_profiles[user_id]["personality"]["assertiveness"] += 0.1
            user_profiles[user_id]["personality"]["humor"] -= 0.05

        # Keep personality traits within 0-1 range
        for trait, value in user_profiles[user_id]["personality"].items():
            user_profiles[user_id]["personality"][trait] = max(0, min(1, value))

    else:
        # Handle case where user profile is not found
        logging.warning(f"User profile not found for user ID: {user_id}")
        sentiment = 0

    # --- Reasoning Prompt for Gemini ---
    reasoning_prompt = f"""
    You are an advanced AI assistant designed for complex reasoning and deep thinking.

    Here is the user's query: {query}
    Here is the relevant conversation history: {relevant_history}
    Here is a summary of web search results: {summarized_search}

    1. **User's Intent:** Describe the user's most likely goal or intention. Be specific.
    2. **User's Sentiment:** What is the user's sentiment (positive, negative, neutral)? Explain your reasoning.
    3. **Relevant Context:** What is the most important information from the conversation history or web search results? Explain its relevance.
    4. **Possible Bot Actions:** Considering the user's intent, sentiment, and context, suggest 3 different actions the bot could take. For each action:
        * Describe the action in detail.
        * Indicate the appropriate dialogue state (greeting, question_answering, storytelling, general_conversation, planning, farewell, seeking_clarification, providing_information, expressing_emotion, making_request, confirming_understanding, negotiating, persuading, arguing, expressing_opinion, sharing_experience, joking).
        * Explain why this action is appropriate in this context.

    Provide your analysis in a well-structured format using bullet points or numbered lists. Be thorough and insightful in your reasoning. 
    """

    try:
        reasoning_analysis = await generate_response_with_rate_limit(reasoning_prompt, user_id)
    except Exception as e:
        logging.error(f"Error during advanced reasoning: {e}")
        return "I'm having trouble thinking right now. Please try again later.", sentiment

    # Check for unexpected responses from Gemini
    if reasoning_analysis is None or reasoning_analysis.strip().lower() == "question_answering":
        logging.warning(f"Gemini reasoning analysis returned an unexpected response: {reasoning_analysis}")
        reasoning_analysis = "I'm still learning how to reason effectively. I'll try my best to understand your query."

    logging.info(f"Gemini Reasoning Analysis: {reasoning_analysis}")

    # --- Dialogue State Transition ---
    if user_id and user_id in user_profiles:
        current_state = user_profiles[user_id]["dialogue_state"]
        try:
            next_state = await dialogue_state_tracker.classify_dialogue_act(query)
        except Exception as e:
            logging.error(f"Error in classify_dialogue_act: {e}")
            next_state = await dialogue_state_tracker.transition_state(
                current_state, query, user_id, user_profiles[user_id]["context"]
            )
        if next_state is None:
            logging.warning("Dialogue state tracker returned None. Using default transition.")
            next_state = "general_conversation"
        user_profiles[user_id]["dialogue_state"] = next_state

    # --- Retrieve Relevant Past Data from Knowledge Graph ---
    if user_id:
        past_plans = await retrieve_long_term_memory(user_id, "plan", query=content, top_k=3)
        past_evaluations = await retrieve_long_term_memory(
            user_id, "plan_evaluation", query=content, top_k=3
        )
    else:
        past_plans = []
        past_evaluations = []

    # --- Summarize Relevant History ---
    max_history_summary_length = max(50, len(relevant_history) // 2)
    summarized_history = summarizer(
        relevant_history, max_length=max_history_summary_length, min_length=50
    )[0]["summary_text"]

    # --- Construct Prompt for Gemini Response ---
    context_str = ""
    if user_id and user_id in user_profiles and user_profiles[user_id]["context"]:
        context_str = "Here's a summary of the recent conversation:\n"
        for turn in user_profiles[user_id]["context"]:
            if 'query' in turn and 'response' in turn:
                context_str += f"User: {turn['query']}\nBot: {turn['response']}\n"
            else:
                logging.warning(f"Incomplete turn in context: {turn}")

    prompt = (
        f"You are a friendly and helpful AI assistant. "
        f"Respond thoughtfully, integrating knowledge from the web and past conversations, considering the user's personality, sentiment, and the context. "
        f"Ensure your responses are informative, engaging, and avoid overly formal language. "
        f"The current dialogue state is: {user_profiles.get(user_id, {}).get('dialogue_state', 'general_conversation')}. "
        f"Here is a summary of the relevant chat history:\n{summarized_history}\n"
        f"And here is a summary of web search results:\n{summarized_search}\n"
        f"{context_str}"
        f"Now respond to the following message: {query} "
        f"Here is a detailed reasoning analysis to aid in formulating your response:\n{reasoning_analysis}"
        f"Here are some relevant past plans:\n{past_plans}\n"
        f"And here are some relevant past plan evaluations:\n{past_evaluations}\n"
    )

    # --- Incorporate Personality Traits ---
    if user_id and user_id in user_profiles and user_profiles[user_id]["personality"]:
        prompt += "\nPersonality traits to consider when responding:\n"
        for trait, value in user_profiles[user_id]["personality"].items():
            prompt += f"- {trait}: {value}\n"

    # --- Generate Response with Gemini ---
    try:
        if (
            user_id
            and user_id in user_profiles
            and user_profiles[user_id]["dialogue_state"] == "planning"
        ):
            response_text = await complex_dialogue_manager(user_profiles, user_id, message)
        else:
            response_text = await generate_response_with_rate_limit(prompt, user_id)

        if response_text is None:
            logging.error("Gemini API returned None. Using default error message.")
            response_text = (
                "I'm sorry, I'm having trouble processing your request right now."
            )
    except Exception as e:
        logging.error(f"Error in generating response with Gemini: {e}")
        response_text = (
            "I'm sorry, I encountered an error while processing your request."
        )

    # --- Explainable AI: Add Reasoning Explanation (More detailed) ---
    explanation = f"**Reasoning:**\n{reasoning_analysis}\n"
    explanation += (
        f"**Dialogue State:** {user_profiles.get(user_id, {}).get('dialogue_state', 'general_conversation')}\n"
    )
    explanation += (
        f"**Personality Influences:** {user_profiles.get(user_id, {}).get('personality', 'Not yet established')}\n"
    )
    response_text = response_text + "\n\n" + explanation

    # --- Fix Link Formatting ---
    try:
        response_text = await fix_link_format(response_text)
    except Exception as e:
        logging.error(f"Error in fix_link_format: {e}")

    logging.info("Exiting perform_very_advanced_reasoning with response: %s", response_text)
    return response_text, sentiment

# --- Feedback Analysis from Database ---
async def analyze_feedback_from_db():
    try:
        async with aiosqlite.connect(DB_FILE) as db:
            async with db.execute("SELECT feedback FROM feedback") as cursor:
                async for row in cursor:
                    feedback = row[0]
                    logging.info(f"Feedback: {feedback}")
    except Exception as e:
        logging.error(f"Feedback analysis exception: {e}")

# --- Database Interaction ---
db_ready = False
db_lock = asyncio.Lock()
db_queue = asyncio.Queue()

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

async def init_db():
    global db_ready
    async with db_lock:
        await create_chat_history_table()
        db_ready = True

def load_user_profiles():
    if os.path.exists(USER_PROFILES_FILE):
        with open(USER_PROFILES_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

def save_user_profiles():
    profiles_copy = defaultdict(lambda: {
        "preferences": {"communication_style": "friendly", "topics_of_interest": []},
        "demographics": {"age": None, "location": None},
        "history_summary": "",
        "context": [],
        "personality": {"humor": 0.5, "kindness": 0.8, "assertiveness": 0.6},
        "dialogue_state": "greeting",
        "long_term_memory": [],
        "last_bot_action": None,
        "interests": [],
        "query": "",
        "planning_state": {},
        "interaction_history": []
    })

    for user_id, profile in user_profiles.items():
        profiles_copy[user_id].update(profile)
        profiles_copy[user_id]["context"] = list(profile["context"])

    with open(USER_PROFILES_FILE, "w") as f:
        json.dump(profiles_copy, f, indent=4)

async def save_chat_history(user_id, message, user_name, bot_id, bot_name):
    await db_queue.put((user_id, message, user_name, bot_id, bot_name))

async def process_db_queue():
    while True:
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

async def save_feedback_to_db(user_id, feedback):
    async with db_lock:
        async with aiosqlite.connect(DB_FILE) as db:
            await db.execute(
                "INSERT INTO feedback (user_id, feedback, timestamp) VALUES (?, ?, ?)",
                (user_id, feedback, datetime.now(timezone.utc).isoformat())
            )
            await db.commit()
    feedback_count.inc()

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

        max_summary_length = max(50, len(history_text) // 2)  

        logging.info(f"Summarizer input length: {len(history_text)}")
        logging.info(f"Summarizer max_length: {max_summary_length}")
        summarized_history = summarizer(history_text, max_length=max_summary_length, min_length=50)[0]["summary_text"]
        logging.info(f"Summarizer output length: {len(summarized_history)}")

        return summarized_history 

# --- Discord Events ---
@bot.event
async def on_ready():
    logging.info(f"Logged in as {bot.user.name} ({bot.user.id})")
    global user_profiles
    user_profiles = load_user_profiles()
    await init_db()
    bot.loop.create_task(process_db_queue())
    await analyze_feedback_from_db()


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
async def on_message(message):
    try:
        if message.author == bot.user:
            return

        message_counter.inc()
        active_users.inc()
        start_time = time.time()

        user_id = str(message.author.id)
        content = message.content.strip()

        if user_id not in user_profiles:
            user_profiles[user_id] = {
                "preferences": {"communication_style": "friendly", "topics_of_interest": []},
                "demographics": {"age": None, "location": None},
                "history_summary": "",
                "context": deque(maxlen=CONTEXT_WINDOW_SIZE),
                "personality": {"humor": 0.5, "kindness": 0.8, "assertiveness": 0.6},
                "dialogue_state": "greeting",
                "long_term_memory": [],
                "last_bot_action": None,
                "interests": [],
                "query": "",
                "planning_state": {},
                "interaction_history": []
            }
            logging.info(f"Created new profile for user {user_id}")

        user_profiles[user_id]["context"].append({"role": "user", "content": content})

        if bot.user.mentioned_in(message):
            await identify_user_interests(user_id, content)

            relevant_history = await get_relevant_history(user_id, content)
            summarized_search = await gemini_search_and_summarize(content)

            response_text, sentiment = await perform_very_advanced_reasoning(
                content, relevant_history, summarized_search, user_id, message, content
            )

            user_profiles[user_id]["context"].append({"role": "bot", "content": response_text})
            await save_chat_history(bot.user.id, response_text, bot.user.name, bot.user.id, bot.user.name)

            if len(response_text) > 4000:
                await send_long_message(message.channel, response_text)
            elif len(response_text) > 1000:
                try:
                    prompt = (
                        f"Please provide a concise summary of the following text:\n\n"
                        f"{response_text}"
                    )
                    max_summary_length = min(3900, len(response_text) // 2)
                    summarization_config = generation_config.copy()
                    summarization_config["max_output_tokens"] = max_summary_length
                    summarizer_model = genai.GenerativeModel(
                        model_name="gemini-1.5-flash-exp-0827",
                        generation_config=summarization_config,
                    )
                    gemini_summary = summarizer_model.generate_content(prompt)
                    summarized_response = gemini_summary.text
                    await message.channel.send(summarized_response)
                except Exception as e:
                    logging.error(f"Error summarizing response with Gemini: {e}")
                    await send_long_message(message.channel, response_text)
            else:
                await message.channel.send(response_text)

        await save_chat_history(user_id, content, message.author.name, bot.user.id, bot.user.name)

        end_time = time.time()
        response_time = end_time - start_time
        response_time_histogram.observe(response_time)
        response_time_summary.observe(response_time)

    except Exception as e:
        logging.error(f"Error processing message: {e}", exc_info=True)
        error_counter.inc()
        await message.channel.send("I'm sorry, I encountered an error while processing your request.")

    finally:
        active_users.dec()

async def send_long_message(channel, message_text):
    chunk_size = 2000  # Adjust as needed
    chunks = [message_text[i:i + chunk_size] for i in range(0, len(message_text), chunk_size)]
    for chunk in chunks:
        await channel.send(chunk) 

bot.run(discord_token)

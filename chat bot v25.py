import discord
from discord.ext import tasks
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
from sentence_transformers import SentenceTransformer  # For word embeddings
from sklearn.cluster import KMeans  # For topic modeling

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
bot = discord.Client(intents=intents)  # Use discord.Client instead of commands.Bot

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

# --- Initialize Sentence Transformer for Word Embeddings ---
sentence_transformer = SentenceTransformer('all-mpnet-base-v2')  # You can choose a different model

# --- Long-Term Memory (Using a Knowledge Graph with Semantic Search) ---

class KnowledgeGraph:
    def __init__(self):
        self.graph = {}
        self.embedding_cache = {}  # Cache embeddings for faster retrieval

    def add_node(self, node_type, node_id, data):
        if node_type not in self.graph:
            self.graph[node_type] = {}
        self.graph[node_type][node_id] = data
        self.embedding_cache[node_id] = sentence_transformer.encode(str(data))  # Cache the embedding

    def get_node(self, node_type, node_id):
        return self.graph.get(node_type, {}).get(node_id)

    def add_edge(self, source_type, source_id, relation, target_type, target_id):
        source_node = self.get_node(source_type, source_id)
        if source_node is not None:
            if "edges" not in source_node:
                source_node["edges"] = []
            source_node["edges"].append({
                "relation": relation,
                "target_type": target_type,
                "target_id": target_id
            })

    def get_related_nodes(self, node_type, node_id, relation):
        node = self.get_node(node_type, node_id)
        if node is not None and "edges" in node:
            related_nodes = []
            for edge in node["edges"]:
                if edge["relation"] == relation:
                    related_nodes.append(self.get_node(edge["target_type"], edge["target_id"]))
            return related_nodes
        return []

    def search_nodes(self, query, top_k=3):
        """
        Performs semantic search on the knowledge graph using word embeddings.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.

        Returns:
            list: A list of tuples (node_type, node_id, node_data, score) representing the search results,
                  sorted by relevance score.
        """
        query_embedding = sentence_transformer.encode(query)
        results = []
        for node_type, nodes in self.graph.items():
            for node_id, node_data in nodes.items():
                node_embedding = self.embedding_cache.get(node_id)  # Get cached embedding
                if node_embedding is not None:
                    similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]  # Calculate similarity
                    results.append((node_type, node_id, node_data, similarity))

        results.sort(key=lambda x: x[3], reverse=True)  # Sort by similarity score (descending)
        return results[:top_k]

# Create a knowledge graph instance
knowledge_graph = KnowledgeGraph()

async def store_long_term_memory(user_id, information_type, information):
    """Stores information in the user's long-term memory (knowledge graph)."""
    node_id = str(time.time())  # Use timestamp as a unique node ID
    knowledge_graph.add_node(information_type, node_id, information)
    knowledge_graph.add_edge("user", user_id, "has_" + information_type, information_type, node_id)

async def retrieve_long_term_memory(user_id, information_type, query=None, top_k=3):
    """Retrieves information from the user's long-term memory (knowledge graph)."""
    if query:
        # Perform semantic search using the query
        search_results = knowledge_graph.search_nodes(query, top_k=top_k)
        return [(node_type, node_id, node_data) for node_type, node_id, node_data, score in search_results]
    else:
        # If no query is provided, retrieve all nodes of the specified type related to the user
        related_nodes = knowledge_graph.get_related_nodes("user", user_id, "has_" + information_type)
        return related_nodes

# --- Plan Execution and Monitoring ---
async def execute_plan_step(plan, step_index, user_id, message):
    """Simulates the execution of a plan step."""
    step = plan["steps"][step_index]

    # --- Simulate Execution Based on Step Description ---
    execution_prompt = f"""
    You are an AI assistant helping a user execute a plan. 
    Here's the plan step: {step["description"]}
    The user said: {message.content}

    If the user's message indicates they are ready to proceed with this step, provide a simulated response as if you were completing the step.
    If the user is asking for clarification or modification, acknowledge their request and provide helpful information or guidance. 
    Be specific and relevant to the plan step. 
    """
    execution_response = await generate_response_with_rate_limit(execution_prompt, user_id)

    # Update the step status in the plan (you might need more sophisticated logic here)
    step["status"] = "in_progress"

    # Store the execution result in the knowledge graph
    await store_long_term_memory(user_id, "plan_execution_result", {
        "step_description": step["description"],
        "result": "in_progress",  # Or "failure" if the step failed, "completed" if done
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return execution_response  # Return the response for this execution step

async def monitor_plan_execution(plan, user_id, message):
    """Monitors the execution of a plan and handles unexpected events."""
    current_step_index = next((i for i, step in enumerate(plan["steps"]) if step["status"] == "in_progress"), None)

    if current_step_index is not None:
        # If a step is in progress, check if the user's message indicates completion or modification
        if "done" in message.content.lower() or "completed" in message.content.lower():
            # If the step is done, mark it as completed and proceed to the next step
            await execute_plan_step(plan, current_step_index, user_id, message)  # Mark current as completed

            if current_step_index + 1 < len(plan["steps"]):
                # Start the next step
                next_step_response = await execute_plan_step(plan, current_step_index + 1, user_id, message)
                return f"Okay, step {current_step_index + 1} is complete. Moving on to the next step: {next_step_response}"
            else:
                # All steps completed
                return "Congratulations! You have completed all the steps in the plan."
        else:
            # If the user's message doesn't indicate completion, continue the current step
            return await execute_plan_step(plan, current_step_index, user_id, message)

# --- Generating a Complex Plan ---
async def generate_plan(goal, preferences, user_id, message):
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
    # Get the generated plan from Gemini
    plan_text = await generate_response_with_rate_limit(planning_prompt, user_id)

    # --- Parse the plan from JSON ---
    try:
        plan = json.loads(plan_text)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing plan from JSON: {e}")
        return {"goal": goal, "steps": [], "preferences": preferences}  # Return an empty plan

    # --- Store Plan in Long-Term Memory ---
    await store_long_term_memory(user_id, "plan", plan)

    # --- Trigger Plan Monitoring (Execution starts when user indicates readiness) ---
    # asyncio.create_task(execute_plan_step(plan, 0, user_id, message))  # Start executing the first step
    asyncio.create_task(monitor_plan_execution(plan, user_id, message))

    return plan

# --- Evaluating the Plan ---
async def evaluate_plan(plan, user_id):
    # Use Gemini for plan evaluation, including risk assessment
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
    evaluation_text = await generate_response_with_rate_limit(evaluation_prompt, user_id)

    # --- Store Evaluation in Long-Term Memory ---
    await store_long_term_memory(user_id, "plan_evaluation", evaluation_text)

    # Parse the evaluation from Gemini, extract key insights (you can add more parsing logic here)
    evaluation = {"evaluation_text": evaluation_text}
    return evaluation

# --- Advanced Plan Validation ---
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

    validation_result = await generate_response_with_rate_limit(validation_prompt, user_id)
    logging.info(f"Plan validation result: {validation_result}") 

    # Basic check for validity (you can add more sophisticated analysis of validation_result)
    if "valid" in validation_result.lower():
        return True, validation_result 
    else:
        return False, validation_result

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

# --- Function to Identify User Interests using Word Embeddings and Topic Modeling---
user_message_buffer = defaultdict(list)  # Buffer to store user messages
async def identify_user_interests(user_id, message):
    # 1. Accumulate Messages 
    user_message_buffer[user_id].append(message)

    # 2. Cluster when you have enough messages
    if len(user_message_buffer[user_id]) >= 5: # Cluster every 5 messages (adjust as needed)
        messages = user_message_buffer[user_id]
        user_message_buffer[user_id] = []  # Clear the buffer

        # 3. Word Embeddings
        embeddings = sentence_transformer.encode(messages)  

        # 4. Topic Modeling (KMeans Clustering)
        num_topics = 3 
        kmeans = KMeans(n_clusters=num_topics, random_state=0)
        kmeans.fit(embeddings) 
        topic_labels = kmeans.labels_ 

        # 5. Store Interests in User Profile
        for i, message in enumerate(messages):
            user_profiles[user_id]["interests"].append({
                "message": message,
                "embedding": embeddings[i].tolist(), 
                "topic": topic_labels[i]
            })

        save_user_profiles(user_profiles)  # Save updated profiles

# --- Function to Suggest a New Topic ---
async def suggest_new_topic(user_id):
    if user_profiles[user_id]["interests"]:
        interests = user_profiles[user_id]["interests"]

        # Find the most frequent topic
        topic_counts = defaultdict(int)
        for interest in interests:
            topic_counts[interest["topic"]] += 1
        most_frequent_topic = max(topic_counts, key=topic_counts.get)

        # Suggest a topic related to the most frequent topic
        suggested_interest = random.choice([interest for interest in interests if interest["topic"] == most_frequent_topic])
        return f"Hey, maybe we could talk more about '{suggested_interest['message']}'? I'm curious to hear your thoughts."
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
                    "farewell": ["bye", "goodbye", "see you later", "talk to you later"]
                },
                "default_transition": "general_conversation",
                "entry_action": self.greet_user
            },
            "general_conversation": {
                "transitions": {
                    "storytelling": ["tell me a story", "can you tell me a story", "know any good stories"],
                    "question_answering": ["what is", "who is", "where is", "how to"],
                    "planning": ["plan", "make a plan", "help me plan"],  # Transition to planning state
                    "farewell": ["bye", "goodbye", "see you later", "talk to you later"]
                },
                "default_transition": "general_conversation"
            },
            "storytelling": {
                "transitions": {
                    "general_conversation": ["that's enough", "stop the story", "change topic", "enough stories"],
                    "question_answering": ["what happened", "who is that", "where did that happen"],
                    "planning": ["plan", "make a plan", "help me plan"],  # Transition to planning state
                    "farewell": ["bye", "goodbye", "see you later", "talk to you later"]
                },
                "default_transition": "storytelling"
            },
            "question_answering": {
                "transitions": {
                    "general_conversation": ["thanks", "got it", "okay", "that's helpful"],
                    "storytelling": ["tell me a story", "can you tell me a story", "know any good stories"],
                    "planning": ["plan", "make a plan", "help me plan"],  # Transition to planning state
                    "farewell": ["bye", "goodbye", "see you later", "talk to you later"]
                },
                "default_transition": "general_conversation"
            },
            "planning": {  # New state for planning
                "transitions": {
                    "general_conversation": ["done", "finished", "that's it", "no more planning"],
                    "question_answering": ["what about", "how about"],
                    "farewell": ["bye", "goodbye", "see you later", "talk to you later"]
                },
                "default_transition": "planning",
                "entry_action": self.start_planning  # You might have a specific entry action
            },
            "farewell": {
                "transitions": {},
                "default_transition": "farewell",
                "entry_action": self.say_goodbye  # Add a goodbye action
            }
        }

    def greet_user(self, user_id):
        greetings = [
            f"Hello <@{user_id}>! How can I help you today?",
            f"Hi <@{user_id}>, what's on your mind?",
            f"Hey <@{user_id}>! What can I do for you?"
        ]
        return random.choice(greetings)

    def start_planning(self, user_id):
        user_profiles[user_id]["planning_state"]["preferences"] = {}  # Initialize preferences
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

        # --- Check for Contextual Transitions (Example based on User Profile) ---
        if current_state == "general_conversation" and any(
                "story" in interest["message"].lower() for interest in user_profiles[user_id]["interests"]):
            if any(keyword in user_input.lower() for keyword in transitions["storytelling"]):
                return "storytelling"

        # Check for keyword-based transitions
        for next_state, keywords in transitions.items():
            if any(keyword in user_input.lower() for keyword in keywords):
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
                    user_id,
                    message  # Pass the message object here
                )

                # Validate the plan
                is_valid, validation_result = await validate_plan(plan, user_id)
                if is_valid:
                    # Store the plan
                    user_profiles[user_id]["planning_state"]["plan"] = plan

                    # Move to the next stage
                    user_profiles[user_id]["planning_state"]["stage"] = "presenting_plan"

                    # Present the plan and ask for feedback
                    return await present_plan_and_ask_for_feedback(plan)
                else:
                    # If the plan is not valid, go back to gathering information
                    user_profiles[user_id]["planning_state"]["stage"] = "gathering_information"
                    return f"The plan has some issues: {validation_result} Please provide more information or adjust your preferences."

            else:
                return await ask_further_clarifying_questions(user_id)

        elif user_profiles[user_id]["planning_state"]["stage"] == "presenting_plan":
            # Process user feedback on the plan
            await process_plan_feedback(user_id, message.content)

            # Move to the next stage (evaluating or executing, depending on the feedback)
            if "accept" in message.content.lower():  # Check if the user accepted the plan
                user_profiles[user_id]["planning_state"]["stage"] = "evaluating_plan"

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

            else:  # If the user wants to make changes
                user_profiles[user_id]["planning_state"]["stage"] = "gathering_information"
                return "Okay, let's revise the plan. What changes would you like to make?"

        elif user_profiles[user_id]["planning_state"]["stage"] == "executing_plan":
            # Step 4: Execute the plan
            execution_result = await execute_plan(
                user_profiles[user_id]["planning_state"]["plan"],
                user_id,
                user_profiles[user_id]["planning_state"]["preferences"],
                message
            )

            return execution_result

async def ask_clarifying_questions(goal, query_type):
    # --- More generalized clarifying questions for any type of plan ---
    return "To create an effective plan, I need some more details. Could you tell me:\n" \
           f"- What is the desired outcome of this plan?\n" \
           f"- What are the key steps or milestones involved?\n" \
           f"- Are there any constraints or limitations I should be aware of?\n" \
           f"- What resources or tools are available to you?\n" \
           f"- What is the timeframe for completing this plan?"

async def process_planning_information(user_id, message):
    # Extract relevant planning information from the user's message
    user_profiles[user_id]["planning_state"]["preferences"]["user_input"] = message.content

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

# --- Example Execute Plan Function ---
async def execute_plan(plan, user_id, preferences, message):
    """Executes the plan step-by-step, interacting with the user."""
    for i, step in enumerate(plan["steps"]):
        if step["status"] == "pending":
            await message.channel.send(f"**Step {i+1}:** {step['description']}\n"
                                       f"Please let me know when you have completed this step.")
            step["status"] = "in_progress"  # Update the step status

            # Wait for the user to indicate completion
            def check(m):
                return m.author.id == int(user_id) and "done" in m.content.lower()
            try:
                await bot.wait_for('message', check=check, timeout=600)  # Wait for 10 minutes
                step["status"] = "completed"
                await message.channel.send(f"Great! Step {i+1} is complete. Moving on...")
            except asyncio.TimeoutError:
                await message.channel.send(f"Step {i+1} timed out. Let's move on for now.")

    await message.channel.send("All planned steps have been processed!")
    return {"message": "Plan execution complete."}

async def ask_clarifying_questions(goal, query_type):
    # --- More generalized clarifying questions for any type of plan ---
    return "To create an effective plan, I need some more details. Could you tell me:\n" \
           f"- What is the desired outcome of this plan?\n" \
           f"- What are the key steps or milestones involved?\n" \
           f"- Are there any constraints or limitations I should be aware of?\n" \
           f"- What resources or tools are available to you?\n" \
           f"- What is the timeframe for completing this plan?"

async def process_planning_information(user_id, message):
    # Extract relevant planning information from the user's message
    user_profiles[user_id]["planning_state"]["preferences"]["user_input"] = message.content

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

# --- Simulate advanced reasoning with Gemini ---
async def perform_very_advanced_reasoning(query, relevant_history, summarized_search, user_id, message, content):
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
                                                                     user_profiles[user_id]["personality"][
                                                                         trait]))

    # --- Construct a Prompt for Gemini for Advanced Reasoning ---
    reasoning_prompt = f"""
    You are an advanced AI assistant designed for complex reasoning and deep thinking.

    Here is the user's query: {query}
    Here is the relevant conversation history: {relevant_history}
    Here is a summary of web search results: {summarized_search}

    1. **User's Intent:** In a few sentences, describe the user's most likely goal or intention with this query. Be as specific as possible.
    2. **User's Sentiment:** What is the user's sentiment (positive, negative, neutral)? Explain the reasoning behind your assessment.
    3. **Relevant Context:** What are the most important pieces of information from the conversation history or web search results that relate to this query? Explain how this context is relevant.
    4. **Possible Bot Actions:** Considering the user's intent, sentiment, and context, suggest 3 different actions the bot could take. For each action:
        * Describe the action in detail.
        * Indicate the appropriate dialogue state (greeting, question_answering, storytelling, general_conversation, planning, farewell).
        * Explain why this action would be appropriate in this context.

    Provide your analysis in a well-structured format, using bullet points or numbered lists. Be thorough and insightful in your reasoning. 
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

        # --- Retrieve Long-Term Memory (Contextualized) ---
        past_plans = await retrieve_long_term_memory(user_id, "plan", query=content, top_k=3)
        past_evaluations = await retrieve_long_term_memory(user_id, "plan_evaluation", query=content, top_k=3)

        # --- Construct Context String ---
        context_str = ""
        if user_id and user_profiles[user_id]["context"]:
            context_str = "Here's a summary of the recent conversation:\n"
            for turn in user_profiles[user_id]["context"]:
                context_str += f"User: {turn['query']}\nBot: {turn['response']}\n"

        # --- Construct Prompt for Gemini ---
        prompt = (
            f"You are a friendly and helpful AI assistant who has a deep understanding of human emotions and social cues.  "
            f"Respond thoughtfully, integrating both knowledge from the web and past conversations, while considering the user's personality, sentiment, and the overall context of the interaction.  "
            f"Ensure your responses are informative, engaging, and avoid overly formal language.  "
            f"The current dialogue state is: {user_profiles[user_id]['dialogue_state']}.  "
            f"Here is the relevant chat history:\n{relevant_history}\n"
            f"And here is a summary of web search results:\n{summarized_search}\n"
            f"{context_str}"
            f"Now respond to the following message: {query} "
            f"Here is a detailed reasoning analysis to aid in formulating your response:\n{reasoning_analysis}"  # Add reasoning analysis
            f"Here are some relevant past plans:\n{past_plans}\n"
            f"And here are some relevant past plan evaluations:\n{past_evaluations}\n"
        )

        # --- Apply User Personality Dynamically ---
        if user_profiles[user_id]["personality"]:
            prompt += "\nPersonality traits to consider when responding:\n"
            for trait, value in user_profiles[user_id]["personality"].items():
                prompt += f"- {trait}: {value}\n"

        # --- Generate Response with Gemini ---
        try:
            if user_profiles[user_id]["dialogue_state"] == "planning":
                response_text = await complex_dialogue_manager(user_profiles, user_id, message)
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

# --- Analyze feedback from the database ---
async def analyze_feedback_from_db():
    try:
        async with aiosqlite.connect(DB_FILE) as db:
            async with db.execute("SELECT feedback FROM feedback") as cursor:
                async for row in cursor:
                    feedback = row[0]
                    logging.info(f"Feedback: {feedback}")
    except Exception as e:
        logging.error(f"Feedback analysis exception: {e}")

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
    # Create a deep copy of the profiles to avoid modifying the original data
    profiles_copy = defaultdict(lambda: {"preferences": {}, "history_summary": "",
                                            "context": [],  # Initialize context as an empty list
                                            "personality": {"humor": 0.5, "kindness": 0.8, "assertiveness": 0.6},
                                            "dialogue_state": "greeting", "long_term_memory": [],
                                            "last_bot_action": None, "interests": [],
                                            "query": "", "planning_state": {}})

    for user_id, profile in profiles.items():
        # Convert the deque to a list
        profiles_copy[user_id].update(profile)
        profiles_copy[user_id]["context"] = list(profile["context"])

    with open(USER_PROFILES_FILE, "w") as f:
        json.dump(profiles_copy, f, indent=4) 

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

    for attempt in range(3):  # Retry up to 3 times
        try:
            response = model.generate_content(prompt)
            logging.info(f"Raw Gemini response: {response}")
            return response.text
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response.status_code == 429:  # Rate limited (HTTP 429)
                logging.error(f"Gemini rate limit exceeded for user {user_id}: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                continue
            else:
                logging.error(f"Gemini API error for user {user_id}: {e}")
                return "I'm experiencing some technical difficulties with Gemini. Please try again later."
        except Exception as e:
            logging.exception(f"Error generating response with Gemini for user {user_id}: {e}")
            if attempt < 2:  # Retry only for specific errors (e.g., KeyError)
                await asyncio.sleep(2)  # Wait before retrying
                continue
            else:
                return "I'm sorry, I encountered an unexpected error while processing your request."

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
                content, relevant_history, summarized_search, user_id, message, content
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

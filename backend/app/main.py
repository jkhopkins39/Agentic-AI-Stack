import os # Importing os module for operating system functionalities
import shutil # Importing shutil module for high-level file operations
import psycopg2 # PostgreSQL database adapter
from psycopg2.extras import RealDictCursor # For dict-like cursor results
import uuid # For generating unique session IDs
import json # For JSON operations
import re # For regex pattern matching
import time
import asyncio
from typing import Tuple
import concurrent.futures
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from threading import Thread
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from datetime import datetime, timedelta # For date/time operations
import smtplib # For email service
from email.mime.text import MIMEText # Additionally used for email service
from email.mime.multipart import MIMEMultipart # Additionally used for email service
SMTP_AVAILABLE = True
from typing import Annotated, Literal, Optional, List, Dict, Any # Different data types we need
from dotenv import load_dotenv # Importing dotenv to get API key from .env file
from pydantic import BaseModel, Field # Used for validation and structuring message classification
from typing_extensions import TypedDict # State typed dict contains typed keys messages, message_type, order_data
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Data path reads in txt file for policy RAG
DATA_PATH = os.path.join(os.getcwd())
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    print(f"Loaded {len(documents)} documents from {DATA_PATH}")
    return documents

# This will print out the entire PDF that is stored in documents array
# print(documents[0])

def split_text(documents: list[Document]):
    # Initialize text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, # Size of each chunk in characters
        length_function=len, # Function to compute the length of the text
        chunk_overlap=100, # Overlap between consec chunks
        add_start_index=True, # Flag to add start index to each chunk
    )

    # Make our list of chunks of text, could handle splitting of multiple documents
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    document = chunks[0]

    # This is so we can visualize what just happened and what was split and how
    print(document.page_content)
    print(document.metadata)

    return chunks

# Path to the directory to save Chroma database
CHROMA_PATH = "chroma"
def save_to_chroma(chunks: list[Document]):
    """
    Save the given list of Document objects to a Chroma database.
    Args:
    chunks (list[Document]): List of Document objects representing text chunks to save.
    Returns:
    None
    """
    import time
  
    # Clear out the existing database directory if it exists
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing database at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
        time.sleep(1)  # Gives filesystem time to clean up
  
    # Ensure the directory is completely gone
    while os.path.exists(CHROMA_PATH):
        time.sleep(0.5)
  
    print(f"Creating new database with {len(chunks)} chunks...")
  
    try:
        # Create a new Chroma database from the documents using OpenAI embeddings
        # https://medium.com/@callumjmac/implementing-rag-in-langchain-with-chroma-a-step-by-step-guide-16fc21815339
        # We need to site this as a source in our final documentation
        db = Chroma.from_documents(
            chunks,
            OpenAIEmbeddings(),
            persist_directory=CHROMA_PATH
        )
        print(f"Successfully saved {len(chunks)} chunks to {CHROMA_PATH}.")
    
    except Exception as e:
        print(f"Error creating database: {e}")
        # If there's an error, clean up any partial database
        if os.path.exists(CHROMA_PATH):
          shutil.rmtree(CHROMA_PATH)
        raise


def load_policy_text():
    # Load policy as a document, this run uses the txt not pdf
    policy_path = "policy.txt"
    if os.path.exists(policy_path):
        with open(policy_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [Document(page_content=content, metadata={"source": policy_path})]
    return []

def generate_data_store():
    documents = load_documents() # Load documents from a source
    policy_docs = load_policy_text() # Load policy text
    all_documents = documents + policy_docs # Combine all documents
    chunks = split_text(all_documents) # Split documents into chunks
    save_to_chroma(chunks) # Save the processed data to a data store

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(',')
VERBOSE_CONSUMER_LOGS = os.getenv("VERBOSE_CONSUMER_LOGS", "false").lower() in {"1", "true", "yes", "on"}


def verbose_polling_log(message: str) -> None:
    """Emit noisy consumer/orchestrator logs only when explicitly enabled."""
    if VERBOSE_CONSUMER_LOGS:
        print(message)

from contextlib import asynccontextmanager
PRIORITY_TOPICS = {
    'Order': ['tasks.order.p1', 'tasks.order.p2', 'tasks.order.p3'],
    'Email': ['tasks.email.p1', 'tasks.email.p2', 'tasks.email.p3'],
    'Policy': ['tasks.policy.p1', 'tasks.policy.p2', 'tasks.policy.p3'],
    'Message': ['tasks.message.p1', 'tasks.message.p2', 'tasks.message.p3'],
}


class PriorityConsumer:
    """
    Consumes from priority topics (p1, p2, p3) and processes with agents
    """
    def __init__(self):
        self.consumers = {}
        self.running = False
        
    async def start(self):
        """Start all priority consumers"""
        self.running = True
        
        # Start a consumer task for each agent type
        tasks = [
            asyncio.create_task(self.run_priority_consumer("Order", order_agent)),
            asyncio.create_task(self.run_priority_consumer("Email", email_agent)),
            asyncio.create_task(self.run_priority_consumer("Policy", policy_agent)),
            asyncio.create_task(self.run_priority_consumer("Message", message_agent)),
        ]
        
        print("🔵 All priority consumers started")
        await asyncio.gather(*tasks)
        
    async def run_priority_consumer(self, agent_type: str, agent_handler_func):
        """
        Run a priority consumer that checks p1 > p2 > p3 in order
        """
        topics = PRIORITY_TOPICS[agent_type]
    
        # Create consumers for all priority levels
        consumers = {}
        for i, topic in enumerate(topics, start=1):
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                group_id=f"{agent_type.lower()}-agent-group-v2",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                session_timeout_ms=30000,  # Increased to 30s to handle email operations
                request_timeout_ms=30000
            )
            await consumer.start()
            
            # Wait for assignment
            while not consumer.assignment():
                await asyncio.sleep(0.5)
            
    
            consumers[i] = consumer
            print(f"🔵 Started consumer for {topic}")
        
        # Debug info
        print(f"✅ {agent_type} has {len(consumers)} consumers ready")
        for i, c in consumers.items():
            print(f"   P{i}: {c.assignment()}")
        
        # Priority polling loop
        last_p3_check = time.time()
        p3_check_interval = 5.0
        
        try:
            while self.running:
                event = None
                current_priority = None
                
                # Check p1 (highest priority) first
                try:
                    data = await consumers[1].getmany(timeout_ms=50, max_records=1)
                    if data:
                        for tp, messages in data.items():
                            if messages:
                                msg = messages[0]
                                event, current_priority = msg.value, 1
                                topic = topics[0]
                                print(f"✅ {agent_type} got message from p1")
                                break
                except Exception as e:
                    print(f"⚠️ {agent_type} p1: {type(e).__name__}: {e}")

                # Then p2
                if not event:
                    try:
                        data = await consumers[2].getmany(timeout_ms=50, max_records=1)
                        if data:
                            for tp, messages in data.items():
                                if messages:
                                    msg = messages[0]
                                    event, current_priority = msg.value, 2
                                    topic = topics[1]
                                    print(f"✅ {agent_type} got message from p2")
                                    break
                    except Exception as e:
                        print(f"⚠️ {agent_type} p2: {type(e).__name__}: {e}")

                # Check p3 periodically  
                if not event:
                    now = time.time()
                    if now - last_p3_check >= p3_check_interval:
                        last_p3_check = now
                        verbose_polling_log(f"🔍 {agent_type} checking p3...")
                        try:
                            data = await consumers[3].getmany(timeout_ms=50, max_records=1)
                            if data:
                                for tp, messages in data.items():
                                    if messages:
                                        msg = messages[0]
                                        event, current_priority = msg.value, 3
                                        topic = topics[2]
                                        print(f"✅ {agent_type} got message from p3")
                                        break
                        except Exception as e:
                            print(f"⚠️ {agent_type} p3: {type(e).__name__}: {e}")
                
                if event:
                    # Skip messages older than 1 hour to prevent backlog processing
                    event_time = event.get("event_timestamp", 0)
                    message_age_ms = time.time() * 1000 - event_time
                    if message_age_ms > 3600000:  # 1 hour = 3,600,000 ms
                        print(f"⏭️  Skipping old message (age: {message_age_ms/1000:.0f}s)")
                        await consumers[current_priority].commit()
                        continue  # Skip to next message
                    
                    session_id = event.get("session_id", "unknown")
                    print(f"→ Consumed {session_id} from {topic}")
    
                    try:
                        # Extract state
                        state = event.get("state", {})
                        
                        # Process with agent handler
                        if asyncio.iscoroutinefunction(agent_handler_func):
                            result = await agent_handler_func(state)
                        else:
                            result = agent_handler_func(state)
                        
                        # Extract agent response
                        last_message = result.get("messages", [{}])[-1]
                        agent_response = last_message.get("content", "")
                        
                        # Clean agent prefix if present
                        if ": " in agent_response:
                            agent_response = agent_response.split(": ", 1)[1]
                        
                        print(f"→ {agent_type} agent processed {session_id}")

                        # Get conversation info
                        conversation_id = event.get("conversation_id")
                        query_text = event.get("query_text", state.get("messages", [{}])[0].get("content", "") if state.get("messages") else "")

                        # Save query and response to conversation
                        if conversation_id and query_text:
                            try:
                                conn = get_database_connection()
                                if conn:
                                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                                        cursor.execute(
                                            "SELECT COUNT(*) as count FROM queries WHERE conversation_id = %s",
                                            (conversation_id,)
                                        )
                                        result_count = cursor.fetchone()
                                        message_order = (result_count['count'] if result_count else 0) + 1

                                    save_query_to_conversation(
                                        conversation_id=conversation_id,
                                        user_message=query_text,
                                        agent_type=agent_type,
                                        agent_response=agent_response,
                                        message_order=message_order,
                                        user_id=state.get("user_data", {}).get("id") if isinstance(state.get("user_data"), dict) else None
                                    )
                                    print(f"→ Saved message to conversation {conversation_id}")
                            except Exception as save_error:
                                print(f"⚠️ Error saving to conversation: {save_error}")
                        
                        # Publish response to agent.responses
                        response_event = {
                            "session_id": session_id,
                            "conversation_id": conversation_id,
                            "agent_type": agent_type.upper() + "_AGENT",
                            "message": agent_response,
                            "status": "completed",
                            "priority": current_priority,
                            "timestamp": int(time.time() * 1000),
                            "correlation_id": event.get("correlation_id", f"corr-{uuid.uuid4()}")
                        }
                        
                        await producer.send_and_wait(
                            RESPONSE_TOPIC,
                            value=response_event,
                            key=session_id
                        )
                        
                        print(f"→ Published agent response for {session_id}")
                        
                        # Commit offset after successful processing
                        await consumers[current_priority].commit()
                        
                    except Exception as e:
                        print(f"✗ {agent_type} agent error for {session_id}: {e}")
                
                # Small sleep to prevent tight loop
                await asyncio.sleep(0.01)
                
        finally:
            for consumer in consumers.values():
                await consumer.stop()
            print(f"🔴 {agent_type} priority consumer stopped")
    async def stop(self):
        """Stop all consumers"""
        self.running = False

    app = FastAPI()

    # Globals
    producer: AIOKafkaProducer | None = None
    routing_task: asyncio.Task | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global producer, priority_consumer
        
        # Startup
    ensure_default_users()
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda v: v.encode("utf-8") if v else None
    )
    await producer.start()
    print("✓ Kafka producer started")
        
    # Start orchestrator consumer
    asyncio.create_task(orchestrator_consumer())
        
    # Start priority consumers
    priority_consumer = PriorityConsumer()
    asyncio.create_task(priority_consumer.start())
        
    yield
        
    # Shutdown
    if priority_consumer:
        await priority_consumer.stop()
        
    if producer:
        await producer.stop()
        print("✗ Kafka producer stopped")

app = FastAPI(lifespan=lifespan)


producer: AIOKafkaProducer | None = None

INGRESS_TOPIC = 'system.ingress'
RESPONSE_TOPIC = 'agent.responses'

async def publish_to_kafka(topic: str, priority: int, payload: dict, key: str | None = None):
    if not producer:
        raise RuntimeError("Kafka producer not initialized")
    try:
        await producer.send_and_wait(
            topic,
            value=payload,
            key=key  # ✅ Let the serializer handle encoding
        )
    except Exception as e:
        print(f"❌ Publish error: {e}")
        raise


# This will be called automatically when needed
# generate_data_store()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""

def query_rag(query_text):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
    Args:
        - query_text (str): The text to query the RAG system with.
    Returns:
        - formatted_response (str): Formatted response including the generated text and sources.
        - response_text (str): The generated response text.
    """
    # YOU MUST - Use same embedding function as before
    embedding_function = OpenAIEmbeddings()

    # Check if database exists, if not, generate it
    if not os.path.exists(CHROMA_PATH):
        print("Database not found. Generating new data store...")
        generate_data_store()

    # Prepare the database with error handling
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    except Exception as e:
        print(f"Error loading database: {e}")
        print("Regenerating database...")
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        # In case it failed to earlier
        generate_data_store()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
  
    # This searches the chroma vector database for documents most similar to query_text. Limits it to top 5 results
    results = db.similarity_search_with_relevance_scores(query_text, k=5)

    # If there are no results retrieved in the search, return error
    if len(results) == 0:
        print(f"⚠️ No results found in RAG database for query: {query_text}")
        return None, "I couldn't find relevant policy information. Please contact customer support for assistance."

    # Filter results by relevance score (lower threshold to include more context)
    # Relevance scores in Chroma are typically between 0 and 1, with higher being better
    # Lower the threshold to 0.5 to get more context
    filtered_results = [(doc, score) for doc, score in results if score >= 0.5]
    
    if len(filtered_results) == 0:
        print(f"⚠️ All results below relevance threshold. Using all results anyway.")
        filtered_results = results

    # Combine context from matching documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
    
    print(f"✓ Found {len(filtered_results)} relevant chunks for policy query")
    print(f"Context preview: {context_text[:200]}...")
 
    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt_template.format(context=context_text, question=query_text)
  
    # Initialize OpenAI chat model (use same instance as rest of code)
    model = ChatOpenAI()

    # Generate response text based on the prompt
    # ChatOpenAI can accept a string directly, but using messages format for consistency
    messages = [{"role": "user", "content": formatted_prompt}]
    response = model.invoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)
 
    # Get sources of the matching documents
    sources = [doc.metadata.get("source", None) for doc, _score in filtered_results]
 
    # Format and return response including generated text and sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, response_text

# Uncomment the lines below to test
# query_text = "What is your return policy?"
# formatted_response, response_text = query_rag(query_text)
# print(response_text)

load_dotenv()

# Session configuration - deprecated, use user_email from state instead
# Keeping as fallback for backward compatibility only
SESSION_EMAIL = "jeremyyhop@gmail.com"

from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-haiku-20240307")

# Load ChatGPT 4o Mini
#llm = init_chat_model("openai:gpt-4o-mini")

#Define message classifier and insert our model options as the literal types
class MessageClassifier(BaseModel):
    message_type: Literal["Order", "Email", "Policy", "Message", "Order Receipt", "Change Information"] = Field(
        ...,
        description="Classify if the user message is related to orders, emails, policy,"+
        " order receipt requests, changing user information, and if it's none of those: messaging."
    )

# Define user information parser for change information requests
# The user information parser is used to break a query into variables that can be used to update the database
class UserInformationParser(BaseModel):
    current_user_email: Optional[str] = Field(None, description="The user's current email address mentioned in the request")
    new_user_email: Optional[str] = Field(None, description="The new email address the user wants to change to")
    current_firstname: Optional[str] = Field(None, description="The user's current first name mentioned in the request")
    new_firstname: Optional[str] = Field(None, description="The new first name the user wants to change to")
    current_lastname: Optional[str] = Field(None, description="The user's current last name mentioned in the request")
    new_lastname: Optional[str] = Field(None, description="The new last name the user wants to change to")
    current_phone: Optional[str] = Field(None, description="The user's current phone number mentioned in the request")
    new_phone: Optional[str] = Field(None, description="The new phone number the user wants to change to")

# Define order receipt parser for order receipt requests
# Similar to the user information parser, this is used to understand a query in terms of variables
class OrderReceiptParser(BaseModel):
    user_email: Optional[str] = Field(None, description="The user's email address to identify their orders")
    order_number: Optional[str] = Field(None, description="Specific order number mentioned (e.g., ORD-2024-001)")
    chronological_request: Optional[str] = Field(None, description="Chronological request like 'last', 'most recent', 'latest', 'previous'")
    product_name: Optional[str] = Field(None, description="Product name mentioned to find orders containing that product")
    time_reference: Optional[str] = Field(None, description="Time reference like 'yesterday', 'last week', 'this month'")

# LangGraph uses states to inform each node, messages is a list that stores the conversation history
# State must track these variables so that the agentic system as a whole retains context
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str
    order_data: dict  # Store structured order data for agents to use
    user_data: dict  # Store user information for change requests
    parsed_changes: dict  # Store parsed change information
    conversation_id: str  # Track conversation ID
    session_id: str  # Track session ID
    conversation_context: dict  # Store conversation context and memory
    message_order: int  # Track message order in conversation
    information_changed: bool  # Flag to indicate if user information was changed
    changes_made: list  # List of changes made for email notification
    needs_email_notification: bool  # Flag to trigger email notification

def classify_priority(query_text: str, message_type: str, event: dict) -> int:
    """
    Classify message priority:
    1 = Critical (errors, urgent issues)
    2 = High (orders, account changes)
    3 = Normal (general queries)
    """
    query_lower = query_text.lower()
    
    # P1: Critical
    if any(word in query_lower for word in ['urgent', 'emergency', 'error', 'broken', 'not working']):
        return 1
    
    # P2: High priority
    if message_type in ['Order', 'Change Information']:
        return 2
    
    # P3: Normal
    return 3

#last message is stored in the -1 column of our messages array.
def classify_message(state: State):
    last_message = state["messages"][-1]
    # Handle both dict and object message formats
    message_content = last_message.get("content") if isinstance(last_message, dict) else last_message.content
    print(f"CLASSIFYING MESSAGE: {message_content}")

    #we use a LangChain method to wrap the base language model to conform with message classifier schema. 
    classifier_llm = llm.with_structured_output(MessageClassifier)

    #Result is stored as result of classifier invocation, we can print and return the message type and message itself later
    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as one of the following:
            - 'Order': if the user asks about specific order information such as shipping status, tracking, price, order quantity, order number, etc.
            - 'Email': if the user explicitly mentions email or requests information to be sent via email
            - 'Policy': if the user asks about returns, refunds, return policy, shipping policy,
            exchange policy, warranty, terms of service, company policies, return timeframes, return process, or any store/company rules and procedures
            - 'Order Receipt': if the user asks about seeing the receipt of a previous order
            - 'Change Information': if the user requests to change, update, or modify their personal information like name, email, phone number, address, etc.
            - 'Message': if the user asks a general question not related to orders, policies, or email requests
            
            Examples of Policy questions:
            - "What is your return policy?"
            - "How long do I have to return an item?"
            - "Can I return this product?"
            - "What are your shipping policies?"
            - "Do you accept returns?"
            - "How do returns work?"
            
            Be very specific: 
            - If the message contains words like 'return', 'policy', 'refund', 'exchange', 'warranty', or asks about company procedures, classify as 'Policy'
            - If the message contains words like 'change', 'update', 'modify' combined with personal information (name, email, phone, address), classify as 'Change Information'
            - If the message asks for receipts or receipt information, classify as 'Order Receipt'"""
        },
        {"role": "user", "content": message_content}
    ])
    print(f"CLASSIFIED AS: {result.message_type}")
    
    priority = classify_priority(
        message_content,  # ✅ Use the variable you already extracted above
        result.message_type,
        state
    )
    
    return {
        "message_type": result.message_type,
        "priority": priority
    }


#Pretty basic router that gets the message type of our current state, routes to respective node
def router(state: State):
    message_type = state.get("message_type", "Message")
    if message_type == "Order":
        return {"next": "order"}
    if message_type == "Email":
        return {"next": "email"}
    if message_type == "Policy":
        return {"next": "policy"}
    if message_type == "Order Receipt":
        return {"next": "email"}
    if message_type == "Change Information":
        return {"next": "message"}
    if message_type == "Message":
        return {"next": "message"}
    return {"next": "message"}

# The order agent will be used for retrieval of order data upon request

async def order_agent(state: State):
    last_message = state["messages"][-1]

    user_message = last_message.get("content") if isinstance(last_message, dict) else last_message.content

    session_id = state.get("session_id", str(uuid.uuid4()))
    conversation_id = state.get("conversation_id", str(uuid.uuid4()))
    
    # Get user information from state
    user_email = state.get("user_email") or (state.get("user_data", {}).get("email") if isinstance(state.get("user_data"), dict) else None)
    user_id = state.get("user_id") or (state.get("user_data", {}).get("id") if isinstance(state.get("user_data"), dict) else None)
    
    # Look up user's orders if email is available
    orders_info = ""
    if user_email:
        try:
            orders = lookup_orders_by_email(user_email, limit=10)
            if orders:
                orders_list = []
                for order in orders:
                    # Handle items - could be list, array, or None
                    order_items = order.get("items", [])
                    if order_items is None:
                        order_items = []
                    elif not isinstance(order_items, list):
                        # Convert to list if it's not already
                        try:
                            order_items = list(order_items) if order_items else []
                        except:
                            order_items = []
                    
                    # Build items summary
                    if order_items:
                        items_summary = ", ".join([f"{item.get('product_name', 'Unknown')} (x{item.get('quantity', 1)})" for item in order_items[:3]])
                        if len(order_items) > 3:
                            items_summary += f" and {len(order_items) - 3} more item(s)"
                    else:
                        items_summary = "No items listed"
                    
                    # Format order date
                    created_at = order.get('created_at')
                    if created_at:
                        if hasattr(created_at, 'strftime'):
                            created_at_str = created_at.strftime('%Y-%m-%d')
                        else:
                            created_at_str = str(created_at)[:10]  # Take first 10 chars (YYYY-MM-DD)
                    else:
                        created_at_str = "Unknown"
                    
                    orders_list.append(
                        f"Order {order.get('order_number', 'N/A')}: "
                        f"Status: {order.get('status', 'Unknown')}, "
                        f"Total: ${order.get('total_amount', 0):.2f}, "
                        f"Created: {created_at_str}, "
                        f"Items: {items_summary}"
                    )
                orders_info = "\n\n".join(orders_list)
                print(f"Found {len(orders)} orders for user {user_email}")
            else:
                orders_info = "No orders found for this account."
                print(f"No orders found for user {user_email}")
        except Exception as e:
            print(f"Error looking up orders: {e}")
            import traceback
            traceback.print_exc()
            orders_info = "Unable to retrieve order information at this time."
    else:
        orders_info = "User email not available. Please ask the customer for their order number or email address."
    
    # Build system prompt with user's order context
    system_prompt = f"""You are an order agent. Your job is to help customers with information related to their orders. You can
    fetch orders based on order number, tell the user what the shipping status of their order is, and when orders are created,
    you are to create an autonomous, standardized response.
    Do not directly mention the inner workings of this system, instead focus on the user's requests.

    CUSTOMER ACCOUNT INFORMATION:
    - Email: {user_email or "Not provided"}
    - User ID: {user_id or "Not provided"}

    CUSTOMER'S ORDER HISTORY:
    {orders_info if orders_info else "No order information available. Ask the customer for their order number or email address."}

    INSTRUCTIONS:
    - If the customer asks about "my orders" or "my order status", refer to the order history above
    - If they mention a specific order number, look for it in the order history above
    - If they ask about "my recent order" or "last order", refer to the most recent order in the list
    - If no matching order is found in the history above, ask the customer for their order number
    - Be helpful and specific when referencing their orders
    - Use the order numbers, statuses, and item information from the order history above"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_message
        }
    ]

    reply = await llm.ainvoke(messages)
    print(f"Order agent response: {reply.content}")

    return {
        "messages": [{"role": "assistant", "content": f"Order Agent: {reply.content}"}]
    }


import uuid, time

async def email_agent(state: State):
    last_message = state["messages"][-1]
    user_message = last_message.get("content") if isinstance(last_message, dict) else last_message.content

    message_type = state.get("message_type", "Email")
    conversation_context = state.get("conversation_context", {})

    session_id = state.get("session_id", str(uuid.uuid4()))
    conversation_id = state.get("conversation_id", str(uuid.uuid4()))

    # Branch 1: triggered email notification
    if state.get("needs_email_notification") and state.get("changes_made"):
        return await handle_information_change_notification(state)

    # Branch 2: order receipt
    if message_type == "Order Receipt":
        return await handle_order_receipt_request(state)

    # Default branch: fall back to LLM
    messages = [
        {"role": "system", "content": """You are an email agent. 
        Your job is to help customers by delivering data in a structured format via email.
        Do not expose system internals; focus only on the user’s request."""},
        {"role": "user", "content": user_message},
    ]

    reply = await llm.ainvoke(messages)
    print(f"Email agent response: {reply.content}")

    return {
        "messages": [{"role": "assistant", "content": f"Email Agent: {reply.content}"}]
    }


# It checks the state for changes made and sends an email notification
def handle_information_change_notification(state: State):
    """Handle sending email notifications for information changes"""
    changes_made = state.get("changes_made", [])
    
    if not changes_made:
        return {"messages": [{"role": "assistant", "content": "Email Agent: No changes to notify about."}]}
    
    # Get user email from state (logged-in user context)
    user_email = state.get("user_email") or (state.get("user_data", {}).get("email") if isinstance(state.get("user_data"), dict) else None)
    
    # Fallback to SESSION_EMAIL only if no user_email in state (backward compatibility)
    if not user_email:
        user_email = SESSION_EMAIL
        print(f"Warning: No user_email in state for email notification, using fallback SESSION_EMAIL")
    
    # Send the information change email
    email_sent = send_information_change_email(changes_made, recipient_email=user_email)
    
    
    if email_sent:
        # Notify the user the email was sent to their email
        response = f"I've sent you a security notification email about the changes made to your account. Please check your email at {user_email}."
    else:
        # If the email fails, notify the user of changes made
        changes_text = ", ".join(changes_made)
        response = f"Your {changes_text} has been updated successfully. I attempted to send you a security notification email, but there was an issue with the email service."
    
    return {
        "messages": [{"role": "assistant", "content": f"Email Agent: {response}"}],
        "needs_email_notification": False  # Reset the flag
    }

"""Handle order receipt requests with context memory and multi-turn conversation.
User either provides it all at once or we ask follow-up questions. This function uses
an LLM to parse the request and extract relevant variables. It is called conditionally
to reduce number of API calls and latency"""
async def handle_order_receipt_request(state: State):
    last_message = state["messages"][-1]
    user_message = last_message.get("content") if isinstance(last_message, dict) else last_message.content

    conversation_context = state.get("conversation_context", {})
    
    # Parse the order receipt request
    parser_llm = llm.with_structured_output(OrderReceiptParser)
    
    parsing_result = parser_llm.invoke([
        {
            "role": "system",
            "content": """You are an expert at parsing order receipt requests. 
            Extract the following information from the user's message:
            - user_email: The email address they provide to identify their orders
            - order_number: Specific order number mentioned (ex: ORD-2024-001, ORDER-123)
            - chronological_request: Words like 'last', 'most recent', 'latest', 'previous', 'first'
            - product_name: Product name mentioned to find orders containing that product
            - time_reference: Time references like 'yesterday', 'last week', 'this month'
            
            Example:
            "I need a receipt for order ORD-2024-001" -> order_number: "ORD-2024-001"
            "Can you send me my last order receipt?" -> chronological_request: "last"
            "I want the receipt for my laptop order" -> product_name: "laptop"
            "Send receipt for my order from last week" -> time_reference: "last week"
            """
        },
        {"role": "user", "content": user_message}
    ])
    
    print(f"Parsed receipt request: {parsing_result}")
    
    # Get user email from state (logged-in user context)
    user_email = state.get("user_email") or (state.get("user_data", {}).get("email") if isinstance(state.get("user_data"), dict) else None)
    
    # Fallback to SESSION_EMAIL only if no user_email in state (backward compatibility)
    if not user_email:
        user_email = SESSION_EMAIL
        print(f"Warning: No user_email in state for order receipt, using fallback SESSION_EMAIL")
    
    # Use user email to process the request
    return await process_receipt_request(parsing_result.model_dump(), user_email, conversation_context, state)

async def process_receipt_request(parsed_request: dict, user_email: str, conversation_context: dict, state: State):
    order_data = None
    
    if parsed_request.get("order_number") and parsed_request["order_number"] not in ['<UNKNOWN>', None, '']:
        # Look up by order number
        order_data = lookup_order_by_number(parsed_request["order_number"])
        if not order_data:
            response = f"I couldn't find an order with number '{parsed_request['order_number']}'. Please check the order number and try again."
        # If the order number exists but email doesn't match, notify user
        elif order_data["email"].lower() != user_email.lower():
            response = f"The order '{parsed_request['order_number']}' doesn't belong to the email address {user_email}. Please verify your information."
            order_data = None
    
    elif parsed_request.get("chronological_request"):
        # Look up most recent order
        orders = lookup_orders_by_email(user_email, limit=1)
        if orders:
            order_data = orders[0]
        else:
            response = f"I couldn't find any orders for {user_email}. Please check your email address."
    
    elif parsed_request.get("product_name") and parsed_request["product_name"] not in ['<UNKNOWN>', None, '']:
        # Look up by product name
        orders = lookup_orders_by_product_name(parsed_request["product_name"], user_email)
        if orders:
            order_data = orders[0]  # Get the most recent order with that product
        else:
            response = f"I couldn't find any orders containing '{parsed_request['product_name']}' for {user_email}."
    
    else:
        # Default to most recent order
        orders = lookup_orders_by_email(user_email, limit=1)
        if orders:
            order_data = orders[0]
        else:
            response = f"I couldn't find any orders for {user_email}. Please check your email address."
    
    # If we found an order, send the receipt (non-blocking to avoid Kafka timeout)
    if order_data:
        # Run email sending in executor to avoid blocking Kafka consumer
        loop = asyncio.get_event_loop()
        try:
            email_sent = await loop.run_in_executor(
                None, 
                send_order_receipt_email, 
                order_data, 
                user_email
            )
        except Exception as e:
            print(f"⚠️ Error in email executor: {e}")
            email_sent = False
        
        if email_sent:
            response = f"I've sent the receipt for order {order_data['order_number']} to {user_email}."
        else:
            # Show the receipt upon email failure
            receipt_content = format_order_receipt(order_data)
            response = f"Here's your order receipt:\n\n{receipt_content}\n\nNote: I had trouble sending the email, but here's your receipt information above."
    
    # Clean up conversation context
    updated_context = conversation_context.copy()
    updated_context["user_email"] = user_email
    
    return {
        "messages": [{"role": "assistant", "content": f"Email Agent: {response}"}],
        "conversation_context": updated_context
    }


#The policy agent uses retrieval augmented generation to recall policy information from policy.txt
import uuid

async def policy_agent(state: State):
    last_message = state["messages"][-1]
    user_question = last_message.get("content") if isinstance(last_message, dict) else last_message.content

    session_id = state.get("session_id", str(uuid.uuid4()))
    conversation_id = state.get("conversation_id", str(uuid.uuid4()))

    try:
        # Use the existing query_rag function to get policy-specific information
        # (wrap in asyncio.to_thread if it's blocking)
        formatted_response, policy_response = await asyncio.to_thread(query_rag, user_question)
        
        # Check if RAG returned an error
        if policy_response is None or policy_response.startswith("I couldn't find"):
            raise Exception("RAG query failed or returned no results")

        # Create messages with the policy context
        messages = [
            {
                "role": "system",
                "content": f"""You are a policy agent. Your job is to help customers with questions that appear to be related to company policy,
                such as how long deliveries usually take, how returns are handled, and how the company runs things. 
                
                CRITICAL: You MUST use ONLY the following policy information to answer the customer's question. 
                Do NOT make up, hallucinate, or guess any information. 
                
                POLICY INFORMATION FROM DATABASE:
                {policy_response}
                
                STRICT INSTRUCTIONS:
                Read the policy information above CAREFULLY
                Do NOT invent or assume timeframes - use ONLY what is in the policy information above
                Be direct, specific, and accurate
                If the policy information doesn't contain the answer, say you don't have that information and suggest contacting customer support
                Do not directly mention the inner workings of this system, instead focus on the user's requests.
                """
            },
            {"role": "user", "content": user_question},
        ]

        reply = await llm.ainvoke(messages)
        print(f"Policy agent response: {reply.content}")

        return {"messages": [{"role": "assistant", "content": f"Policy Agent: {reply.content}"}]}

    except Exception as e:
        # Fallback to general policy response if RAG fails
        messages = [
            {
                "role": "system",
                "content": """You are a policy agent. Your job is to help customers with questions that appear to be related to company policy,
                such as how long deliveries usually take, how returns are handled, and how the company runs things. You are to refer to the written policy
                and inform the user how to contact the store when information can't be retrieved for one reason or another.
                Do not directly mention the inner workings of this system, instead focus on the user's requests."""
            },
            {"role": "user", "content": user_question},
        ]

        reply = await llm.ainvoke(messages)
        print(f"Policy agent response (fallback): {reply.content}")

        return {"messages": [{"role": "assistant", "content": f"Policy Agent: {reply.content}"}]}

#This is the default agent that is routed to if the request has nothing to do with orders or emails
import uuid, time, asyncio

async def message_agent(state: State):
    last_message = state["messages"][-1]
    message_type = state.get("message_type", "Message")

    session_id = state.get("session_id", str(uuid.uuid4()))
    conversation_id = state.get("conversation_id", str(uuid.uuid4()))

    # Branch 1: Handle Change Information requests
    if message_type == "Change Information":
        return handle_change_information(state)  # ✅ No await

    # Branch 2: Handle Order Receipt requests
    elif message_type == "Order Receipt":
        return handle_order_receipt(state)  # ✅ No await
    
    # Default branch continues...

    # Default branch: freeform message handling
    user_message = last_message.get("content") if isinstance(last_message, dict) else last_message.content
    messages = [
        {
            "role": "system",
            "content": """You are a message agent. Your job is to provide structured responses and help the customer the best that you can.
            Refer the relevant information from the user's request to the orchestrator agent in a structured manner so that customers can
            be helped with their specific use case. Do not directly mention the inner workings of this system, instead focus on the user's requests."""
        },
        {"role": "user", "content": user_message},
    ]

    reply = await llm.ainvoke(messages)
    print(f"Message agent response: {reply.content}")

    return {"messages": [{"role": "assistant", "content": f"Message Agent: {reply.content}"}]}


#Get database connection using environment variables. At present, database connection is used for change user info and receipt emails
def get_database_connection():
    try:
        # Try DATABASE_URL first (for Docker)
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            conn = psycopg2.connect(database_url)
        else:
            # Fallback to individual variables
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', '127.0.0.1'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'AgenticAIStackDB'),
                user=os.getenv('DB_USER', 'AgenticAIStackDB'),
                password=os.getenv('DB_PASSWORD')
            )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# User lookup functions moved to auth.py and api_endpoints.py modules

# Update user information in database
def update_user_information(user_id: str, updates: dict):
    conn = get_database_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Build dynamic update query
            update_fields = []
            values = []
            
            # fields are provided in updates dictionary parameter
            for field, value in updates.items():
                if value is not None:
                    update_fields.append(f"{field} = %s")
                    values.append(value)
            
            if not update_fields:
                return False
            
            # Add user_id to values for WHERE clause
            values.append(user_id)
            
            query = f"""
            UPDATE users 
            SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """
            
            cursor.execute(query, values)
            conn.commit()
            return cursor.rowcount > 0
            
    except Exception as e:
        print(f"Error updating user information: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

# Database lookup functions moved to auth.py and api_endpoints.py modules

# Look up order by product name, optionally filtered by user email
def lookup_orders_by_product_name(product_name: str, user_email: Optional[str] = None):
    conn = get_database_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            base_query = """
            SELECT DISTINCT o.id, o.order_number, o.status, o.total_amount, o.currency, o.created_at,
                   u.email, u.first_name, u.last_name,
                   array_agg(
                       json_build_object(
                           'product_name', p2.name,
                           'quantity', oi2.quantity,
                           'unit_price', oi2.unit_price,
                           'total_price', oi2.quantity * oi2.unit_price
                       )
                   ) as items
            FROM orders o
            JOIN users u ON o.user_id = u.id
            JOIN order_items oi ON o.id = oi.order_id
            JOIN products p ON oi.product_id = p.id
            JOIN order_items oi2 ON o.id = oi2.order_id
            JOIN products p2 ON oi2.product_id = p2.id
            WHERE LOWER(p.name) LIKE LOWER(%s)
            """
            
            params = [f"%{product_name}%"]
            
            if user_email:
                base_query += " AND LOWER(u.email) = LOWER(%s)"
                params.append(user_email)
            
            base_query += """
            GROUP BY o.id, o.order_number, o.status, o.total_amount, o.currency, o.created_at,
                     u.email, u.first_name, u.last_name
            ORDER BY o.created_at DESC
            """
            
            cursor.execute(base_query, params)
            orders = cursor.fetchall()
            return [dict(order) for order in orders]
    except Exception as e:
        print(f"Error looking up orders by product name: {e}")
        return []
    finally:
        conn.close()

# Send information change notification email using SendGrid SMTP relay. In production recipient email will be session email
def send_information_change_email(changes_made: list, recipient_email: str = "jeremyyhop@gmail.com"):
    if not SMTP_AVAILABLE:
        print("SMTP not available. Email would have been sent to:", recipient_email)
        print(f"Information change notification: {', '.join(changes_made)}")
        return False
    
    # SendGrid SMTP configuration
    smtp_server = "smtp.sendgrid.net"
    smtp_port = 587  # Using TLS port
    smtp_username = "apikey"
    smtp_password = os.getenv('SENDGRID_API_KEY')  # Required - must be set in .env file
    
    try:
        # Format the changes
        changes_text = ", ".join(changes_made)
        
        # Create the email message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "Account Information Changed - Agentic AI Stack"
        msg['From'] = 'jeremyyhop@gmail.com'
        msg['To'] = recipient_email
        
        # Create HTML content
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                    Account Information Updated
                </h2>
                
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0;">
                    <p style="margin: 0; font-size: 16px;">
                        <strong>Your {changes_text} has been successfully updated.</strong>
                    </p>
                </div>
                
                <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <p style="margin: 0; color: #856404;">
                        <strong>Security Notice:</strong> If you did not make this change, contact us at agenticaistack@gmail.com.
                    </p>
                </div>
                
                <div style="margin: 30px 0;">
                    <p><strong>Need Help?</strong></p>
                    <p>If this change was not made by you, please contact our support team immediately:</p>
                    <p style="background-color: #e3f2fd; padding: 10px; border-radius: 3px;">
                        📧 <a href="mailto:agenticaistack@gmail.com" style="color: #1976d2;">agenticaistack@gmail.com</a>
                    </p>
                </div>
                
                <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666;">
                    <p>This is an automated message from your Agentic AI Stack system.</p>
                    <p>Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create plain text version
        text_content = f"""
Account Information Updated

Your {changes_text} has been successfully updated.

SECURITY NOTICE: If you did not make this change, please contact us immediately.

Need Help?
If this change was not made by you, please contact our support team:
Email: agenticaistack@gmail.com

This is an automated message from your Agentic AI Stack system.
Please do not reply to this email.
        """
        
        # Attach parts
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(html_content, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Send the email using SMTP
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Enable TLS encryption
        server.login(smtp_username, smtp_password)
        server.send_message(msg)
        server.quit()
        
        print(f"Information change notification sent successfully via SMTP to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"Error sending information change email via SMTP: {e}")
        print("Email would have been sent to:", recipient_email)
        print(f"Information change notification: {', '.join(changes_made)}")
        return False

# Send order receipt email using SendGrid SMTP relay. In production recipient email will be session email
def send_order_receipt_email(order_data: dict, recipient_email: str = "jeremyyhop@gmail.com"):
    """Send order receipt email using SendGrid SMTP relay with timeout protection"""
    if not SMTP_AVAILABLE:
        print("SMTP not available. Email would have been sent to:", recipient_email)
        print("Order receipt content:", format_order_receipt(order_data))
        return False
    
    # SendGrid SMTP configuration
    smtp_server = "smtp.sendgrid.net"
    smtp_port = 587  # Using TLS port
    smtp_username = "apikey"
    smtp_password = os.getenv('SENDGRID_API_KEY')  # Required - must be set in .env file
    
    # Check if API key is configured
    if not smtp_password:
        print("⚠️ SENDGRID_API_KEY not configured. Cannot send email.")
        print("Email would have been sent to:", recipient_email)
        print("Order receipt content:", format_order_receipt(order_data))
        return False
    
    try:
        # Format the order receipt
        receipt_content = format_order_receipt(order_data)
        
        # Create the email message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"Order Receipt - {order_data['order_number']}"
        msg['From'] = 'jeremyyhop@gmail.com'
        msg['To'] = recipient_email
        
        # Create HTML content
        html_content = f"""
        <html>
        <body>
            <h2>Order Receipt</h2>
            <pre style="font-family: Arial, sans-serif; white-space: pre-wrap;">{receipt_content}</pre>
            <br><br>
            <p>If you have any questions, please contact our customer support.</p>
        </body>
        </html>
        """
        
        # Create plain text version
        text_content = f"Order Receipt\n\n{receipt_content}\n\nThank you for your business!\nIf you have any questions, please contact our customer support."
        
        # Attach parts
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(html_content, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Send the email using SMTP with timeout (5 seconds per operation)
        import socket
        socket.setdefaulttimeout(5)  # Set global socket timeout
        
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=5)
        server.starttls()  # Enable TLS encryption
        server.login(smtp_username, smtp_password)
        server.send_message(msg)
        server.quit()
        
        print(f"✓ Email sent successfully via SMTP to {recipient_email}")
        return True
        
    except socket.timeout:
        print(f"⚠️ SMTP connection timeout. Email not sent to {recipient_email}")
        print("Order receipt content:", format_order_receipt(order_data))
        return False
    except smtplib.SMTPAuthenticationError as e:
        print(f"⚠️ SMTP authentication failed (invalid API key): {e}")
        print("Email would have been sent to:", recipient_email)
        print("Order receipt content:", format_order_receipt(order_data))
        return False
    except Exception as e:
        print(f"⚠️ Error sending email via SMTP: {e}")
        print("Email would have been sent to:", recipient_email)
        print("Order receipt content:", format_order_receipt(order_data))
        return False
    finally:
        # Reset socket timeout
        socket.setdefaulttimeout(None)

def format_order_receipt(order_data: dict) -> str:
    """Format order data into a readable receipt"""
    if not order_data:
        return "No order data available."
    
    receipt = f"""
ORDER RECEIPT
=============

Order Number: {order_data['order_number']}
Customer: {order_data['first_name']} {order_data['last_name']}
Email: {order_data['email']}
Order Date: {order_data['created_at'].strftime('%Y-%m-%d %H:%M:%S') if order_data['created_at'] else 'N/A'}
Status: {order_data['status'].title()}

ITEMS:
------"""
    
    total = 0
    if order_data.get('items'):
        for item in order_data['items']:
            item_total = float(item['total_price'])
            total += item_total
            receipt += f"""
• {item['product_name']}
  Quantity: {item['quantity']}
  Unit Price: ${item['unit_price']:.2f}
  Total: ${item_total:.2f}"""
    
    receipt += f"""

------
TOTAL: ${order_data['total_amount']:.2f} {order_data['currency']}
======
"""
    
    return receipt

# Handle change information requests with multi-turn conversation support
def handle_change_information(state: State):
    """Handle change information requests with multi-turn conversation support"""
    last_message = state["messages"][-1]
    user_message = last_message.get("content") if isinstance(last_message, dict) else last_message.content
    conversation_context = state.get("conversation_context", {})
    
    # User email will be retrieved from state (logged-in user context)
    
    # Use LLM to parse the change information request
    parser_llm = llm.with_structured_output(UserInformationParser)
    
    parsing_result = parser_llm.invoke([
        {
            "role": "system",
            "content": """You are an expert at parsing user information change requests. 
            Extract the following information from the user's message:
            - current_user_email: The email address they provide to identify themselves
            - new_user_email: If they want to change their email to a new one
            - current_firstname: Their current first name if mentioned
            - new_firstname: The new first name they want to change to
            - current_lastname: Their current last name if mentioned  
            - new_lastname: The new last name they want to change to
            - current_phone: Their current phone number if mentioned
            - new_phone: The new phone number they want to change to
            
            Examples:
            "I want to change my name from Joe to John. My email is john.doe@example.com"
            → current_firstname: "Joe", new_firstname: "John", current_user_email: "john.doe@example.com"
            
            "Please update my phone number from 555-1234 to 555-5678. My email is jane@example.com"
            → current_phone: "555-1234", new_phone: "555-5678", current_user_email: "jane@example.com"
            
            "I want to change my first name to Joe"
            → new_firstname: "Joe"
            
            Only extract information that is clearly stated in the message."""
        },
        {"role": "user", "content": user_message}
    ])
    
    print(f"Parsed change information: {parsing_result}")
    
    # Get user email from state (logged-in user context)
    user_email = state.get("user_email") or (state.get("user_data", {}).get("email") if isinstance(state.get("user_data"), dict) else None)
    
    # Fallback to SESSION_EMAIL only if no user_email in state (backward compatibility)
    if not user_email:
        user_email = SESSION_EMAIL
        print(f"Warning: No user_email in state, using fallback SESSION_EMAIL")
    
    # Use user email to look up the user
    user_data = lookup_user_by_email(user_email)
    
    # Prepare updates based on parsed information
    updates = {}
    changes_made = []
    
    if parsing_result.new_firstname:
        updates['first_name'] = parsing_result.new_firstname
        changes_made.append(f"first name to '{parsing_result.new_firstname}'")
    
    if parsing_result.new_lastname:
        updates['last_name'] = parsing_result.new_lastname
        changes_made.append(f"last name to '{parsing_result.new_lastname}'")
    
    if parsing_result.new_phone:
        updates['phone'] = parsing_result.new_phone
        changes_made.append(f"phone number to '{parsing_result.new_phone}'")
    
    if parsing_result.new_user_email:
        updates['email'] = parsing_result.new_user_email
        changes_made.append(f"email address to '{parsing_result.new_user_email}'")
    
    # Check if user exists
    if not user_data:
        response = f"I couldn't find your account with email {user_email}. Please contact customer support for assistance."
        return {
            "messages": [{"role": "assistant", "content": f"Message Agent: {response}"}],
            "parsed_changes": parsing_result.model_dump(),
            "user_data": {}
        }
    
    # If no updates identified
    if not updates:
        response = "I understand you want to change your information, but I couldn't identify what specific changes you'd like to make. Could you be more specific about what information you'd like to update?"
        return {
            "messages": [{"role": "assistant", "content": f"Message Agent: {response}"}],
            "parsed_changes": parsing_result.model_dump(),
            "user_data": {}
        }
    
    # If we have both updates and user data, proceed with update
    if user_data:
        # perform update by passing user id and dictionary of updates to be applied
        success = update_user_information(user_data['id'], updates)
        
        if success:
            changes_text = ", ".join(changes_made)
            response = f"Great! I've successfully updated your {changes_text}. Your account information has been updated in our system."
            
            # Update conversation context
            updated_context = conversation_context.copy()
            updated_context["user_identified"] = True
            updated_context["user_id"] = user_data['id']
            updated_context["user_email"] = user_email
            
            # Set flags for email notification
            information_changed = True
            needs_email_notification = True
            
        else:
            response = "I apologize, but there was an issue updating your information. Please try again later or contact customer support for assistance."
            updated_context = conversation_context
        
        return {
            "messages": [{"role": "assistant", "content": f"Message Agent: {response}"}],
            "conversation_context": updated_context,
            "parsed_changes": parsing_result.model_dump(),
            "user_data": user_data,
            "information_changed": information_changed if 'information_changed' in locals() else False,
            "changes_made": changes_made,
            "needs_email_notification": needs_email_notification if 'needs_email_notification' in locals() else False
        }
    else:
        response = "I couldn't find your account with that email address. Please double-check your email and try again, or contact customer support for assistance."
        return {
            "messages": [{"role": "assistant", "content": f"Message Agent: {response}"}],
            "parsed_changes": parsing_result.model_dump(),
            "user_data": {}
        }

# Handle order receipt requests
def handle_order_receipt(state: State):
    """Handle order receipt requests"""
    last_message = state["messages"][-1]
    
    response = "I can help you with order receipts. Please provide your order number or email address, and I'll retrieve your receipt information for you."
    
    return {
        "messages": [{"role": "assistant", "content": f"Message Agent: {response}"}]
    }



# Orchestrator agent to manage agent responses and trigger followup actions
def orchestrator_agent(state: State):
    """Enhanced orchestrator that manages agent responses and triggers follow-up actions"""
    last_message = state["messages"][-1]

    
    # Check if information was changed and we need to send a notification email
    if state.get("information_changed") and state.get("needs_email_notification"):
        print("Orchestrator detected information change - routing to email agent for notification")
        # Route to email agent for notification, but first pass through the current response
        if last_message.content.startswith(("Message Agent:", "Order Agent:", "Email Agent:", "Policy Agent:")):
            agent_response = last_message.content
            if ": " in agent_response:
                clean_response = agent_response.split(": ", 1)[1]
            else:
                clean_response = agent_response
            print(f"Orchestrator passing through: {clean_response}")
            
            # Return state that will trigger email notification
            return {
                "messages": [{"role": "assistant", "content": clean_response}],
                "needs_follow_up_email": True  # Flag for the graph to route to email agent
            }
    
    # Check if the last message is from an agent
    if last_message.content.startswith(("Message Agent:", "Order Agent:", "Email Agent:", "Policy Agent:")):
        # Extract the agent response and pass it through cleanly
        agent_response = last_message.content
        
        # Remove the "Agent:" prefix for cleaner user experience
        if ": " in agent_response:
            clean_response = agent_response.split(": ", 1)[1]
        else:
            clean_response = agent_response
        
        print(f"Orchestrator passing through: {clean_response}")
        return {"messages": [{"role": "assistant", "content": clean_response}]}
    else:
        # Fallback for unexpected cases
        print(f"Orchestrator fallback response")
        return {"messages": [{"role": "assistant", "content": "I'm here to help you. How can I assist you today?"}]}
    


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:52057",
        "*",  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers for chat history and admin dashboard
from .chathistory import router as chathistory_router
from .admindashboard import router as admindashboard_router

app.include_router(chathistory_router)
app.include_router(admindashboard_router)

class IngressMessage(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    query_text: str
    correlation_id: Optional[str] = None
    event_timestamp: Optional[int] = None
    conversation_id: Optional[str] = None

# Import modular components
from .auth import (
    authenticate_user,
    ensure_default_users,
    lookup_user_by_email,
    LoginRequest,
    LoginResponse,
)
from .api_endpoints import (
    UserProfile, UserAddress, OrderItem, Order, 
    UserProfileResponse, OrdersResponse, HealthResponse,
    get_user_profile, get_user_orders, get_order_details,
    lookup_orders_by_email, lookup_order_by_number,
    create_order, CreateOrderRequest
)
from database import (
    create_conversation,
    get_conversation,
    update_conversation_context,
    save_query_to_conversation
)

priority_consumer: PriorityConsumer | None = None


# Create a single producer instance at startup
producer: AIOKafkaProducer | None = None


@app.post("/publish/ingress")
async def publish_to_ingress(message: IngressMessage):
    """Publish message from frontend to system.ingress topic"""
    if not message.event_timestamp:
        message.event_timestamp = int(time.time() * 1000)
    if not message.correlation_id:
        message.correlation_id = f"corr-{uuid.uuid4()}"
    
    # Look up user_id from email if not provided
    user_id = message.user_id
    if not user_id and message.user_email:
        user_data = lookup_user_by_email(message.user_email)
        if user_data:
            user_id = str(user_data['id'])
    
    # Get or create conversation for this session
    if message.conversation_id:
        conversation_id = message.conversation_id
    else:
        conversation = get_conversation(message.session_id)
        if not conversation:
            # Create new conversation
            conversation_id = create_conversation(
                session_id=message.session_id,
                user_email=message.user_email,
                user_id=user_id
            )
            if not conversation_id:
                conversation_id = str(uuid.uuid4())  # Fallback if DB fails
        else:
            conversation_id = conversation['id']
            # Update conversation with user info if not already set
            if user_id and not conversation.get('user_id'):
                update_conversation_context(conversation_id, conversation.get('context', {}), 
                                          user_id=user_id, user_email=message.user_email)
    
    kafka_message = {
        "session_id": message.session_id,
        "user_id": user_id,
        "user_email": message.user_email,
        "query_text": message.query_text,
        "correlation_id": message.correlation_id,
        "conversation_id": conversation_id,
        "event_timestamp": message.event_timestamp,
        "event_type": "CUSTOMER_QUERY",
        "status": "pending"
    }
    
    try:
        await producer.send_and_wait(
            INGRESS_TOPIC,
            value=kafka_message,
            key=message.session_id
        )
        
        print(f"→ Published ingress for {message.session_id}, conversation: {conversation_id}")
        return {
            "success": True,
            "message": f"Published to {INGRESS_TOPIC}",
            "conversation_id": conversation_id
        }
    except Exception as e:
        print(f"✗ Publish error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/agent-responses/{session_id}")
async def websocket_agent_responses(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"✓ WebSocket connected: {session_id}")
    
    consumer = AIOKafkaConsumer(
        RESPONSE_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=f"ws-{session_id}",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",  # ✅ Change back to latest
        enable_auto_commit=True,
        session_timeout_ms=30000,  # Increased to 30s to handle email operations
        request_timeout_ms=30000
    )
    
    await consumer.start()
    
    # ✅ Add timestamp filter here too
    try:
        async for message in consumer:
            event = message.value
            
            # Only send messages from the last 10 seconds (prevents old duplicates)
            event_time = event.get("timestamp", 0)
            if time.time() * 1000 - event_time > 10000:
                continue
                
            if event.get("session_id") == session_id:
                await websocket.send_json(event)
                print(f"→ Sent to frontend: {session_id} - {event.get('agent_type', 'UNKNOWN')}")
    except Exception as e:
        print(f"✗ WebSocket consumer error: {e}")
    finally:
        await consumer.stop()
        await websocket.close()
        print(f"✗ WebSocket disconnected: {session_id}")

# Authentication Endpoint
@app.post("/api/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user and return user profile"""
    try:
        # Authenticate user
        user_data = authenticate_user(request.email, request.password)
        
        if not user_data:
            return LoginResponse(
                success=False,
                message="Invalid email or password"
            )
        
        # Format user profile
        user_profile = UserProfile(
            id=str(user_data['id']),
            email=user_data['email'],
            first_name=user_data.get('first_name'),
            last_name=user_data.get('last_name'),
            phone=user_data.get('phone'),
            is_admin=user_data.get('is_admin', False),
            created_at=user_data['created_at'].isoformat(),
            updated_at=user_data['updated_at'].isoformat()
        )
        
        return LoginResponse(
            success=True,
            message="Login successful",
            user=user_profile.dict()
        )
        
    except Exception as e:
        print(f"Error in login endpoint: {e}")
        return LoginResponse(
            success=False,
            message="Internal server error"
        )

# Health Check Endpoint
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check system health status"""
    try:
        # Check database connection
        conn = get_database_connection()
        db_status = "healthy" if conn else "unhealthy"
        if conn:
            conn.close()
        
        # Check Kafka producer
        kafka_status = "healthy" if producer else "unhealthy"
        
        return HealthResponse(
            status="healthy" if db_status == "healthy" and kafka_status == "healthy" else "degraded",
            timestamp=datetime.utcnow().isoformat(),
            database=db_status,
            kafka=kafka_status
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            database="error",
            kafka="error"
        )

# User Profile Endpoint
@app.get("/api/user/profile", response_model=UserProfileResponse)
async def get_user_profile_endpoint(user_email: str):
    """Get user profile information including addresses and order stats"""
    return await get_user_profile(user_email)

# User Orders Endpoint
@app.get("/api/user/orders", response_model=OrdersResponse)
async def get_user_orders_endpoint(user_email: str, page: int = 1, limit: int = 10):
    """Get user's orders with pagination"""
    return await get_user_orders(user_email, page, limit)

# Order Details Endpoint
@app.get("/api/orders/{order_number}", response_model=Order)
async def get_order_details_endpoint(order_number: str):
    """Get detailed information about a specific order"""
    return await get_order_details(order_number)

# Create Order Endpoint
@app.post("/api/orders", response_model=Order)
async def create_order_endpoint(request: CreateOrderRequest):
    """Create a new order for a user"""
    return await create_order(request)

# Get Products Endpoint
@app.get("/api/products")
async def get_products_endpoint():
    """Get list of available products"""
    from database.connection import get_database_connection
    from psycopg2.extras import RealDictCursor
    
    conn = get_database_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id, name, description, price, stock_quantity
                FROM products
                ORDER BY name
            """)
            products = cursor.fetchall()
            return {"products": [dict(product) for product in products]}
    except Exception as e:
        print(f"Error fetching products: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()


async def orchestrator_consumer():
    """
    Enhanced orchestrator that:
    1. Consumes from system.ingress
    2. Classifies and routes to priority topics
    3. Does NOT call agents directly
    """
    consumer = AIOKafkaConsumer(
        INGRESS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="orchestrator-group-v2",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        session_timeout_ms=30000,  # Increased to 30s to handle email operations
        request_timeout_ms=30000
    )
    
    await consumer.start()
    print(f"🔵 Orchestrator consumer started on {INGRESS_TOPIC}")

# Wait for partition assignment
    while not consumer.assignment():
        print("⏳ Waiting for partition assignment...")
        await asyncio.sleep(0.5)

    print(f"✅ Orchestrator assigned partitions: {consumer.assignment()}")

    verbose_polling_log("🔍 Orchestrator ENTERING async for loop")
    try:
        async for message in consumer:
            print(f"⚡ LOOP ITERATION - got message from partition {message.partition}")  # ADD THIS
            verbose_polling_log("🔍 ORCHESTRATOR GOT MESSAGE!")  # Add this first line
            event = message.value
            session_id = event.get("session_id", "unknown")
            print(f"→ Orchestrator received event for {session_id}")
            
            try:
                # Get user information from event
                user_email = event.get("user_email")
                user_id = event.get("user_id")
                user_data = {}
                
                # Look up user if email is provided
                if user_email:
                    user_lookup = lookup_user_by_email(user_email)
                    if user_lookup:
                        user_data = {
                            "id": str(user_lookup.get("id", "")),
                            "email": user_lookup.get("email", ""),
                            "first_name": user_lookup.get("first_name"),
                            "last_name": user_lookup.get("last_name"),
                            "phone": user_lookup.get("phone")
                        }
                
                # Build state from event
                state = {
                    "messages": [{"role": "user", "content": event.get("query_text", "")}],
                    "message_type": None,
                    "order_data": {},
                    "user_data": user_data,
                    "user_email": user_email,  # Add user_email to state for easy access
                    "user_id": user_id,  # Add user_id to state for easy access
                    "parsed_changes": {},
                    "conversation_id": event.get("conversation_id", str(uuid.uuid4())),
                    "session_id": session_id,
                    "conversation_context": {},
                    "message_order": 0,
                    "information_changed": False,
                    "changes_made": [],
                    "needs_email_notification": False,
                    "needs_follow_up_email": False,
                    "next": None
                }
                
                # Classify message type (your existing classify_message function)
                classification = classify_message(state)
                state.update(classification)
                
                message_type = state.get("message_type", "Message")
                
                # Classify priority
                priority = classify_priority(
                    event.get("query_text", ""), 
                    message_type, 
                    event
                )
                
                # Determine target topic based on message type and priority
                topic_key = message_type if message_type in PRIORITY_TOPICS else "Message"
                priority_topics = PRIORITY_TOPICS.get(topic_key, PRIORITY_TOPICS["Message"])
                target_topic = priority_topics[priority - 1]  # priority 1->p1, 2->p2, 3->p3
                
                # Create routing event
                routing_event = {
                    "session_id": session_id,
                    "conversation_id": state["conversation_id"],
                    "message_type": message_type,
                    "priority": priority,
                    "query_text": event.get("query_text", ""),
                    "state": state,
                    "correlation_id": event.get("correlation_id"),
                    "event_timestamp": event.get("event_timestamp", int(time.time() * 1000))
                }
                
                # Publish to priority topic
                await producer.send_and_wait(
                    target_topic,
                    value=routing_event,
                    key=session_id
                )
                
                print(f"→ Routed {session_id} to {target_topic} (type: {message_type}, priority: P{priority})")
                
            except Exception as e:
                print(f"✗ Orchestrator error for {session_id}: {e}")
                
    finally:
        await consumer.stop()
        print("🔴 Orchestrator consumer stopped")


#Initialize a state graph, then we add nodes, naming them and linking their respective agents.
graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("order", order_agent)
graph_builder.add_node("email", email_agent)
graph_builder.add_node("policy", policy_agent)
graph_builder.add_node("message", message_agent)
graph_builder.add_node("orchestrator", orchestrator_agent)

#We also add edges from start to classifier; once classified, goes to router
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {
        "order": "order", 
        "email": "email", 
        "policy": "policy", 
        "message": "message"
    }
)
graph_builder.add_edge("order", "orchestrator")
graph_builder.add_edge("email", "orchestrator")
graph_builder.add_edge("policy", "orchestrator")
graph_builder.add_edge("message", "orchestrator")

# Add conditional edge from orchestrator to handle follow-up emails
graph_builder.add_conditional_edges(
    "orchestrator",
    lambda state: "email" if state.get("needs_follow_up_email") else "end",
    {
        "email": "email",
        "end": END
    }
)
graph = graph_builder.compile()

"""def run_chatbot():
    #Enhanced chatbot with conversation memory. Create session ID and conversation tracking
    session_id = str(uuid.uuid4())
    print(f"Starting conversation w/ session ID: {session_id}")
    
    # Initialize conversation in database
    conversation_id = create_conversation(session_id)
    if not conversation_id:
        print("Warning: Running in memory-only mode. Agent may not remember past interactions.")
        conversation_id = str(uuid.uuid4())
    
    # Initialize our first state with conversation tracking
    state = {
        "messages": [], 
        "message_type": None, 
        "order_data": {}, 
        "user_data": {}, 
        "parsed_changes": {},
        "conversation_id": conversation_id,
        "session_id": session_id,
        "conversation_context": {},
        "message_order": 0,
        "information_changed": False,
        "changes_made": [],
        "needs_email_notification": False
    }
    
    print("Enhanced chatbot with conversation memory.")
    print("=" * 50)
    print("I can help you with orders, policies, information changes, and general questions.")
    print("Press Ctrl + C to exit.")
    print("-" * 50)

    while True:
        user_input = input("\nUser: ").strip()
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Thanks for chatting! Have a great day!")
            break
            
        if not user_input:
            continue

        # Increment message order
        state["message_order"] = state.get("message_order", 0) + 1
        
        # Add user message to state
        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        try:
            # Process through the graph
            result = graph.invoke(state)

            # Update state with result
            state.update(result)

            # Display the response and save to conversation
            if state.get("messages") and len(state["messages"]) > 0:
                last_message = state["messages"][-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    response_content = last_message["content"]
                    print(f"Assistant: {response_content}")
                    
                    # Save query and response to database (if available)
                    try:
                        save_query_to_conversation(
                            conversation_id, 
                            user_input, 
                            "orchestrator",  # Final response always comes through orchestrator
                            response_content, 
                            state["message_order"],
                            state.get("conversation_context", {}).get("user_id")
                        )
                        
                        # Update conversation context in database
                        if state.get("conversation_context"):
                            update_conversation_context(
                                conversation_id,
                                state["conversation_context"],
                                state.get("conversation_context", {}).get("user_id"),
                                state.get("conversation_context", {}).get("user_email")
                            )
                    except Exception as e:
                        print(f"Could not save to database: {e}")
                        
                else:
                    print(f"Assistant: {last_message}")
                    
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again or contact support // agenticaistack@gmail.com.")

    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


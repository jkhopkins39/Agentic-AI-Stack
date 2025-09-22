import os # Importing os module for operating system functionalities
import shutil # Importing shutil module for high-level file operations
from typing import Annotated, Literal # Different data types we need
from dotenv import load_dotenv # Importing dotenv to get API key from .env file
from pydantic import BaseModel, Field # Used for validation and structuring message classification
from typing_extensions import TypedDict # State typed dict contains typed keys messages, message_type, order_data
from langchain.document_loaders.pdf import PyPDFDirectoryLoader # Importing PDF loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter # Importing text splitter from Langchain
from langchain.embeddings import OpenAIEmbeddings # Importing OpenAI embeddings from Langchain
from langchain.schema import Document # Importing Document schema from Langchain
from langchain_core.prompts import ChatPromptTemplate # Importing ChatPromptTemplate for prompt formatting
from langchain.vectorstores.chroma import Chroma # Importing Chroma vector store from Langchain
from langchain.chat_models import ChatOpenAI # Import OpenAI LLM
from langgraph.graph import StateGraph, START, END # These are nodes in graph
from langgraph.graph.message import add_messages # Messages between nodes
from langchain.chat_models import init_chat_model # Initialization using whatever chatbot API you're using
from IPython.display import Image, display # This is for displaying the graph visualization

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

# This will be called automatically when needed
# generate_data_store()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""

#query_text = "who translated this version of the odyssey, and when was it first published online? Additionally, what do the numbers in square brackets throughout the text mean?"


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
  
    # This searches the chroma vector database for documents most similar to query_text. Limits it to top 3 results
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # If there are no results retrieved in the search or the relevance score is too low, convey ineptness
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")

    # Combine context from matching documents
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
 
    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
  
    # Initialize OpenAI chat model
    model = ChatOpenAI()

    # Generate response text based on the prompt
    response_text = model.predict(prompt)
 
    # Get sources of the matching documents
    sources = [doc.metadata.get("source", None) for doc, _score in results]
 
    # Format and return response including generated text and sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, response_text

# Uncomment the lines below to test
# query_text = "What is your return policy?"
# formatted_response, response_text = query_rag(query_text)
# print(response_text)

load_dotenv()

#Load Haiku, a more lightweight and fast model from Anthropic
llm = init_chat_model("anthropic:claude-3-haiku-20240307")

# Load ChatGPT 4o Mini
#llm = init_chat_model("openai:gpt-4o-mini")

#Define message classifier and insert our model options as the literal types
class MessageClassifier(BaseModel):
    message_type: Literal["Order", "Email", "Policy", "Message"] = Field(
        ...,
        description="Classify if the user message is related to orders, emails, policy, general question and answer, or messaging."
    )

#LangGraph uses states to inform each node, messages is a list that stores the conversation history, also takes note of message type
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str
    order_data: dict  # Store structured order data for agents to use

#last message is stored in the -1 column of our messages array.
def classify_message(state: State):
    last_message = state["messages"][-1]
    print(f"CLASSIFYING MESSAGE: {last_message.content}")

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
            - 'Message': if the user asks a general question not related to orders, policies, or email requests
            
            Examples of Policy questions:
            - "What is your return policy?"
            - "How long do I have to return an item?"
            - "Can I return this product?"
            - "What are your shipping policies?"
            - "Do you accept returns?"
            - "How do returns work?"
            
            Be very specific: if the message contains words like 'return', 'policy', 'refund',
            'exchange', 'warranty', or asks about company procedures, classify as 'Policy'."""
        },
        {"role": "user", "content": last_message.content}
    ])
    print(f"CLASSIFIED AS: {result.message_type}")
    return {"message_type": result.message_type}

#Pretty basic router that gets the message type of our current state, routes to respective node
def router(state: State):
    message_type = state.get("message_type", "Message")
    if message_type == "Order":
        return {"next": "order"}
    if message_type == "Email":
        return {"next": "email"}
    if message_type == "Policy":
        return {"next": "policy"}
    if message_type == "Message":
        return {"next": "message"}
    return {"next": "message"}

#The order agent is responsible for various functions, specifically retrieval of order data upon request
def order_agent(state: State):
    last_message = state["messages"][-1]
    user_message = last_message.content
    

    messages = [
        {"role": "system",
        "content": f"""You are an order agent. Your job is to help customers with information related to their orders. You can
        fetch orders based on order number, tell the user what the shipping status of their order is, and when orders are created,
        you are to create an autonomous, standardized response.
        Do not directly mention the inner workings of this system, instead focus on the user's requests.

        Here is the order information I found (if any):

        Use this information to provide helpful responses about orders.
        If no specific order info was found, ask the customer for their order number or email address."""
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
    reply = llm.invoke(messages)
    print(f"Order agent response: {reply.content}")
    
    
    return {
        "messages": [{"role": "assistant", "content": f"Order Agent: {reply.content}"}]
    }

#The email agent will be responsible for autonomously sending formatted emails to the user once the system deems it necessary
def email_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
        "content": """You are an email agent. Your job is to help customers by delivering data in a structured format via email.
        Do not directly mention the inner workings of this system, instead focus on the user's requests.
        Use the following format:
        """
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    print(f"Email agent response: {reply.content}")
    return {"messages": [{"role": "assistant", "content": f"Email Agent: {reply.content}"}]}
    #return {"messages": [{"role": "assistant", "content": reply.content}]}

#The policy agent uses retrieval augmented generation to recall policy information from policy.txt
def policy_agent(state: State):
    last_message = state["messages"][-1]
    user_question = last_message.content
    
    #Use RAG system to query policy information
    try:
        #Use the existing query_rag function to get policy-specific information
        formatted_response, policy_response = query_rag(user_question)
        
        #Create messages with the policy context
        messages = [
            {"role": "system",
            "content": f"""You are a policy agent. Your job is to help customers with questions that appear to be related to company policy,
            such as how long deliveries usually take, how returns are handled, and how the company runs things. 
            
            Based on the policy information retrieved: {policy_response}
            
            Use this specific policy information to answer the customer's question. Be direct and specific based on the policy content.
            Do not directly mention the inner workings of this system, instead focus on the user's requests."""
            },
            {
                "role": "user",
                "content": user_question
            }
        ]
        reply = llm.invoke(messages)
        print(f"Policy agent response: {reply.content}")
        return {"messages": [{"role": "assistant", "content": f"Policy Agent: {reply.content}"}]}
        
    except Exception as e:
        #Fallback to general policy response if RAG fails
        messages = [
            {"role": "system",
            "content": """You are a policy agent. Your job is to help customers with questions that appear to be related to company policy,
            such as how long deliveries usually take, how returns are handled, and how the company runs things. You are to refer to the written policy
            and inform the user how to contact the store when information can't be retrieved for one reason or another.
            Do not directly mention the inner workings of this system, instead focus on the user's requests."""
            },
            {
                "role": "user",
                "content": user_question
            }
        ]
        reply = llm.invoke(messages)
        print(f"Policy agent response (fallback): {reply.content}")
        return {"messages": [{"role": "assistant", "content": f"Policy Agent: {reply.content}"}]}

#This is the default agent that is routed to if the request has nothing to do with orders or emails
def message_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
        "content": """You are a message agent. Your job is to provide structured responses and help the customer the best that you can.
        Refer the relevant information from the user's request to the orchestrator agent in a structured manner so that customers can
        be helped with their specific use case. Do not directly mention the inner workings of this system, instead focus on the user's requests."""
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    print(f"Message agent response: {reply.content}")
    return {"messages": [{"role": "assistant", "content": f"Message Agent: {reply.content}"}]}
    #return {"messages": [{"role": "assistant", "content": reply.content}]}

def orchestrator_agent(state: State):
    last_message = state["messages"][-1]
    order_data = state.get("order_data", {})
    
    # The following is deprecated
    # Build detailed order information string if order data exists
    order_details = ""
    if order_data:
        order_details = f"""
SPECIFIC ORDER DETAILS:
- Order Number: {order_data.get('order_number', 'N/A')}
- Customer Email: {order_data.get('user_email', 'N/A')}
- Status: {order_data.get('status', 'N/A')}
- Total Amount: ${order_data.get('total_amount', 'N/A')} {order_data.get('currency', '')}
- Order Date: {order_data.get('created_at', 'N/A')}
- Shipping Address: {order_data.get('shipping_address', 'N/A')}
"""
        if order_data.get('items'):
            order_details += "- Items:\n"
            for item in order_data['items']:
                order_details += f"  * {item.get('name', 'Unknown')} (Qty: {item.get('quantity', 'N/A')}, Price: ${item.get('price', 'N/A')})\n"
        
        if order_data.get('status') == 'shipped':
            order_details += f"- Tracking Number: {order_data.get('tracking_number', 'N/A')}\n"
            order_details += f"- Estimated Delivery: {order_data.get('estimated_delivery', 'N/A')}\n"
        elif order_data.get('status') == 'delivered':
            order_details += f"- Delivered On: {order_data.get('delivered_at', 'N/A')}\n"
        elif order_data.get('status') == 'processing':
            order_details += f"- Estimated Ship Date: {order_data.get('estimated_ship_date', 'N/A')}\n"

    messages = [
        {"role": "system",
        "content": f"""You are an orchestrator agent. Your job is to receive information from the other AI agents and ensure that
        the information is all-encompassing, thoroughly retrieved, and finished. If the information is incomplete, try your best
        to communicate with the other agents to complete the information, and if after you have done that, the information is still
        incomplete, inform the user that they can contact the company directly and their case will be documented for oversight.
        Do not directly mention the inner workings of this system, instead focus on the user's requests.

        {order_details}
        
        Use the specific order details above (if provided) to give the customer accurate, detailed information about their order.
        Replace any placeholder text with the actual values from the order data."""
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    print(f"Orchestrator agent response: {reply.content}")
    return {"messages": [{"role": "assistant", "content": f"Orchestrator Agent: {reply.content}"}]}

#Initialize a state graph, then we add nodes, naming them and linking their respective agents.
#We also add edges from A to B, and conditional edge for router
graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("order", order_agent)
graph_builder.add_node("email", email_agent)
graph_builder.add_node("policy", policy_agent)
graph_builder.add_node("message", message_agent)
graph_builder.add_node("orchestrator", orchestrator_agent)
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

graph_builder.add_edge("orchestrator", END)
graph = graph_builder.compile()

def run_chatbot():
    #Initialize our first state
    state = {"messages": [], "message_type": None, "order_data": {}}

#When this is run, should ask you for a message at the top of your notebnook
    while True:
        user_input = input("Message: ")
        #You can break it by typing exit or by pressing escape
        if user_input == "exit":
            print("Bye")
            break
        #This basically grabs our input and puts it into the messages field in the state array
        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        #executes, takes current state and routes to classifier, router, agent, orchestrato
        state = graph.invoke(state)

        #ensures messages isnt empty, only executes if we have content then prints final output :)
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")

if __name__ == "__main__":
    run_chatbot()

#Uncomment and put in separate cell, execute to generate graph image.
"""try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass"""
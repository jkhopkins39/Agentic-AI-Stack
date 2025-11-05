import uuid
import warnings
import re
from langgraph.graph import StateGraph, START, END
from agents import (
    State,
    classify_message,
    router,
    order_agent,
    email_agent,
    policy_agent,
    message_agent,
    orchestrator_agent
)
from database import (
    create_conversation,
    update_conversation_context,
    save_query_to_conversation
)

# Suppress LangChain deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# Initialize a state graph, then we add nodes, naming them and linking their respective agents
graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("order", order_agent)
graph_builder.add_node("email", email_agent)
graph_builder.add_node("policy", policy_agent)
graph_builder.add_node("message", message_agent)
graph_builder.add_node("orchestrator", orchestrator_agent)

# Add edges from start to classifier; once classified, goes to router
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


def run_chatbot():
    """Enhanced chatbot with conversation memory"""
    session_id = str(uuid.uuid4())
    
    # Initialize conversation in database
    conversation_id = create_conversation(session_id)
    if not conversation_id:
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
        "needs_email_notification": False,
        "notification_preference": None,
        "needs_follow_up_notification": False
    }
    
    print("=" * 60)
    print("AI Assistant - Ready to help!")
    print("I can help you with orders, policies, information changes, and general questions.")
    print("Type 'exit', 'quit', or 'q' to exit.")
    print("=" * 60)

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
                
                # Extract content from message (handle both dict and message objects)
                response_content = None
                if isinstance(last_message, dict):
                    response_content = last_message.get("content", "")
                elif hasattr(last_message, "content"):
                    # Handle LangChain message objects
                    content = last_message.content
                    # If content is itself a string representation of a message object, extract just the text
                    if isinstance(content, str):
                        response_content = content
                    else:
                        response_content = str(content)
                else:
                    response_content = str(last_message)
                
                # Clean up the response content
                if response_content:
                    # Replace literal \n with actual newlines
                    response_content = response_content.replace("\\n", "\n")
                    # Remove any agent prefixes if present
                    if response_content.startswith(("Message Agent:", "Order Agent:", "Email Agent:", "Policy Agent:")):
                        if ": " in response_content:
                            response_content = response_content.split(": ", 1)[1]
                    
                    # Ensure we have a clean string (remove any object representation artifacts)
                    if isinstance(response_content, str):
                        # Remove any LangChain message object string representations
                        if "content='" in response_content and "additional_kwargs=" in response_content:
                            # Extract just the content part from the string representation
                            match = re.search(r"content='([^']*(?:\\.[^']*)*)'", response_content)
                            if match:
                                response_content = match.group(1).replace("\\n", "\n")
                    
                    # Print the cleaned response
                    print(f"\nAssistant:\n{response_content}\n")
                    
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
                        # Silently fail database saves to keep output clean
                        pass
                    
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again or contact support // agenticaistack@gmail.com.")


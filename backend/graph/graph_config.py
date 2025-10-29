import uuid
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
        "needs_email_notification": False,
        "notification_preference": None,
        "needs_follow_up_notification": False
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


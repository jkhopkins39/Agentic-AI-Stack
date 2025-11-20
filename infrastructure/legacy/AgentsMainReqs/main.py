#!/usr/bin/env python3
"""
Integrated Agentic AI System
It's got basic mode and enhanced mode with validation and correlation id
"""

from email import message
from dotenv import load_dotenv
from typing import Annotated, Literal, Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from IPython.display import Image, display
import sys
import os
import logging

load_dotenv()

llm = init_chat_model("anthropic:claude-3-haiku-20240307")


SAMPLE_ORDERS = {
    "ORD-2024-001": {
        "order_number": "ORD-2024-001",
        "user_email": "john.doe@email.com",
        "status": "shipped",
        "total_amount": 129.99,
        "currency": "USD",
        "correlation_id": "550e8400-e29b-41d4-a716-446655440001",  # Added for enhanced features
        "validation_status": "validated",  # Added for enhanced features
        "validation_errors": [],  # Added for enhanced features
        "items": [
            {"name": "Wireless Headphones", "quantity": 1, "price": 99.99},
            {"name": "Phone Case", "quantity": 2, "price": 15.00}
        ],
        "shipping_address": "123 Main St, Anytown, CA 90210",
        "created_at": "2024-01-15",
        "shipped_at": "2024-01-17",
        "tracking_number": "1Z999AA1234567890",
        "estimated_delivery": "2024-01-20"
    },
    # Add more sample orders for testing
    "TEST-001": {
        "order_number": "TEST-001",
        "user_email": "jane.smith@email.com",
        "status": "pending",
        "total_amount": 1500.00,
        "currency": "USD",
        "correlation_id": "550e8400-e29b-41d4-a716-446655440002",
        "validation_status": "pending",
        "validation_errors": [],
        "items": [
            {"name": "Laptop Computer", "quantity": 1, "price": 1500.00}
        ],
        "shipping_address": "456 Oak Ave, Another City, NY 10001",
        "created_at": "2024-01-18",
        "shipped_at": None,
        "tracking_number": None,
        "estimated_delivery": None
    }
}

# Your existing helper functions (unchanged)
def find_order(order_number: str):
    """Helper function to find order by order number"""
    return SAMPLE_ORDERS.get(order_number.upper())

def search_orders_by_email(email: str):
    """Helper function to find orders by email"""
    return [order for order in SAMPLE_ORDERS.values() if order["user_email"].lower() == email.lower()]

# New helper function for correlation support
def find_order_by_correlation_id(correlation_id: str):
    """Helper function to find order by correlation_id"""
    for order in SAMPLE_ORDERS.values():
        if order.get("correlation_id") == correlation_id:
            return order
    return None

# Your existing MessageClassifier (unchanged)
class MessageClassifier(BaseModel):
    message_type: Literal["Order", "Email", "Policy", "Message"] = Field(
        ...,
        description="Classify if the user message is related to orders, emails, policy, general question and answer, or messaging."
    )

# State class 
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str
    order_data: dict  
    correlation_id: Optional[str]
    kafka_event: Optional[Dict[str, Any]]
    priority: Optional[int]
    task_id: Optional[str]
    agent_assignment: Optional[str]

def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as one of the following:
            - 'Order': if the user asks about order information such as shipping status, price, order quantity, etc.
            - 'Email': if the user mentions anything related to email or if they inquire about information that could be sent to them in a structured email
            - 'Policy': if the user asks a question about returns, shipping times, or any other information that seems to be related to company policy
            - 'Message': if the user asks a question about anything not related to an order or policy that would not warrant an email but would warrant an immediate response in the form of a structured chat message"""
        },
        {"role": "user", "content": last_message.content}
    ])
    return {"message_type": result.message_type}

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

def order_agent(state: State):
    last_message = state["messages"][-1]
    user_message = last_message.content
    

    import re
    order_pattern = r'(ORD-\d{4}-\d{3}|TEST-\d{3})' 
    order_match = re.search(order_pattern, user_message.upper())
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, user_message)
    
    order_info = ""
    found_order = None
    
   
    if order_match:
        order_number = order_match.group(1)
        order = find_order(order_number)
        if order:
            found_order = order
            order_info = f"""
FOUND ORDER: {order['order_number']}
Status: {order['status']}
Total: ${order['total_amount']} {order['currency']}
Items: {', '.join([f"{item['name']} (qty: {item['quantity']})" for item in order['items']])}
Shipping Address: {order['shipping_address']}
Order Date: {order['created_at']}
"""
            
            if order.get('validation_status'):
                order_info += f"Validation Status: {order['validation_status']}\n"
            
            if order['status'] == 'shipped':
                order_info += f"Tracking Number: {order['tracking_number']}\n"
                order_info += f"Estimated Delivery: {order.get('estimated_delivery', 'N/A')}\n"
            elif order['status'] == 'delivered':
                order_info += f"Delivered On: {order.get('delivered_at', 'N/A')}\n"
            elif order['status'] == 'processing':
                order_info += f"Estimated Ship Date: {order.get('estimated_ship_date', 'N/A')}\n"
        else:
            order_info = f"Order {order_number} not found in our system."
    
    elif email_match:
        email = email_match.group(0)
        orders = search_orders_by_email(email)
        if orders:
            found_order = orders[0] if orders else None
            order_info = f"Found {len(orders)} order(s) for {email}:\n"
            for order in orders:
                order_info += f"- {order['order_number']}: {order['status']} (${order['total_amount']})\n"
        else:
            order_info = f"No orders found for email {email}."

    messages = [
        {"role": "system",
        "content": f"""You are an order agent. Your job is to help customers with information related to their orders. You can
        fetch orders based on order number, tell the user what the shipping status of their order is, and when orders are created,
        you are to create an autonomous, standardized response. Do not directly mention the inner workings of this system, instead focus on the user's requests.

        Here is the order information I found (if any):
        {order_info}
        
        Use this information to provide helpful responses about orders. If no specific order info was found, ask the customer for their order number or email address."""
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
    reply = llm.invoke(messages)
    
    return {
        "messages": [{"role": "assistant", "content": f"Order Agent: {reply.content}"}],
        "order_data": found_order or {}
    }

def email_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
        "content": """You are an email agent. Your job is to help customers by delivering data in a structured format via email.
        Do not directly mention the inner workings of this system, instead focus on the user's requests."""
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": f"Email Agent: {reply.content}"}]}

def policy_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
        "content": """You are a policy agent. Your job is to help customers with questions that appear to be related to company policy,
        such as how long deliveries usually take, how returns are handled, and how the company runs things. You are to refer to the written policy
        and inform the user how to contact the store when information can't be retrieved for one reason or another.
        Do not directly mention the inner workings of this system, instead focus on the user's requests."""
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": f"Policy Agent: {reply.content}"}]}

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
            "content": user_message
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": f"Message Agent: {reply.content}"}]}

def orchestrator_agent(state: State):
    last_message = state["messages"][-1]
    order_data = state.get("order_data", {})
    
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
        # Enhanced details
        if order_data.get('correlation_id'):
            order_details += f"- Tracking ID: {order_data['correlation_id'][:8]}...\n"
        
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
    return {"messages": [{"role": "assistant", "content": f"Orchestrator Agent: {reply.content}"}]}

def create_basic_graph():
    """Create your original LangGraph (exactly as you had it)"""
    graph_builder = StateGraph(State)
    graph_builder.add_node("classify", classify_message)  # Added classify step
    graph_builder.add_node("router", router)
    graph_builder.add_node("order", order_agent)
    graph_builder.add_node("email", email_agent)
    graph_builder.add_node("policy", policy_agent)
    graph_builder.add_node("message", message_agent)
    graph_builder.add_node("orchestrator", orchestrator_agent)
    
    graph_builder.add_edge(START, "classify")
    graph_builder.add_edge("classify", "router")
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
    
    return graph_builder.compile()

def run_chatbot():
    """Your original interactive chatbot"""
    print("🤖 Interactive Chatbot Mode")
    print("=" * 40)
    print("Type your questions below. Type 'exit' to quit.")
    print("Examples:")
    print("- 'Where is my order ORD-2024-001?'")
    print("- 'What's your return policy?'")
    print("- 'I need help with my account'")
    print("-" * 40)
    
    graph = create_basic_graph()
    
    state = {
        "messages": [], 
        "message_type": None, 
        "order_data": {},
        "correlation_id": None,
        "kafka_event": None,
        "priority": None,
        "task_id": None,
        "agent_assignment": None
    }

    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Bye!")
            break
            
        if not user_input:
            continue

        # Add user message to state
        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        try:
            # Process through your LangGraph
            result = graph.invoke(state)
            
            # Update state with result
            state.update(result)

            # Display the response
            if state.get("messages") and len(state["messages"]) > 0:
                last_message = state["messages"][-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    print(f"🤖 {last_message['content']}")
                else:
                    print(f"🤖 {last_message}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            logging.error(f"Chatbot error: {e}")

def run_enhanced_mode():
    """Run with Kafka integration (optional)"""
    try:
        # Import enhanced components (only if available)
        from agent_config import AgentIntegrationConfig, DatabaseManager
        from policy_integration import integrate_policy_agent_with_scheduler
        
        print("🚀 Enhanced Mode with Kafka Integration")
        print("=" * 50)
        
        # This would initialize the full enhanced system
        # Implementation would go here...
        print("Enhanced mode would start here...")
        print("(Requires Kafka infrastructure to be running)")
        
    except ImportError:
        print("❌ Enhanced mode requires additional files:")
        print("  - agent_config.py")
        print("  - policy_integration.py")
        print("  - aligned_enhanced_agents.py")
        print("\nFalling back to basic mode...")
        run_chatbot()

def main():
    """Main entry point with mode selection"""
    mode = sys.argv[1] if len(sys.argv) > 1 else 'basic'
    
    if mode == 'basic':
        print("Mode: BASIC - Your original interactive chatbot")
        run_chatbot()
    elif mode == 'enhanced':
        print("Kafka integration stuff")
        run_enhanced_mode()
    elif mode == 'test':
        print("Mode: TEST - Testing your agents")
        test_agents()
    else:
        print("Usage: python integrated_main.py [basic|enhanced|test]")
        print("  basic    - Your original interactive chatbot")
        print("  enhanced - With Kafka integration (requires infrastructure)")
        print("  test     - Test agent responses")

def test_agents():
    """Test your agents with sample queries"""
    print("🧪 Testing Agent Responses")
    print("=" * 30)
    
    graph = create_basic_graph()
    
    test_queries = [
        "Where is my order ORD-2024-001?",
        "What's your return policy?", 
        "I need help with my account",
        "Can you email me my order details?",
        "My order TEST-001 hasn't arrived"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing: '{query}'")
        print("-" * 40)
        
        try:
            result = graph.invoke({
                "messages": [{"role": "user", "content": query}],
                "message_type": None,
                "order_data": {}
            })
            
            # Show the response
            if result.get("messages"):
                last_message = result["messages"][-1]
                response = last_message.get("content", str(last_message))
                print(f"✅ Response: {response[:200]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ Agent testing complete!")

if __name__ == "__main__":
    main()
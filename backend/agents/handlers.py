"""Agent handler functions."""
import os
from langchain.chat_models import init_chat_model
from .models import (
    MessageClassifier,
    UserInformationParser,
    OrderSubTypeParser,
    OrderReceiptParser,
    NotificationPreferenceParser,
    State
)
from utils.logging import log_unclassified_query
from database import (
    lookup_user_by_email,
    update_user_information,
    lookup_order_by_number,
    lookup_orders_by_email,
    lookup_orders_by_product_name
)
from notifications import (
    send_notification,
    format_order_receipt,
    get_user_notification_preference,
    set_user_notification_preference,
    send_information_change_email
)
from rag import query_rag

# Session configuration
DEFAULT_SYSTEM_EMAIL = "support@agenticaistack.com"
SESSION_EMAIL = os.getenv('USER_EMAIL', DEFAULT_SYSTEM_EMAIL)

# Load LLM
llm = init_chat_model("anthropic:claude-3-haiku-20240307")


def classify_message(state: State):
    """Classify incoming message into categories"""
    last_message = state["messages"][-1]
    print(f"CLASSIFYING MESSAGE: {last_message.content}")

    # Use LangChain method to wrap the base language model to conform with message classifier schema
    classifier_llm = llm.with_structured_output(MessageClassifier)

    # Result is stored as result of classifier invocation
    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as one of the following:
            - 'Order': if the user asks about specific order information such as shipping status, tracking, price, order quantity, order number, order receipts, etc.
            - 'Email': if the user explicitly mentions email or requests information to be sent via email
            - 'Policy': if the user asks about returns, refunds, return policy, shipping policy,
            exchange policy, warranty, terms of service, company policies, return timeframes, return process, or any store/company rules and procedures
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
            - If the message asks for receipts, receipt information, order status, tracking, or any order-related information, classify as 'Order'"""
        },
        {"role": "user", "content": last_message.content}
    ])
    print(f"CLASSIFIED AS: {result.message_type}")
    
    # Log general/unclassified messages for evaluation metrics
    if result.message_type == "Message":
        session_id = state.get("session_id", "unknown")
        conversation_id = state.get("conversation_id", "unknown")
        log_unclassified_query(
            query=last_message.content,
            session_id=session_id,
            conversation_id=conversation_id,
            attempted_classification="Message"
        )
    
    # If it's an Order type, determine the sub-type
    if result.message_type == "Order":
        sub_type_llm = llm.with_structured_output(OrderSubTypeParser)
        sub_type_result = sub_type_llm.invoke([
            {
                "role": "system",
                "content": """Classify the specific type of order request:
                - 'receipt': if the user asks for order receipts, receipt information, or wants to see/download receipts
                - 'status': if the user asks about order status, order progress, or order updates
                - 'tracking': if the user asks about shipping tracking, delivery status, or tracking numbers
                - 'general': for other order-related questions like order history, order details, etc.
                
                Examples:
                - "Can I get a receipt for my order?" -> receipt
                - "What's the status of my order?" -> status
                - "Where is my package?" -> tracking
                - "Show me my order history" -> general"""
            },
            {"role": "user", "content": last_message.content}
        ])
        print(f"ORDER SUB-TYPE: {sub_type_result.order_sub_type}")
        return {"message_type": result.message_type, "order_sub_type": sub_type_result.order_sub_type}
    
    return {"message_type": result.message_type}


def router(state: State):
    """Route to appropriate agent based on message type"""
    message_type = state.get("message_type", "Message")
    if message_type == "Order":
        return {"next": "order"}
    if message_type == "Email":
        return {"next": "email"}
    if message_type == "Policy":
        return {"next": "policy"}
    if message_type == "Change Information":
        return {"next": "message"}
    if message_type == "Message":
        return {"next": "message"}
    return {"next": "message"}


def order_agent(state: State):
    """Handle order-related requests"""
    last_message = state["messages"][-1]
    user_message = last_message.content
    order_sub_type = state.get("order_sub_type", "general")
    
    # Handle order receipt requests
    if order_sub_type == "receipt":
        return handle_order_receipt_request(state)
    
    # Handle other order types (status, tracking, general)
    messages = [
        {"role": "system",
        "content": f"""You are an order agent. Your job is to help customers with information related to their orders. You can
        fetch orders based on order number, tell the user what the shipping status of their order is, and when orders are created,
        you are to create an autonomous, standardized response.
        Do not directly mention the inner workings of this system, instead focus on the user's requests.

        Order request type: {order_sub_type}
        
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


def email_agent(state: State):
    """Handle email requests and notifications"""
    last_message = state["messages"][-1]
    user_message = last_message.content
    
    # Check if this is a triggered email notification for information changes
    if state.get("needs_email_notification") and state.get("changes_made"):
        return handle_information_change_notification(state)
    
    # Default email handling for other email requests
    messages = [
        {"role": "system",
        "content": """You are an email agent. Your job is to help customers by delivering data in a structured format via email.
        Do not directly mention the inner workings of this system, instead focus on the user's requests.
        """
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
    reply = llm.invoke(messages)
    print(f"Email agent response: {reply.content}")
    return {"messages": [{"role": "assistant", "content": f"Email Agent: {reply.content}"}]}


def handle_information_change_notification(state: State):
    """Handle sending email notifications for information changes"""
    changes_made = state.get("changes_made", [])
    
    if not changes_made:
        return {"messages": [{"role": "assistant", "content": "Email Agent: No changes to notify about."}]}
    
    # Send the information change email
    email_sent = send_information_change_email(changes_made)
    
    if email_sent:
        response = f"I've sent you a security notification email about the changes made to your account. Please check your email at {SESSION_EMAIL}."
    else:
        changes_text = ", ".join(changes_made)
        response = f"Your {changes_text} has been updated successfully. I attempted to send you a security notification email, but there was an issue with the email service."
    
    return {
        "messages": [{"role": "assistant", "content": f"Email Agent: {response}"}],
        "needs_email_notification": False  # Reset the flag
    }


def handle_order_receipt_request(state: State):
    """Handle order receipt requests with context memory and multi-turn conversation"""
    last_message = state["messages"][-1]
    user_message = last_message.content
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
    
    # Use session email to process the request
    return process_receipt_request(parsing_result.model_dump(), SESSION_EMAIL, conversation_context, state)


def process_receipt_request(parsed_request: dict, user_email: str, conversation_context: dict, state: State):
    """Process the receipt request based on parsed data"""
    order_data = None
    
    if parsed_request.get("order_number") and parsed_request["order_number"] not in ['<UNKNOWN>', None, '']:
        # Look up by order number
        order_data = lookup_order_by_number(parsed_request["order_number"])
        if not order_data:
            response = f"I couldn't find an order with number '{parsed_request['order_number']}'. Please check the order number and try again."
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
            order_data = orders[0]
        else:
            response = f"I couldn't find any orders containing '{parsed_request['product_name']}' for {user_email}."
    
    else:
        # Default to most recent order
        orders = lookup_orders_by_email(user_email, limit=1)
        if orders:
            order_data = orders[0]
        else:
            response = f"I couldn't find any orders for {user_email}. Please check your email address."
    
    # If we found an order, check notification preference and send receipt
    if order_data:
        user_data = lookup_user_by_email(user_email)
        
        if not user_data:
            response = f"I found your order but couldn't retrieve your account information. Please contact support."
        else:
            # Check if user has a notification preference set
            preference = get_user_notification_preference(user_data['id'])
            
            if not preference or not preference.get('preferred_method'):
                # Ask user for preference
                response = f"I found your order {order_data['order_number']}! How would you like to receive the receipt? Reply with 'email', 'text', or 'both'."
                updated_context = conversation_context.copy()
                updated_context["pending_receipt_order"] = order_data
                updated_context["user_email"] = user_email
                updated_context["awaiting_notification_preference"] = True
                
                return {
                    "messages": [{"role": "assistant", "content": f"Email Agent: {response}"}],
                    "conversation_context": updated_context,
                    "order_data": order_data
                }
            else:
                # Use stored preference
                delivery_method = preference.get('preferred_method', 'email')
                notification_sent = send_notification('order_receipt', order_data, user_data, delivery_method)
                
                if notification_sent:
                    method_text = "email" if delivery_method == "email" else "text" if delivery_method == "sms" else "email and text"
                    response = f"I've sent the receipt for order {order_data['order_number']} via {method_text}."
                else:
                    receipt_content = format_order_receipt(order_data)
                    response = f"Here's your order receipt:\n\n{receipt_content}\n\nNote: I had trouble sending the notification, but here's your receipt information above."
    
    # Clean up conversation context
    updated_context = conversation_context.copy()
    updated_context["user_email"] = user_email
    
    return {
        "messages": [{"role": "assistant", "content": f"Email Agent: {response}"}],
        "conversation_context": updated_context
    }


def policy_agent(state: State):
    """Handle policy-related questions using RAG"""
    last_message = state["messages"][-1]
    user_question = last_message.content
    
    # Use RAG system to query policy information
    try:
        formatted_response, policy_response = query_rag(user_question)
        
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
        # Fallback to general policy response if RAG fails
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


def message_agent(state: State):
    """Handle general messages and information change requests"""
    last_message = state["messages"][-1]
    message_type = state.get("message_type", "Message")
    conversation_context = state.get("conversation_context", {})
    
    # Check if we're awaiting a notification preference response
    if conversation_context.get("awaiting_notification_preference"):
        return handle_notification_preference_response(state)
    
    # Handle Change Information requests 
    if message_type == "Change Information":
        return handle_change_information(state)
    
    # Default message handling
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


def handle_change_information(state: State):
    """Handle change information requests with multi-turn conversation support"""
    last_message = state["messages"][-1]
    user_message = last_message.content
    conversation_context = state.get("conversation_context", {})
    
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
    
    # Use session email to look up the user
    user_data = lookup_user_by_email(SESSION_EMAIL)
    
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
        response = f"I couldn't find your account with email {SESSION_EMAIL}. Please contact customer support for assistance."
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
        success = update_user_information(user_data['id'], updates)
        
        if success:
            changes_text = ", ".join(changes_made)
            
            # Check notification preference
            preference = get_user_notification_preference(user_data['id'])
            
            if not preference or not preference.get('preferred_method'):
                # Ask user for preference
                response = f"Great! I've successfully updated your {changes_text}. Would you like a confirmation via email, text, or both?"
                
                updated_context = conversation_context.copy()
                updated_context["user_identified"] = True
                updated_context["user_id"] = user_data['id']
                updated_context["user_email"] = SESSION_EMAIL
                updated_context["awaiting_notification_preference"] = True
                updated_context["pending_notification_type"] = "info_change"
                updated_context["pending_changes_made"] = changes_made
                
                information_changed = True
                needs_email_notification = False
            else:
                # Use stored preference to send notification
                delivery_method = preference.get('preferred_method', 'email')
                notification_sent = send_notification('info_change', {'changes_made': changes_made}, user_data, delivery_method)
                
                method_text = "email" if delivery_method == "email" else "text" if delivery_method == "sms" else "email and text"
                
                if notification_sent:
                    response = f"Great! I've successfully updated your {changes_text}. A confirmation has been sent via {method_text}."
                else:
                    response = f"Great! I've successfully updated your {changes_text}. Your account information has been updated in our system."
                
                updated_context = conversation_context.copy()
                updated_context["user_identified"] = True
                updated_context["user_id"] = user_data['id']
                updated_context["user_email"] = SESSION_EMAIL
                
                information_changed = True
                needs_email_notification = False
            
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


def handle_notification_preference_response(state: State):
    """Handle user's response to notification preference question"""
    last_message = state["messages"][-1]
    user_message = last_message.content.lower().strip()
    conversation_context = state.get("conversation_context", {})
    
    # Parse the preference from user message
    parser_llm = llm.with_structured_output(NotificationPreferenceParser)
    parsing_result = parser_llm.invoke([
        {
            "role": "system",
            "content": """Parse the user's notification preference from their message.
            Look for keywords:
            - 'email' or 'e-mail' -> preferred_method: 'email'
            - 'text', 'sms', 'message' -> preferred_method: 'sms'
            - 'both', 'all', 'email and text' -> preferred_method: 'both'
            
            Examples:
            "email please" -> preferred_method: 'email'
            "send it via text" -> preferred_method: 'sms'
            "both would be great" -> preferred_method: 'both'"""
        },
        {"role": "user", "content": user_message}
    ])
    
    print(f"Parsed notification preference: {parsing_result}")
    
    # Get user data
    user_data = lookup_user_by_email(SESSION_EMAIL)
    
    if not user_data:
        response = "I couldn't find your account. Please contact support."
        return {
            "messages": [{"role": "assistant", "content": f"Message Agent: {response}"}],
            "conversation_context": conversation_context
        }
    
    # Validate and save preference
    if parsing_result.preferred_method:
        # Save the preference
        success = set_user_notification_preference(user_data['id'], {
            'preferred_method': parsing_result.preferred_method
        })
        
        if success:
            # Check if there's a pending notification to send
            if conversation_context.get("pending_receipt_order"):
                order_data = conversation_context["pending_receipt_order"]
                notification_sent = send_notification('order_receipt', order_data, user_data, parsing_result.preferred_method)
                
                method_text = "email" if parsing_result.preferred_method == "email" else "text" if parsing_result.preferred_method == "sms" else "email and text"
                
                if notification_sent:
                    response = f"Perfect! I've saved your preference and sent the receipt for order {order_data['order_number']} via {method_text}."
                else:
                    receipt_content = format_order_receipt(order_data)
                    response = f"I've saved your preference. Here's your receipt:\n\n{receipt_content}\n\nNote: I had trouble sending the notification."
                
                # Clear pending data
                updated_context = conversation_context.copy()
                updated_context.pop("pending_receipt_order", None)
                updated_context.pop("awaiting_notification_preference", None)
                
            elif conversation_context.get("pending_notification_type") == "info_change":
                changes_made = conversation_context.get("pending_changes_made", [])
                notification_sent = send_notification('info_change', {'changes_made': changes_made}, user_data, parsing_result.preferred_method)
                
                method_text = "email" if parsing_result.preferred_method == "email" else "text" if parsing_result.preferred_method == "sms" else "email and text"
                
                if notification_sent:
                    response = f"Perfect! I've saved your preference and sent a confirmation via {method_text}."
                else:
                    response = f"I've saved your preference, but had trouble sending the confirmation notification."
                
                # Clear pending data
                updated_context = conversation_context.copy()
                updated_context.pop("pending_notification_type", None)
                updated_context.pop("pending_changes_made", None)
                updated_context.pop("awaiting_notification_preference", None)
            else:
                method_text = "email" if parsing_result.preferred_method == "email" else "text" if parsing_result.preferred_method == "sms" else "email and text"
                response = f"Great! I've saved your notification preference. You'll receive future notifications via {method_text}."
                
                updated_context = conversation_context.copy()
                updated_context.pop("awaiting_notification_preference", None)
        else:
            response = "I had trouble saving your preference. Please try again later."
            updated_context = conversation_context
    else:
        response = "I didn't understand your preference. Please reply with 'email', 'text', or 'both'."
        updated_context = conversation_context
    
    return {
        "messages": [{"role": "assistant", "content": f"Message Agent: {response}"}],
        "conversation_context": updated_context
    }


def orchestrator_agent(state: State):
    """Enhanced orchestrator that manages agent responses and triggers follow-up actions"""
    last_message = state["messages"][-1]
    
    # Check if information was changed and we need to send a notification email
    if state.get("information_changed") and state.get("needs_email_notification"):
        print("Orchestrator detected information change - routing to email agent for notification")
        if last_message.content.startswith(("Message Agent:", "Order Agent:", "Email Agent:", "Policy Agent:")):
            agent_response = last_message.content
            if ": " in agent_response:
                clean_response = agent_response.split(": ", 1)[1]
            else:
                clean_response = agent_response
            print(f"Orchestrator passing through: {clean_response}")
            
            return {
                "messages": [{"role": "assistant", "content": clean_response}],
                "needs_follow_up_email": True
            }
    
    # Check if the last message is from an agent
    if last_message.content.startswith(("Message Agent:", "Order Agent:", "Email Agent:", "Policy Agent:")):
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


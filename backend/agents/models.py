"""This file contains models to instantiate agents as well as state for graph."""

from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# Define message classifier and insert our model options as the literal types
class MessageClassifier(BaseModel):
    message_type: Literal["Order", "Email", "Policy", "Message", "Change Information"] = Field(
        ...,
        description="Classify if the user message is related to orders, emails, policy,"+
        " changing user information, and if it's none of those: messaging."
    )


# Define user information parser for change information requests
class UserInformationParser(BaseModel):
    current_user_email: Optional[str] = Field(None, description="The user's current email address mentioned in the request")
    new_user_email: Optional[str] = Field(None, description="The new email address the user wants to change to")
    current_firstname: Optional[str] = Field(None, description="The user's current first name mentioned in the request")
    new_firstname: Optional[str] = Field(None, description="The new first name the user wants to change to")
    current_lastname: Optional[str] = Field(None, description="The user's current last name mentioned in the request")
    new_lastname: Optional[str] = Field(None, description="The new last name the user wants to change to")
    current_phone: Optional[str] = Field(None, description="The user's current phone number mentioned in the request")
    new_phone: Optional[str] = Field(None, description="The new phone number the user wants to change to")


# Define order sub-type parser to determine if an order request is for receipt, status, etc.
class OrderSubTypeParser(BaseModel):
    order_sub_type: Literal["receipt", "status", "tracking", "general"] = Field(
        ...,
        description="Classify the specific type of order request: receipt for order receipts, status for order status, tracking for shipping tracking, general for other order questions"
    )


# Define order receipt parser for order receipt requests
class OrderReceiptParser(BaseModel):
    user_email: Optional[str] = Field(None, description="The user's email address to identify their orders")
    order_number: Optional[str] = Field(None, description="Specific order number mentioned (e.g., ORD-2024-001)")
    chronological_request: Optional[str] = Field(None, description="Chronological request like 'last', 'most recent', 'latest', 'previous'")
    product_name: Optional[str] = Field(None, description="Product name mentioned to find orders containing that product")
    time_reference: Optional[str] = Field(None, description="Time reference like 'yesterday', 'last week', 'this month'")


# Define notification preference parser for handling user notification preference requests
class NotificationPreferenceParser(BaseModel):
    preferred_method: Optional[Literal["email", "sms", "both"]] = Field(None, description="User's preferred notification method: 'email', 'sms', or 'both'")
    receipt_notifications: Optional[bool] = Field(None, description="Whether user wants to receive receipt notifications")
    info_change_notifications: Optional[bool] = Field(None, description="Whether user wants to receive information change notifications")
    order_update_notifications: Optional[bool] = Field(None, description="Whether user wants to receive order update notifications")


# LangGraph state definition
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str
    order_sub_type: Optional[str]  # Sub-type for order requests (receipt, status, tracking, general)
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
    notification_preference: Optional[str]  # User's preferred notification method (email, sms, both)
    needs_follow_up_notification: bool  # Flag to send notification after action completes


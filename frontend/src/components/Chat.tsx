import React, { useState, useEffect, useRef } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Send, Clock, CheckCircle, XCircle, Bot, Loader2 } from 'lucide-react';
import { useUser } from '../contexts/UserContext';
import { API_BASE_URL, WS_BASE_URL } from '../config';

type MessageStatus = 'pending' | 'fulfilled' | 'unfulfilled';

interface Message {
  id: string;
  content: string;
  status: MessageStatus;
  timestamp: Date;
  isUser: boolean;
  agentType?: string;
}

interface ChatProps {
  conversationId?: string;
  sessionId?: string;
  onNewChat?: () => void;
  onMessageSent?: () => void;
  onConversationCreated?: (conversationId: string, sessionId: string) => void;
}

export function Chat({ conversationId: initialConversationId, sessionId: initialSessionId, onNewChat, onMessageSent, onConversationCreated }: ChatProps = {}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [sessionId, setSessionId] = useState(() => initialSessionId || `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`);
  const [conversationId, setConversationId] = useState<string | undefined>(initialConversationId);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const lastLoadedConversationRef = useRef<string | undefined>(undefined);
  const shouldAutoScrollRef = useRef(true);
  const messagesRef = useRef<Message[]>([]);
  const skipNextHistoryLoadRef = useRef(false);
  const receivedMessageIds = useRef<Set<string>>(new Set());
  const { userProfile } = useUser();

  // Get user name for display
  const userName = userProfile?.profile 
    ? `${userProfile.profile.first_name || ''} ${userProfile.profile.last_name || ''}`.trim() || userProfile.profile.email
    : 'User';

  // Keep messagesRef in sync with messages state
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // Track previous conversationId to detect transitions
  const prevConversationIdRef = useRef<string | undefined>(initialConversationId);

  // Update conversationId when prop changes
  useEffect(() => {
    if (initialConversationId !== conversationId) {
      setConversationId(initialConversationId);
    }
  }, [initialConversationId, conversationId]);

  // Load conversation history when conversationId changes
  useEffect(() => {
    const prevConversationId = prevConversationIdRef.current;
    prevConversationIdRef.current = conversationId;

    if (conversationId && conversationId !== lastLoadedConversationRef.current) {
      if (skipNextHistoryLoadRef.current) {
        console.log('Skipping history load for newly created conversation');
        skipNextHistoryLoadRef.current = false;
        lastLoadedConversationRef.current = conversationId;
      } else {
        loadConversationHistory(conversationId);
      }
    } else if (!conversationId && prevConversationId !== undefined) {
      // New chat - clear messages and reset state
      // Only reset if we're transitioning from a conversation to no conversation
      // This prevents resetting on initial mount when conversationId starts as undefined
      console.log('New chat: clearing previous conversation state');
      setMessages([]);
      messagesRef.current = [];
      receivedMessageIds.current.clear();
      setInputValue('');
      lastLoadedConversationRef.current = undefined;
      // Create new session ID for new chat
      const newSessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      setSessionId(newSessionId);
      // Reset scroll position and enable auto-scroll for new chat
      shouldAutoScrollRef.current = true;
      setTimeout(() => scrollToBottom(true), 100);
      console.log('New chat started, reset state');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId]);

  // Update sessionId when prop changes
  useEffect(() => {
    if (initialSessionId && initialSessionId !== sessionId) {
      setSessionId(initialSessionId);
    }
  }, [initialSessionId]);


  const loadConversationHistory = async (convId: string) => {
    if (lastLoadedConversationRef.current === convId) {
      return;
    }
    
    setLoadingHistory(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/conversations/${convId}`);
      const data = await response.json();
      
      // Convert conversation messages to Chat messages
      const loadedMessages: Message[] = [];
      for (const msg of data.messages || []) {
        // Add user message
        loadedMessages.push({
          id: `user-${msg.message_order}`,
          content: msg.query_text,
          status: 'fulfilled',
          timestamp: new Date(msg.created_at),
          isUser: true
        });
        
        // Add agent response
        loadedMessages.push({
          id: `agent-${msg.message_order}`,
          content: msg.agent_response,
          status: 'fulfilled',
          timestamp: new Date(msg.created_at),
          isUser: false,
          agentType: msg.agent_type
        });
      }
      
      setMessages(loadedMessages);
      messagesRef.current = loadedMessages;
      lastLoadedConversationRef.current = convId;
      // Clear received message IDs when loading a new conversation to prevent false duplicates
      receivedMessageIds.current.clear();
      if (data.conversation?.session_id) {
        setSessionId(data.conversation.session_id);
      }
      // Scroll to bottom after loading history
      setTimeout(() => {
        scrollToBottom(true);
        shouldAutoScrollRef.current = true;
      }, 100);
    } catch (error) {
      console.error('Error loading conversation history:', error);
    } finally {
      setLoadingHistory(false);
    }
  };

  // Check if user is near the bottom of the messages container
  const isNearBottom = () => {
    if (!messagesContainerRef.current) return true;
    const container = messagesContainerRef.current;
    const threshold = 150; // pixels from bottom
    const scrollTop = container.scrollTop;
    const scrollHeight = container.scrollHeight;
    const clientHeight = container.clientHeight;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    
    // If user is at the very top (within 50px), definitely don't auto-scroll
    if (scrollTop < 50) {
      return false;
    }
    
    // Otherwise check if near bottom
    const isNear = distanceFromBottom < threshold;
    return isNear;
  };

  // Auto-scroll to bottom when new messages arrive (only if user is near bottom)
  const scrollToBottom = (force = false) => {
    if (force) {
      // Force scroll - always scroll
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      shouldAutoScrollRef.current = true;
    } else if (shouldAutoScrollRef.current) {
      // Only scroll if user is near bottom
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Handle scroll events to detect if user scrolled up
  const handleScroll = () => {
    if (messagesContainerRef.current) {
      const wasNearBottom = shouldAutoScrollRef.current;
      const isNear = isNearBottom();
      shouldAutoScrollRef.current = isNear;
      
      // Log for debugging (can remove later)
      if (wasNearBottom !== isNear) {
        console.log(`Auto-scroll ${isNear ? 'enabled' : 'disabled'} - distance from bottom: ${messagesContainerRef.current.scrollHeight - messagesContainerRef.current.scrollTop - messagesContainerRef.current.clientHeight}px`);
      }
    }
  };

  useEffect(() => {
    // Always check scroll position before deciding to auto-scroll
    // This ensures we respect user's scroll position
    if (messages.length === 0) return;
    
    const checkAndScroll = () => {
      if (messagesContainerRef.current && messagesEndRef.current) {
        const isNear = isNearBottom();
        shouldAutoScrollRef.current = isNear;
        
        // Only auto-scroll if user is near bottom
        if (isNear) {
          messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
      }
    };
    
    // Use requestAnimationFrame to ensure DOM is updated before checking
    const timeoutId = setTimeout(() => {
      requestAnimationFrame(checkAndScroll);
    }, 100);
    
    return () => clearTimeout(timeoutId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages.length]);

  // WebSocket connection for real-time responses
  useEffect(() => {
    let reconnectTimeout: number | null = null;
    let isMounted = true;
    
    const connectWebSocket = () => {
      if (!isMounted) return;
      
      try {
        // Close existing connection if any
        if (wsRef.current) {
          wsRef.current.close();
        }
        
        const ws = new WebSocket(`${WS_BASE_URL}/ws/agent-responses/${sessionId}`);
        
        ws.onopen = () => {
          if (!isMounted) {
            ws.close();
            return;
          }
          console.log('WebSocket connected');
          setIsConnected(true);
          
          // Add greeting message on first connection if no messages exist
          setMessages(prev => {
            if (prev.length === 0) {
              const greetingMessage: Message = {
                id: `greeting-${Date.now()}`,
                content: 'Hello! How can I help you?',
                status: 'fulfilled',
                timestamp: new Date(),
                isUser: false,
                agentType: 'orchestrator'
              };
              return [greetingMessage];
            }
            return prev;
          });
        };

        ws.onmessage = (event) => {
          if (!isMounted) return;
          
          try {
            const data = JSON.parse(event.data);
            console.log('Received WebSocket message:', data);
            
            // Skip connection confirmation and keepalive messages - these should not be displayed as chat messages
            if (data.status === 'connected' || data.type === 'keepalive') {
              return;
            }
            
            // Only process messages that have actual content - skip empty or default messages
            if (!data.message || data.message.trim() === '' || data.message === 'Response received') {
              return;
            }
            
            // Update conversation_id if provided
            // This happens when a WebSocket response includes a conversation_id for a new conversation
            if (data.conversation_id) {
              if (!conversationId) {
                // Update the ref BEFORE setting conversationId to prevent history reload
                // This ensures we don't clear the current messages when conversationId is set
                if (messagesRef.current.length > 0) {
                  lastLoadedConversationRef.current = data.conversation_id;
                  prevConversationIdRef.current = undefined; // Set previous to undefined so transition is detected correctly
                }
                skipNextHistoryLoadRef.current = true;
                setConversationId(data.conversation_id);
                // Notify parent component that a new conversation was created
                if (onConversationCreated) {
                  onConversationCreated(data.conversation_id, sessionId);
                }
              }
            }
            
            // Remove "Processing your request..." message if it exists and add agent response
            setMessages(prev => {
              // First, remove any "Processing your request..." messages
              const withoutProcessing = prev.filter(msg => 
                !(msg.content === 'Processing your request...' && msg.status === 'pending')
              );
              
              // Create a unique message ID based on content, timestamp, and agent type
              // Use correlation_id if available, otherwise create one
              const messageId = data.correlation_id || `msg-${Date.now()}-${Math.random()}`;
              const messageContent = data.message;
              
              // Check if we've already received this exact message (prevent duplicates)
              if (receivedMessageIds.current.has(messageId)) {
                console.log('Duplicate message detected, ignoring:', messageId);
                return withoutProcessing;
              }
              
              // Mark this message as received
              receivedMessageIds.current.add(messageId);
              
              // Then add the agent response
              const agentMessage: Message = {
                id: messageId,
                content: messageContent,
                status: 'fulfilled',
                timestamp: new Date(),
                isUser: false,
                agentType: data.agent_type || 'Agent'
              };
              
              return [...withoutProcessing, agentMessage];
            });
            
            // Auto-scroll if user is near bottom when new message arrives
            setTimeout(() => {
              if (messagesContainerRef.current && isMounted) {
                const isNear = isNearBottom();
                shouldAutoScrollRef.current = isNear;
                if (isNear && messagesEndRef.current) {
                  messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
                }
              }
            }, 100);
            
            // Trigger chat history refresh
            if (onMessageSent) {
              onMessageSent();
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        ws.onclose = () => {
          if (!isMounted) return;
          console.log('WebSocket disconnected');
          setIsConnected(false);
          // Attempt to reconnect after 3 seconds
          reconnectTimeout = setTimeout(() => {
            if (isMounted) {
              connectWebSocket();
            }
          }, 3000);
        };

        ws.onerror = (error) => {
          if (!isMounted) return;
          console.error('WebSocket error:', error);
          setIsConnected(false);
        };

        wsRef.current = ws;
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        setIsConnected(false);
        if (isMounted) {
          reconnectTimeout = setTimeout(() => {
            if (isMounted) {
              connectWebSocket();
            }
          }, 3000);
        }
      }
    };

    connectWebSocket();

    return () => {
      isMounted = false;
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [sessionId]); // Only depend on sessionId to avoid reconnection loops

  // handleNewChat is now handled by the useEffect when conversationId becomes undefined
  // This function is kept for backwards compatibility but the main logic is in useEffect

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      status: 'pending',
      timestamp: new Date(),
      isUser: true
    };

    setMessages(prev => [...prev, newMessage]);
    const userInput = inputValue;
    setInputValue('');
    
    // Force scroll to bottom when user sends a message
    shouldAutoScrollRef.current = true;
    setTimeout(() => scrollToBottom(true), 50);

    try {
      // Send message to Kafka via the ingress endpoint
      const response = await fetch(`${API_BASE_URL}/publish/ingress`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          session_id: sessionId,
          query_text: userInput,
          correlation_id: `corr-${Date.now()}`,
          event_timestamp: Date.now(),
          user_email: userProfile?.profile?.email || null,
          conversation_id: conversationId || null
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Message published to Kafka:', data);
      
      // Update conversation_id if returned from backend
      // This happens when the first message creates a new conversation
      if (data.conversation_id) {
        if (!conversationId) {
          // Update the ref BEFORE setting conversationId to prevent history reload
          // This ensures we don't clear the current messages when conversationId is set
          if (messagesRef.current.length > 0) {
            lastLoadedConversationRef.current = data.conversation_id;
            prevConversationIdRef.current = undefined; // Set previous to undefined so transition is detected correctly
          }
          skipNextHistoryLoadRef.current = true;
          setConversationId(data.conversation_id);
          // Notify parent component that a new conversation was created
          if (onConversationCreated) {
            onConversationCreated(data.conversation_id, sessionId);
          }
        }
      }
      
      // Update user message status to fulfilled (message was sent successfully)
      setMessages(prev => 
        prev.map(msg => 
          msg.id === newMessage.id 
            ? { ...msg, status: 'fulfilled' as MessageStatus }
            : msg
        )
      );
      
      // Trigger chat history refresh
      if (onMessageSent) {
        onMessageSent();
      }

      // The actual response will come through WebSocket
      // Add a placeholder message indicating we're waiting for response
      const waitingMessage: Message = {
        id: `waiting-${Date.now()}`,
        content: 'Processing your request...',
        status: 'pending',
        timestamp: new Date(),
        isUser: false
      };

      setMessages(prev => [...prev, waitingMessage]);

    } catch (error) {
      console.error('Error sending message:', error);
      
      // Update user message status to unfulfilled
      setMessages(prev => 
        prev.map(msg => 
          msg.id === newMessage.id 
            ? { ...msg, status: 'unfulfilled' as MessageStatus }
            : msg
        )
      );

      // Add error message
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, I encountered an error. Please try again or check if the backend is running.',
        status: 'unfulfilled',
        timestamp: new Date(),
        isUser: false
      };

      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const getStatusIcon = (status: MessageStatus) => {
    switch (status) {
      case 'pending':
        return <Clock className="h-3 w-3" />;
      case 'fulfilled':
        return <CheckCircle className="h-3 w-3" />;
      case 'unfulfilled':
        return <XCircle className="h-3 w-3" />;
    }
  };

  const getStatusColor = (status: MessageStatus) => {
    switch (status) {
      case 'pending':
        return 'bg-yellow-500/10 text-yellow-600';
      case 'fulfilled':
        return 'bg-green-500/10 text-green-600';
      case 'unfulfilled':
        return 'bg-red-500/10 text-red-600';
    }
  };

  // Remove waiting messages when we get actual responses
  // Use a ref to track if we've already cleaned up to avoid infinite loops
  // Removed the cleanup effect - we now handle "Processing your request..." removal
  // directly in the WebSocket message handler to prevent double messages

  return (
    <div className="flex flex-col h-full">
      {/* Connection Status */}
      <div className="px-4 py-2 border-b bg-muted/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-600' : 'bg-red-600'}`} />
            <span className="text-sm text-muted-foreground">
              {isConnected ? 'Connected to Agent Stack' : 'Connecting...'}
            </span>
            {loadingHistory && (
              <span className="text-xs text-muted-foreground">Loading history...</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {conversationId && (
              <span className="text-xs text-muted-foreground">Conversation: {conversationId.slice(-8)}</span>
            )}
            <span className="text-xs text-muted-foreground">Session: {sessionId.slice(-8)}</span>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div 
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-4 space-y-4"
      >
        {loadingHistory ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground mb-4" />
            <p className="text-muted-foreground">Loading conversation history...</p>
          </div>
        ) : messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Bot className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">Welcome back, {userName?.split(' ')[0] || 'User'}!</h3>
            <p className="text-muted-foreground mb-4">
              I can help you with orders, policies, general questions, and more!
            </p>
            <div className="text-sm text-muted-foreground">
              <p>Try asking:</p>
              <p>• "What's my order status?"</p>
              <p>• "What's your return policy?"</p>
              <p>• "I need help with my account"</p>
            </div>
          </div>
        )}
        
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-3 ${
                message.isUser
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted'
              }`}
            >
              {!message.isUser && message.agentType && (
                <div className="flex items-center mb-2">
                  <Bot className="h-3 w-3 mr-1" />
                  <span className="text-xs font-medium opacity-70">
                    {message.agentType.replace('_AGENT', '')} Agent
                  </span>
                </div>
              )}
              <p className="mb-2 whitespace-pre-wrap">{message.content}</p>
              <div className="flex items-center justify-between">
                <span className="text-xs opacity-70">
                  {message.timestamp.toLocaleTimeString()}
                </span>
                {message.isUser && (
                  <Badge variant="secondary" className={`ml-2 ${getStatusColor(message.status)}`}>
                    {getStatusIcon(message.status)}
                    <span className="ml-1 capitalize">{message.status}</span>
                  </Badge>
                )}
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t bg-background">
        <div className="flex justify-center">
          <div className="flex gap-2 items-center w-full max-w-2xl bg-input-background rounded-full p-2 border">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={isConnected ? "Type your message..." : "Connecting to agents..."}
              onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
              disabled={!isConnected}
              className="flex-1 min-w-0 border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0"
            />
            <Button 
              onClick={handleSendMessage} 
              size="icon" 
              className="shrink-0 rounded-full"
              disabled={!isConnected || !inputValue.trim()}
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <div className="text-center mt-2">
          <p className="text-xs text-muted-foreground">
            Press Enter to send • Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
}

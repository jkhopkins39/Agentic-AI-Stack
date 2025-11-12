import React, { useState, useEffect, useRef } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Send, Clock, CheckCircle, XCircle, Bot, User, Loader2 } from 'lucide-react';
import { useUser } from '../contexts/UserContext';

type MessageStatus = 'pending' | 'fulfilled' | 'unfulfilled';

interface Message {
  id: string;
  content: string;
  status: MessageStatus;
  timestamp: Date;
  isUser: boolean;
  agentType?: string;
  isTyping?: boolean;
}

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [sessionId] = useState(() => `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`);
  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { userProfile } = useUser();

  // Get user name for display
  const userName = userProfile?.profile 
    ? `${userProfile.profile.first_name || ''} ${userProfile.profile.last_name || ''}`.trim() || userProfile.profile.email
    : 'User';

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // WebSocket connection for real-time responses
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(`ws://localhost:8000/ws/agent-responses/${sessionId}`);
        
        ws.onopen = () => {
          console.log('🔌 [FRONTEND] WebSocket connected', { sessionId });
          setIsConnected(true);
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log(' [FRONTEND] Received WebSocket message:', {
              sessionId: data.session_id,
              agentType: data.agent_type,
              messageLength: data.message?.length,
              timestamp: data.timestamp,
              fullData: data
            });
            
            // Remove "Processing..." placeholder message when we get actual response
            setMessages(prev => {
              // Filter out "Processing..." messages
              const filtered = prev.filter(msg => 
                !(msg.isTyping && !msg.isUser)
              );
              
              // Add agent response to messages
              const agentMessage: Message = {
                id: `agent-${Date.now()}`,
                content: data.message || 'Response received',
                status: 'fulfilled',
                timestamp: new Date(),
                isUser: false,
                agentType: data.agent_type || 'Agent'
              };

              console.log(' [FRONTEND] Adding message to UI:', agentMessage);
              return [...filtered, agentMessage];
            });
          } catch (error) {
            console.error(' [FRONTEND] Error parsing WebSocket message:', error, event.data);
          }
        };

        ws.onclose = () => {
          console.log(' [FRONTEND] WebSocket disconnected', { sessionId });
          setIsConnected(false);
          // Attempt to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (error) => {
          console.error(' [FRONTEND] WebSocket error:', error, { sessionId });
          setIsConnected(false);
        };

        wsRef.current = ws;
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        setIsConnected(false);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [sessionId]);

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

    try {
      const requestPayload = { 
        session_id: sessionId,
        query_text: userInput,
        correlation_id: `corr-${Date.now()}`,
        event_timestamp: Date.now(),
        user_email: userProfile?.profile?.email || null
      };
      
      console.log(' [FRONTEND] Sending message to backend:', {
        sessionId,
        queryText: userInput,
        userEmail: userProfile?.profile?.email,
        payload: requestPayload
      });
      
      // Send message to Kafka via the ingress endpoint
      const response = await fetch('http://localhost:8000/publish/ingress', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestPayload),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(' [FRONTEND] HTTP error:', {
          status: response.status,
          statusText: response.statusText,
          error: errorText
        });
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log(' [FRONTEND] Message published to Kafka:', data);
      
      // Update user message status to fulfilled (message was sent successfully)
      setMessages(prev => 
        prev.map(msg => 
          msg.id === newMessage.id 
            ? { ...msg, status: 'fulfilled' as MessageStatus }
            : msg
        )
      );

      // The actual response will come through WebSocket
      // Add a placeholder message indicating we're waiting for response
      const waitingMessage: Message = {
        id: `waiting-${Date.now()}`,
        content: '',
        status: 'pending',
        timestamp: new Date(),
        isUser: false,
        isTyping: true
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

  // Note: "Processing..." messages are now removed when WebSocket receives actual response
  // This prevents infinite loop issues

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
          </div>
          <span className="text-xs text-muted-foreground">Session: {sessionId.slice(-8)}</span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
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
              {message.isTyping ? (
                <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Thinking...</span>
                </div>
              ) : (
                <p className="mb-2 whitespace-pre-wrap">{message.content}</p>
              )}
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
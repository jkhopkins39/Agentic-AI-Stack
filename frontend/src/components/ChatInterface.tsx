import React, { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  agentUsed?: string;
  timestamp: Date;
}

interface OrderData {
  order_number?: string;
  status?: string;
  total_amount?: number;
  [key: string]: any;
}

interface ChatResponse {
  response: string;
  agent_used: string;
  order_data?: OrderData;
}

const API_BASE_URL = 'http://localhost:8000';

export const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Hello! I\'m your AI assistant. I can help you with orders, policies, emails, and general questions. How can I assist you today?',
      sender: 'assistant',
      agentUsed: 'System',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post<ChatResponse>(`${API_BASE_URL}/chat`, {
        message: inputMessage,
        user_id: null //implement user auth later
      });

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.data.response,
        sender: 'assistant',
        agentUsed: response.data.agent_used,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);

      // If there's order data, show it
      if (response.data.order_data && Object.keys(response.data.order_data).length > 0) {
        const orderMessage: Message = {
          id: (Date.now() + 2).toString(),
          content: `Order Information: ${JSON.stringify(response.data.order_data, null, 2)}`,
          sender: 'assistant',
          agentUsed: 'Order System',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, orderMessage]);
      }

    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, there was an error processing your request. Please try again.',
        sender: 'assistant',
        agentUsed: 'Error Handler',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header with back button */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link 
              to="/" 
              className="text-blue-600 hover:text-blue-700 font-medium flex items-center"
            >
              ← Back to Home
            </Link>
            <h1 className="text-xl font-bold text-gray-800">AI Assistant Chat</h1>
            <div></div> {/* Spacer for center alignment */}
          </div>
        </div>
      </div>

      <div className="flex flex-col h-[calc(100vh-80px)] max-w-4xl mx-auto p-4">
        {/* Messages Container */}
        <div className="flex-1 bg-white rounded-lg shadow-sm border overflow-y-auto p-6 space-y-4">
          {messages.map(message => (
            <div key={message.id} className="flex flex-col">
              <div className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`
                    max-w-xs lg:max-w-md px-4 py-3 rounded-2xl shadow-sm
                    ${message.sender === 'user' 
                      ? 'bg-blue-600 text-white rounded-br-sm' 
                      : 'bg-gray-100 text-gray-800 rounded-bl-sm'
                    }
                  `}
                >
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">
                    {message.content}
                  </p>
                  
                  {message.agentUsed && message.sender === 'assistant' && (
                    <div className="mt-2">
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                        {message.agentUsed}
                      </span>
                    </div>
                  )}
                </div>
              </div>
              
              <div className={`text-xs text-gray-500 mt-1 ${
                message.sender === 'user' ? 'text-right' : 'text-left'
              }`}>
                {message.timestamp.toLocaleTimeString()}
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-100 text-gray-800 px-4 py-3 rounded-2xl rounded-bl-sm shadow-sm max-w-xs">
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                  </div>
                  <span className="text-sm text-gray-600">Processing...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Container */}
        <div className="bg-white rounded-lg shadow-sm border border-t-0 p-6">
          <div className="flex space-x-4">
            <div className="flex-1 relative">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message here..."
                disabled={isLoading}
                className="
                  w-full px-4 py-3 pr-12 rounded-full border border-gray-300 
                  focus:ring-2 focus:ring-blue-500 focus:border-transparent
                  disabled:bg-gray-100 disabled:cursor-not-allowed
                  text-gray-900 placeholder-gray-500
                "
              />
              {inputMessage.trim() && (
                <div className="absolute inset-y-0 right-0 flex items-center pr-3">
                  <button
                    onClick={sendMessage}
                    disabled={isLoading || !inputMessage.trim()}
                    className="
                      p-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 
                      focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
                      disabled:bg-gray-400 disabled:cursor-not-allowed
                      transition-colors duration-200
                    "
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  </button>
                </div>
              )}
            </div>
            
            {!inputMessage.trim() && (
              <button
                onClick={sendMessage}
                disabled={isLoading || !inputMessage.trim()}
                className="
                  px-6 py-3 bg-blue-600 text-white font-medium rounded-full
                  hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
                  disabled:bg-gray-400 disabled:cursor-not-allowed
                  transition-colors duration-200
                "
              >
                Send
              </button>
            )}
          </div>
          
          {/* Quick Actions */}
          <div className="mt-4 flex flex-wrap gap-2">
            {[
              'Check order ORD-2024-001',
              'What\'s your return policy?',
              'Find orders for john.doe@email.com',
              'How can you help me?'
            ].map((suggestion, index) => (
              <button
                key={index}
                onClick={() => setInputMessage(suggestion)}
                disabled={isLoading}
                className="
                  px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-full
                  hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-400
                  disabled:opacity-50 disabled:cursor-not-allowed
                  transition-colors duration-200
                "
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
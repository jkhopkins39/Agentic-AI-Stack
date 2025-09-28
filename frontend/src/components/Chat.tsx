import React, { useState, useEffect, useRef } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Send, Clock, CheckCircle, XCircle, Bot, User, AlertCircle } from 'lucide-react';

// LangGraph types and interfaces
type MessageStatus = 'pending' | 'fulfilled' | 'unfulfilled' | 'streaming';

interface Message {
  id: string;
  content: string;
  status: MessageStatus;
  timestamp: Date;
  isUser: boolean;
  agentType?: string;
  metadata?: Record<string, any>;
}

interface LangGraphConfig {
  apiUrl: string;
  assistantId: string;
  enableStreaming?: boolean;
  enableMultiAgent?: boolean;
  tools?: string[];
}

interface LangGraphResponse {
  content: string;
  agent?: string;
  status: 'complete' | 'partial' | 'error';
  metadata?: Record<string, any>;
}

function useLangGraph(config: LangGraphConfig) {
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => setIsConnected(true), 1000);
    return () => clearTimeout(timer);
  }, []);

  const submitMessage = async (message: string): Promise<LangGraphResponse> => {
    setIsLoading(true);
    setError(null);
    
    try {

      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Mock multi-agent response
      const agents = ['planner', 'researcher', 'executor'];
      const selectedAgent = agents[Math.floor(Math.random() * agents.length)];
      
      const responses = {
        planner: `Planning approach for: "${message}". I'll break this down into actionable steps.`,
        researcher: `Researching information about: "${message}". Let me gather relevant data and insights.`,
        executor: `Executing task: "${message}". I'll complete this systematically and provide results.`
      };
      
      return {
        content: responses[selectedAgent as keyof typeof responses],
        agent: selectedAgent,
        status: Math.random() > 0.1 ? 'complete' : 'error',
        metadata: {
          processingTime: Math.random() * 2000 + 500,
          confidence: Math.random() * 0.3 + 0.7
        }
      };
    } catch (err) {
      setError('Failed to connect to LangGraph service');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const streamMessage = async function* (message: string): AsyncGenerator<LangGraphResponse, void, unknown> {
    setIsLoading(true);
    const words = `Streaming response for: "${message}". This demonstrates real-time AI agent processing with LangGraph integration.`.split(' ');
    
    for (let i = 0; i < words.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      yield {
        content: words.slice(0, i + 1).join(' '),
        status: i === words.length - 1 ? 'complete' : 'partial'
      };
    }
    setIsLoading(false);
  };

  return {
    isConnected,
    isLoading,
    error,
    submitMessage,
    streamMessage
  };
}

interface ChatProps {
  langGraphConfig: LangGraphConfig;
}

export function Chat({ langGraphConfig }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      content: 'Hello! I\'m powered by LangGraph AI agents. I can help you with planning, research, and task execution. What would you like to work on?',
      status: 'fulfilled',
      timestamp: new Date(),
      isUser: false,
      agentType: 'system'
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const langGraph = useLangGraph(langGraphConfig);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || langGraph.isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content: inputValue,
      status: 'fulfilled',
      timestamp: new Date(),
      isUser: true
    };

    setMessages(prev => [...prev, userMessage]);
    const messageContent = inputValue;
    setInputValue('');

    // Create AI response message
    const aiMessageId = `ai-${Date.now()}`;
    const aiMessage: Message = {
      id: aiMessageId,
      content: '',
      status: 'streaming',
      timestamp: new Date(),
      isUser: false
    };

    setMessages(prev => [...prev, aiMessage]);

    try {
      if (langGraphConfig.enableStreaming) {
        // Handle streaming response
        for await (const chunk of langGraph.streamMessage(messageContent)) {
          setMessages(prev => 
            prev.map(msg => 
              msg.id === aiMessageId 
                ? { 
                    ...msg, 
                    content: chunk.content,
                    status: chunk.status === 'complete' ? 'fulfilled' : 'streaming'
                  }
                : msg
            )
          );
        }
      } else {
        // Handle single response
        const response = await langGraph.submitMessage(messageContent);
        
        setMessages(prev =>
          prev.map(msg =>
            msg.id === aiMessageId
              ? {
                  ...msg,
                  content: response.content,
                  status: response.status === 'complete' ? 'fulfilled' : 'unfulfilled',
                  agentType: response.agent,
                  metadata: response.metadata
                }
              : msg
          )
        );
      }
    } catch (error) {
      setMessages(prev =>
        prev.map(msg =>
          msg.id === aiMessageId
            ? {
                ...msg,
                content: 'Sorry, I encountered an error processing your request. Please try again.',
                status: 'unfulfilled'
              }
            : msg
        )
      );
    }
  };

  const getStatusIcon = (status: MessageStatus) => {
    switch (status) {
      case 'pending':
      case 'streaming':
        return <Clock className="h-3 w-3 animate-spin" />;
      case 'fulfilled':
        return <CheckCircle className="h-3 w-3" />;
      case 'unfulfilled':
        return <XCircle className="h-3 w-3" />;
    }
  };

  const getStatusColor = (status: MessageStatus) => {
    switch (status) {
      case 'pending':
      case 'streaming':
        return 'bg-blue-100 text-blue-800';
      case 'fulfilled':
        return 'bg-green-100 text-green-800';
      case 'unfulfilled':
        return 'bg-red-100 text-red-800';
    }
  };

  const getAgentIcon = (agentType?: string) => {
    if (!agentType) return <Bot className="h-4 w-4" />;
    
    switch (agentType) {
      case 'planner':
        return <Bot className="h-4 w-4 text-blue-600" />;
      case 'researcher':
        return <Bot className="h-4 w-4 text-green-600" />;
      case 'executor':
        return <Bot className="h-4 w-4 text-purple-600" />;
      default:
        return <Bot className="h-4 w-4" />;
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Connection Status */}
      <div className="p-2 border-b bg-muted/50">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${langGraph.isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span>LangGraph {langGraph.isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
          {langGraphConfig.enableMultiAgent && (
            <Badge variant="secondary" className="text-xs">
              Multi-Agent Enabled
            </Badge>
          )}
        </div>
        {langGraph.error && (
          <div className="flex items-center gap-1 text-red-600 text-xs mt-1">
            <AlertCircle className="h-3 w-3" />
            {langGraph.error}
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div key={message.id} className="flex justify-center">
            <div
              className={`max-w-[80%] rounded-lg p-3 ${
                message.isUser
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted'
              }`}
            >
              {/* Message Header */}
              {!message.isUser && message.agentType && message.agentType !== 'system' && (
                <div className="flex items-center gap-2 mb-2 text-xs opacity-70">
                  {getAgentIcon(message.agentType)}
                  <span className="capitalize">{message.agentType} Agent</span>
                </div>
              )}

              {/* Message Content */}
              <p className="mb-2">{message.content}</p>

              {/* Message Footer */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {message.isUser ? (
                    <User className="h-3 w-3 opacity-70" />
                  ) : (
                    getAgentIcon(message.agentType)
                  )}
                  <span className="text-xs opacity-70">
                    {message.timestamp.toLocaleTimeString()}
                  </span>
                </div>

                {/* Status Badge */}
                <Badge variant="secondary" className={`ml-2 ${getStatusColor(message.status)}`}>
                  {getStatusIcon(message.status)}
                  <span className="ml-1 capitalize">
                    {message.status === 'streaming' ? 'typing...' : message.status}
                  </span>
                </Badge>
              </div>

              {/* Metadata */}
              {message.metadata && (
                <div className="mt-2 pt-2 border-t border-border/20">
                  <div className="text-xs opacity-50 space-y-1">
                    {message.metadata.confidence && (
                      <div>Confidence: {(message.metadata.confidence * 100).toFixed(1)}%</div>
                    )}
                    {message.metadata.processingTime && (
                      <div>Processing: {message.metadata.processingTime.toFixed(0)}ms</div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t">
        <div className="flex justify-center">
          <div className="flex gap-2 items-center w-3/5 bg-input-background rounded-full p-2 border">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={langGraph.isLoading ? "AI is thinking..." : "Ask your AI agents..."}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              disabled={!langGraph.isConnected || langGraph.isLoading}
              className="flex-1 min-w-0 border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0"
            />
            <Button 
              onClick={handleSendMessage} 
              size="icon" 
              className="shrink-0 rounded-full"
              disabled={!langGraph.isConnected || langGraph.isLoading || !inputValue.trim()}
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Usage example:
export function ChatWithLangGraph() {
  const langGraphConfig: LangGraphConfig = {
    apiUrl: process.env.REACT_APP_LANGGRAPH_API_URL || 'http://localhost:8000',
    assistantId: 'multi-agent-assistant',
    enableStreaming: true,
    enableMultiAgent: true,
    tools: ['web_search', 'calculator', 'file_manager']
  };

  return <Chat langGraphConfig={langGraphConfig} />;
}
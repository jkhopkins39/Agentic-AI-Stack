import React, { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Send, Clock, CheckCircle, XCircle } from 'lucide-react';

type MessageStatus = 'pending' | 'fulfilled' | 'unfulfilled';

interface Message {
  id: string;
  content: string;
  status: MessageStatus;
  timestamp: Date;
  isUser: boolean;
}

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');

  const handleSendMessage = () => {
    if (!inputValue.trim()) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      status: 'pending',
      timestamp: new Date(),
      isUser: true
    };

    setMessages(prev => [...prev, newMessage]);
    setInputValue('');

    // Simulate processing
    setTimeout(() => {
      setMessages(prev => 
        prev.map(msg => 
          msg.id === newMessage.id 
            ? { ...msg, status: (Math.random() > 0.3 ? 'fulfilled' : 'unfulfilled') as MessageStatus }
            : msg
        )
      );

      // Bot response logic would be implemented here
    }, 2000);
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
        return 'bg-yellow-100 text-yellow-800';
      case 'fulfilled':
        return 'bg-green-100 text-green-800';
      case 'unfulfilled':
        return 'bg-red-100 text-red-800';
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className="flex justify-center"
          >
            <div
              className={`max-w-[80%] rounded-lg p-3 ${
                message.isUser
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted'
              }`}
            >
              <p className="mb-2">{message.content}</p>
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
      </div>

      {/* Input */}
      <div className="p-4 border-t">
        <div className="flex justify-center">
          <div className="flex gap-2 items-center w-3/5 bg-input-background rounded-full p-2 border">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Type your message..."
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              className="flex-1 min-w-0 border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0"
            />
            <Button onClick={handleSendMessage} size="icon" className="shrink-0 rounded-full">
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
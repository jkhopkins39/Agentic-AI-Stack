import React, { useState } from 'react';
import { Button } from './ui/button';
import { Plus, MessageSquare, X } from 'lucide-react';

interface ChatSession {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
}

interface ChatHistoryProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ChatHistory({ isOpen, onClose }: ChatHistoryProps) {
  const [chatSessions] = useState<ChatSession[]>([
    {
      id: '1',
      title: 'Product Return Question',
      lastMessage: 'Thank you for your help!',
      timestamp: new Date('2024-01-15T10:30:00'),
    },
    {
      id: '2',
      title: 'Shipping Inquiry',
      lastMessage: 'When will my order arrive?',
      timestamp: new Date('2024-01-14T15:45:00'),
    },
    {
      id: '3',
      title: 'Account Support',
      lastMessage: 'I need to update my address',
      timestamp: new Date('2024-01-13T09:15:00'),
    },
  ]);

  const handleNewChat = () => {
    console.log('Starting new chat');
  };

  const handleChatSelect = (chatId: string) => {
    console.log('Selected chat:', chatId);
  };

  const formatRelativeTime = (date: Date) => {
    const now = new Date();
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'Just now';
    if (diffInHours < 24) return `${diffInHours}h ago`;
    const diffInDays = Math.floor(diffInHours / 24);
    if (diffInDays < 7) return `${diffInDays}d ago`;
    return date.toLocaleDateString();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed top-20 right-0 h-[calc(100vh-5rem)] w-80 bg-white border-l border-gray-200 shadow-lg z-50 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            <span className="font-medium">Chat History</span>
          </div>
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={onClose}
            className="h-8 w-8 p-0"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        <Button onClick={handleNewChat} className="w-full">
          <Plus className="h-4 w-4 mr-2" />
          New Chat
        </Button>
      </div>

      {/* Chat Sessions List */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-2">
          {chatSessions.map((session) => (
            <button
              key={session.id}
              onClick={() => handleChatSelect(session.id)}
              className="w-full p-3 mb-2 text-left rounded-lg hover:bg-muted transition-colors group"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <h4 className="font-medium text-sm text-foreground truncate">
                    {session.title}
                  </h4>
                  <p className="text-xs text-muted-foreground mt-1 truncate">
                    {session.lastMessage}
                  </p>
                </div>
                <span className="text-xs text-muted-foreground ml-2 flex-shrink-0">
                  {formatRelativeTime(session.timestamp)}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
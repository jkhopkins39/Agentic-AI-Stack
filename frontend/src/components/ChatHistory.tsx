import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Plus, MessageSquare, X, Loader2, Trash2 } from 'lucide-react';
import { useUser } from '../contexts/UserContext';
import { API_BASE_URL } from '../config';

interface ChatSession {
  conversation_id: string;
  session_id: string;
  last_message: string | null;
  message_count: number;
  created_at: string;
  updated_at: string;
}

interface ChatHistoryProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectConversation?: (conversationId: string, sessionId: string) => void;
  onNewChat?: () => void;
  refreshTrigger?: number;
}

export function ChatHistory({ isOpen, onClose, onSelectConversation, onNewChat, refreshTrigger }: ChatHistoryProps) {
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [loading, setLoading] = useState(false);
  const { userProfile } = useUser();

  const fetchConversations = async () => {
    if (!userProfile?.profile?.email) return;
    
    setLoading(true);
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/conversations?user_email=${encodeURIComponent(userProfile.profile.email)}&limit=20`
      );
      const data = await response.json();
      setChatSessions(data.conversations || []);
    } catch (error) {
      console.error('Error fetching conversations:', error);
      setChatSessions([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen && userProfile?.profile?.email) {
      fetchConversations();
    }
  }, [isOpen, userProfile?.profile?.email, refreshTrigger]);

  const handleNewChat = () => {
    if (onNewChat) {
      onNewChat();
    }
    // Refresh the conversation list after a short delay to show the new state
    setTimeout(() => {
      fetchConversations();
    }, 100);
    onClose();
  };

  const handleChatSelect = (conversationId: string, sessionId: string) => {
    if (onSelectConversation) {
      onSelectConversation(conversationId, sessionId);
    }
    onClose();
  };

  const handleDeleteChat = async (e: React.MouseEvent, conversationId: string) => {
    e.stopPropagation(); // Prevent triggering the chat select
    
    if (!window.confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) {
      return;
    }

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/conversations/${conversationId}?user_email=${encodeURIComponent(userProfile?.profile?.email || '')}`,
        {
          method: 'DELETE',
        }
      );

      if (!response.ok) {
        throw new Error('Failed to delete conversation');
      }

      // Refresh the conversation list
      fetchConversations();
      
      // If the deleted conversation was selected, clear the selection
      if (onNewChat) {
        onNewChat();
      }
    } catch (error) {
      console.error('Error deleting conversation:', error);
      alert('Failed to delete conversation. Please try again.');
    }
  };

  const formatRelativeTime = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'Just now';
    if (diffInHours < 24) return `${diffInHours}h ago`;
    const diffInDays = Math.floor(diffInHours / 24);
    if (diffInDays < 7) return `${diffInDays}d ago`;
    return date.toLocaleDateString();
  };

  const getTitle = (session: ChatSession) => {
    if (session.last_message) {
      // Use first 30 chars of last message as title
      return session.last_message.length > 30 
        ? session.last_message.substring(0, 30) + '...'
        : session.last_message;
    }
    return `Chat ${session.message_count > 0 ? `(${session.message_count} messages)` : '(New)'}`;
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
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : chatSessions.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground text-sm">
              No chat history yet. Start a new conversation!
            </div>
          ) : (
            chatSessions.map((session) => (
              <div
                key={session.conversation_id}
                className="w-full mb-2 rounded-lg hover:bg-muted transition-colors group relative"
              >
                <button
                  onClick={() => handleChatSelect(session.conversation_id, session.session_id)}
                  className="w-full p-3 text-left"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0 pr-8">
                      <h4 className="font-medium text-sm text-foreground truncate">
                        {getTitle(session)}
                      </h4>
                      {session.last_message && (
                        <p className="text-xs text-muted-foreground mt-1 truncate">
                          {session.last_message}
                        </p>
                      )}
                      {session.message_count > 0 && (
                        <p className="text-xs text-muted-foreground mt-1">
                          {session.message_count} message{session.message_count !== 1 ? 's' : ''}
                        </p>
                      )}
                    </div>
                    <span className="text-xs text-muted-foreground ml-2 flex-shrink-0">
                      {formatRelativeTime(session.updated_at)}
                    </span>
                  </div>
                </button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => handleDeleteChat(e, session.conversation_id)}
                  className="absolute top-2 right-2 h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity text-red-600 hover:text-red-700 hover:bg-red-50"
                  title="Delete conversation"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

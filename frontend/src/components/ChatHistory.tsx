import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Plus, MessageSquare, Loader2, Trash2 } from 'lucide-react';
import { useUser } from '../contexts/UserContext';
import { API_BASE_URL } from '../config';
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
} from './ui/sidebar';

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
        `http://localhost:8000/api/conversations?user_email=${encodeURIComponent(userProfile.profile.email)}&limit=20`
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
        `http://localhost:8000/api/conversations/${conversationId}?user_email=${encodeURIComponent(userProfile?.profile?.email || '')}`,
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

  return (
    <Sidebar side="right" collapsible="offcanvas" className="border-l">
      <SidebarHeader className="p-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
            <MessageSquare className="h-4 w-4 text-blue-600" />
          </div>
          <div>
            <span className="font-semibold text-gray-900">Chat History</span>
            <p className="text-xs text-gray-600">View and manage conversations</p>
          </div>
        </div>
      </SidebarHeader>

      <SidebarContent className="overflow-y-auto">
        {/* New Chat Button */}
        <div className="p-4">
          <Button onClick={handleNewChat} className="w-full bg-blue-600 hover:bg-blue-700 rounded-lg shadow-md">
            <Plus className="h-4 w-4 mr-2" />
            New Chat
          </Button>
        </div>

        {/* Chat Sessions List */}
        <SidebarGroup>
          <SidebarGroupContent className="p-4">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
              </div>
            ) : chatSessions.length === 0 ? (
              <div className="text-center py-8 text-gray-500 text-sm">
                No chat history yet. Start a new conversation!
              </div>
            ) : (
              <div className="space-y-2">
                {chatSessions.map((session) => (
                  <div
                    key={session.conversation_id}
                    className="w-full rounded-xl hover:bg-gray-50 transition-colors group relative border border-gray-200 shadow-sm"
                  >
                    <button
                      onClick={() => handleChatSelect(session.conversation_id, session.session_id)}
                      className="w-full p-3 text-left"
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0 pr-8">
                          <h4 className="font-medium text-sm text-gray-900 truncate">
                            {getTitle(session)}
                          </h4>
                          {session.last_message && (
                            <p className="text-xs text-gray-600 mt-1 truncate">
                              {session.last_message}
                            </p>
                          )}
                          {session.message_count > 0 && (
                            <p className="text-xs text-gray-500 mt-1">
                              {session.message_count} message{session.message_count !== 1 ? 's' : ''}
                            </p>
                          )}
                        </div>
                        <span className="text-xs text-gray-500 ml-2 flex-shrink-0">
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
                ))}
              </div>
            )}
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}

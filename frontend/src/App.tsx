import React, { useState, useEffect } from 'react';
import { UserProvider, useUser } from './contexts/UserContext';
import { CustomerSidebar } from './components/CustomerSidebar';
import { Chat } from './components/Chat';
import { ChatHistory } from './components/ChatHistory';
import { Login } from './components/Login';
import AdminDashboard from './components/AdminDashboard';

function AppContent() {
  const [isChatHistoryOpen, setIsChatHistoryOpen] = useState(false);
  const [currentView, setCurrentView] = useState<'chat' | 'admin'>('chat');
  const [selectedConversationId, setSelectedConversationId] = useState<string | undefined>();
  const [selectedSessionId, setSelectedSessionId] = useState<string | undefined>();
  const [chatHistoryRefresh, setChatHistoryRefresh] = useState(0);
  const { isAuthenticated, login, userProfile, currentUserEmail } = useUser();

  // Determine if current user has admin access
  const isAdmin = userProfile?.profile?.is_admin === true;

  // Reset app state when user changes (but not on initial mount)
  const prevUserEmailRef = React.useRef<string | null>(null);
  useEffect(() => {
    // Only reset if we're switching from one user to another (not initial mount)
    if (prevUserEmailRef.current !== null && prevUserEmailRef.current !== currentUserEmail) {
      // Reset view to chat, clear selected conversations
      setCurrentView('chat');
      setSelectedConversationId(undefined);
      setSelectedSessionId(undefined);
      setChatHistoryRefresh(prev => prev + 1);
    }
    prevUserEmailRef.current = currentUserEmail;
  }, [currentUserEmail]);

  // Redirect non-admin users away from admin dashboard
  useEffect(() => {
    if (!isAdmin && currentView === 'admin') {
      setCurrentView('chat');
    }
  }, [isAdmin, currentView]);

  if (!isAuthenticated) {
    return <Login onLogin={login} />;
  }

  return (
    <div className="h-screen">
      <CustomerSidebar 
        chatHistoryOpen={isChatHistoryOpen}
        onChatHistoryToggle={setIsChatHistoryOpen}
        selectedConversationId={selectedConversationId}
        selectedSessionId={selectedSessionId}
        onSelectConversation={(conversationId, sessionId) => {
          setSelectedConversationId(conversationId);
          setSelectedSessionId(sessionId);
        }}
        onNewChat={() => {
          setSelectedConversationId(undefined);
          setSelectedSessionId(undefined);
          setChatHistoryRefresh(prev => prev + 1);
        }}
        chatHistoryRefresh={chatHistoryRefresh}
        isAdmin={isAdmin}
        onAdminToggle={() => setCurrentView(currentView === 'chat' ? 'admin' : 'chat')}
        currentView={currentView}
      >
        <div className="relative h-full">
          {currentView === 'chat' ? (
            <Chat 
              conversationId={selectedConversationId}
              sessionId={selectedSessionId}
              onNewChat={() => {
                setSelectedConversationId(undefined);
                setSelectedSessionId(undefined);
              }}
              onMessageSent={() => {
                setChatHistoryRefresh(prev => prev + 1);
              }}
              onConversationCreated={(conversationId, sessionId) => {
                // Update the selected conversation when a new one is created
                // This ensures the chat stays in the current conversation instead of creating a new one
                setSelectedConversationId(conversationId);
                setSelectedSessionId(sessionId);
              }}
            />
          ) : (
            <AdminDashboard />
          )}
        </div>
      </CustomerSidebar>
    </div>
  );
}

export default function App() {
  return (
    <UserProvider>
      <AppContent />
    </UserProvider>
  );
}

import React, { useState } from 'react';
import { UserProvider, useUser } from './contexts/UserContext';
import { CustomerSidebar } from './components/CustomerSidebar';
import { Chat } from './components/Chat';
import { ChatHistory } from './components/ChatHistory';
import { Login } from './components/Login';
import { Button } from './components/ui/button';
import { MessageSquare } from 'lucide-react';

function AppContent() {
  const [isChatHistoryOpen, setIsChatHistoryOpen] = useState(false);
  const { isAuthenticated, login } = useUser();

  if (!isAuthenticated) {
    return <Login onLogin={login} />;
  }

  return (
    <div className="h-screen">
      <CustomerSidebar>
        <div className="relative h-full">
          <Chat />
          
          {/* Chat History Toggle Button */}
          <Button
            onClick={() => setIsChatHistoryOpen(true)}
            className="fixed top-4 right-4 z-40"
            size="sm"
          >
            <MessageSquare className="h-4 w-4 mr-2" />
            Chat History
          </Button>


          {/* Chat History Sidebar */}
          <ChatHistory 
            isOpen={isChatHistoryOpen} 
            onClose={() => setIsChatHistoryOpen(false)} 
          />
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
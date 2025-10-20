import React, { useState } from 'react';
import { UserProvider } from './contexts/UserContext';
import { CustomerSidebar } from './components/CustomerSidebar';
import { Chat } from './components/Chat';
import { ChatHistory } from './components/ChatHistory';
import { Button } from './components/ui/button';
import { MessageSquare } from 'lucide-react';

export default function App() {
  const [isChatHistoryOpen, setIsChatHistoryOpen] = useState(false);

  return (
    <UserProvider>
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
    </UserProvider>
  );
}
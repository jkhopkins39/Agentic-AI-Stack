import React, { useState } from 'react';
import { UserProvider, useUser } from './contexts/UserContext';
import { CustomerSidebar } from './components/CustomerSidebar';
import { Chat } from './components/Chat';
import { ChatHistory } from './components/ChatHistory';
import { Login } from './components/Login';
import AdminDashboard from './components/AdminDashboard';
import { Button } from './components/ui/button';
import { MessageSquare, Settings } from 'lucide-react';

function AppContent() {
  const [isChatHistoryOpen, setIsChatHistoryOpen] = useState(false);
  const [currentView, setCurrentView] = useState<'chat' | 'admin'>('chat');
  const { isAuthenticated, login, userProfile } = useUser();

  if (!isAuthenticated) {
    return <Login onLogin={login} />;
  }

  // Check if user is admin (you can customize this logic)
  const isAdmin = userProfile?.profile?.email === 'admin@example.com' || 
                  userProfile?.profile?.email?.includes('admin');

  return (
    <div className="h-screen">
      <CustomerSidebar>
        <div className="relative h-full">
          {currentView === 'chat' ? <Chat /> : <AdminDashboard />}
          
          {/* Navigation Buttons */}
          <div className="fixed top-4 right-4 z-40 flex gap-2">
            {isAdmin && (
              <Button
                onClick={() => setCurrentView(currentView === 'chat' ? 'admin' : 'chat')}
                variant={currentView === 'admin' ? 'default' : 'outline'}
                size="sm"
              >
                <Settings className="h-4 w-4 mr-2" />
                {currentView === 'chat' ? 'Admin' : 'Chat'}
              </Button>
            )}
            
            <Button
              onClick={() => setIsChatHistoryOpen(true)}
              variant="outline"
              size="sm"
            >
              <MessageSquare className="h-4 w-4 mr-2" />
              Chat History
            </Button>
          </div>

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
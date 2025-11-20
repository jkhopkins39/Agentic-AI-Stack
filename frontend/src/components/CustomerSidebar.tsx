import React, { useState } from 'react';
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
  SidebarSeparator,
  SidebarFooter,
  useSidebar,
} from './ui/sidebar';
import { Orders } from './Orders';
import { Profile } from './Profile';
import { ChatHistory } from './ChatHistory';
import { Button } from './ui/button';
import { 
  Package, 
  User, 
  MessageSquare,
  LogOut,
  PanelRightOpen
} from 'lucide-react';
import { useUser } from '../contexts/UserContext';
import { cn } from './ui/utils';

// Account Sidebar Trigger - uses controlled state
function AccountSidebarTriggerComponent({ onToggle }: { onToggle: () => void }) {
  return (
    <Button
      variant="outline"
      size="sm"
      className="shadow-md"
      onClick={(e) => {
        e.stopPropagation();
        onToggle();
      }}
    >
      Account
    </Button>
  );
}

// Chat History Sidebar Trigger - uses inner SidebarProvider context
function ChatHistorySidebarTriggerComponent() {
  const { toggleSidebar } = useSidebar();

  return (
    <Button
      variant="outline"
      size="sm"
      className="shadow-md"
      onClick={(e) => {
        e.stopPropagation();
        toggleSidebar();
      }}
    >
      Chat History
    </Button>
  );
}

interface CustomerSidebarProps {
  children: React.ReactNode;
  chatHistoryOpen?: boolean;
  onChatHistoryToggle?: (open: boolean) => void;
  selectedConversationId?: string;
  selectedSessionId?: string;
  onSelectConversation?: (conversationId: string, sessionId: string) => void;
  onNewChat?: () => void;
  chatHistoryRefresh?: number;
}

export function CustomerSidebar({ 
  children, 
  chatHistoryOpen = false,
  onChatHistoryToggle,
  selectedConversationId,
  selectedSessionId,
  onSelectConversation,
  onNewChat,
  chatHistoryRefresh
}: CustomerSidebarProps) {
  const [activeTab, setActiveTab] = useState<string | null>(null);
  const [accountSidebarOpen, setAccountSidebarOpen] = useState(true);
  const { userProfile, userOrders, logout } = useUser();

  const handleTabClick = (tab: string) => {
    setActiveTab(activeTab === tab ? null : tab);
  };

  // Get user info for display
  const userName = userProfile?.profile 
    ? `${userProfile.profile.first_name || ''} ${userProfile.profile.last_name || ''}`.trim() || userProfile.profile.email
    : 'Guest User';

  // Quick stats from real data
  const sidebarStats = {
    totalOrders: userOrders?.total_count || 0,
    pendingOrders: userOrders?.orders?.filter(order => order.status === 'pending').length || 0,
    completedOrders: userOrders?.orders?.filter(order => order.status === 'delivered').length || 0,
    totalSpent: userProfile?.total_spent || 0,
    loyaltyPoints: Math.floor((userProfile?.total_spent || 0) * 0.1) // 10% of spending as points
  };

  return (
    <SidebarProvider defaultOpen={true} open={accountSidebarOpen} onOpenChange={setAccountSidebarOpen}>
      <div className="flex h-screen w-full">
        <Sidebar className="border-r">
          <SidebarHeader className="p-4">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-6 w-6 text-primary" />
              <div>
                <span className="font-semibold">Agent Stack</span>
                <p className="text-xs text-muted-foreground">Customer Portal</p>
              </div>
            </div>
          </SidebarHeader>
          
          <SidebarContent className="overflow-y-auto">
            {/* Quick Stats */}
            <SidebarGroup>
              <SidebarGroupLabel>Quick Overview</SidebarGroupLabel>
              <SidebarGroupContent>
                <div className="grid grid-cols-2 gap-2 p-2">
                  <div className="bg-muted/50 rounded-xl p-2 text-center">
                    <div className="text-lg font-bold text-primary">{sidebarStats.totalOrders}</div>
                    <div className="text-xs text-muted-foreground">Orders</div>
                  </div>
                  <div className="bg-muted/50 rounded-xl p-2 text-center">
                    <div className="text-lg font-bold text-green-600">${sidebarStats.totalSpent.toFixed(0)}</div>
                    <div className="text-xs text-muted-foreground">Spent</div>
                  </div>
                </div>
              </SidebarGroupContent>
            </SidebarGroup>

            {/* Main Navigation */}
            <SidebarGroup>
              <SidebarGroupLabel>Account</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      onClick={() => handleTabClick('profile')}
                      isActive={activeTab === 'profile'}
                      className="w-full justify-start rounded-lg"
                    >
                      <User className="h-4 w-4" />
                      Profile
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      onClick={() => handleTabClick('orders')}
                      isActive={activeTab === 'orders'}
                      className="w-full justify-start rounded-lg"
                    >
                      <Package className="h-4 w-4" />
                      Orders
                      {sidebarStats.pendingOrders > 0 && (
                        <span className="ml-auto bg-orange-600 text-white text-xs rounded-full px-2 py-1">
                          {sidebarStats.pendingOrders}
                        </span>
                      )}
                    </SidebarMenuButton>
                  </SidebarMenuItem>

                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>


            {/* Tab Content */}
            {activeTab === 'orders' && <Orders />}
            {activeTab === 'profile' && <Profile />}
          </SidebarContent>

          <SidebarFooter className="p-4">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center">
                <User className="h-4 w-4 text-primary" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{userName}</p>
                <p className="text-xs text-muted-foreground truncate">
                  {userProfile?.profile.email || 'guest@example.com'}
                </p>
              </div>
            </div>
            <Button
              onClick={logout}
              variant="outline"
              size="sm"
              className="w-full shadow-md"
            >
              <LogOut className="h-4 w-4 mr-2" />
              Logout
            </Button>
          </SidebarFooter>
        </Sidebar>

        {/* Main Content */}
        <SidebarProvider defaultOpen={false} open={chatHistoryOpen} onOpenChange={onChatHistoryToggle}>
          <div className="flex-1 flex flex-col">
            <div className="p-4 bg-background rounded-b-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <AccountSidebarTriggerComponent onToggle={() => setAccountSidebarOpen(!accountSidebarOpen)} />
                <h1 className="text-xl font-semibold">Capgemini Agent Stack</h1>
              </div>
              <ChatHistorySidebarTriggerComponent />
            </div>
            </div>
            <div className="flex-1">
              {children}
            </div>
          </div>

          {/* Chat History Sidebar */}
          <ChatHistory 
            isOpen={chatHistoryOpen || false} 
            onClose={() => onChatHistoryToggle?.(false)}
            onSelectConversation={onSelectConversation}
            onNewChat={onNewChat}
            refreshTrigger={chatHistoryRefresh}
          />
        </SidebarProvider>
      </div>
    </SidebarProvider>
  );
}
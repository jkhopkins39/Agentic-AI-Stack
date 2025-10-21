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
} from './ui/sidebar';
import { Orders } from './Orders';
import { Profile } from './Profile';
import { ChatHistory } from './ChatHistory';
import { Button } from './ui/button';
import { 
  Package, 
  User, 
  MessageSquare, 
  History,
  LogOut
} from 'lucide-react';
import { useUser } from '../contexts/UserContext';

interface CustomerSidebarProps {
  children: React.ReactNode;
}

export function CustomerSidebar({ children }: CustomerSidebarProps) {
  const [activeTab, setActiveTab] = useState<string | null>(null);
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
    <SidebarProvider>
      <div className="flex h-screen w-full">
        <Sidebar className="border-r">
          <SidebarHeader className="border-b p-4">
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
                  <div className="bg-muted/50 rounded-lg p-2 text-center">
                    <div className="text-lg font-bold text-primary">{sidebarStats.totalOrders}</div>
                    <div className="text-xs text-muted-foreground">Orders</div>
                  </div>
                  <div className="bg-muted/50 rounded-lg p-2 text-center">
                    <div className="text-lg font-bold text-green-600">${sidebarStats.totalSpent.toFixed(0)}</div>
                    <div className="text-xs text-muted-foreground">Spent</div>
                  </div>
                </div>
              </SidebarGroupContent>
            </SidebarGroup>

            <SidebarSeparator />

            {/* Main Navigation */}
            <SidebarGroup>
              <SidebarGroupLabel>Account</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      onClick={() => handleTabClick('profile')}
                      isActive={activeTab === 'profile'}
                      className="w-full justify-start"
                    >
                      <User className="h-4 w-4" />
                      Profile
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      onClick={() => handleTabClick('orders')}
                      isActive={activeTab === 'orders'}
                      className="w-full justify-start"
                    >
                      <Package className="h-4 w-4" />
                      Orders
                      {sidebarStats.pendingOrders > 0 && (
                        <span className="ml-auto bg-orange-500 text-white text-xs rounded-full px-2 py-1">
                          {sidebarStats.pendingOrders}
                        </span>
                      )}
                    </SidebarMenuButton>
                  </SidebarMenuItem>

                  <SidebarMenuItem>
                    <SidebarMenuButton
                      onClick={() => handleTabClick('history')}
                      isActive={activeTab === 'history'}
                      className="w-full justify-start"
                    >
                      <History className="h-4 w-4" />
                      Chat History
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>


            {/* Tab Content */}
            {activeTab === 'orders' && <Orders />}
            {activeTab === 'profile' && <Profile />}
            {activeTab === 'history' && <ChatHistory isOpen={true} onClose={() => setActiveTab(null)} />}
          </SidebarContent>

          <SidebarFooter className="border-t p-4">
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
              className="w-full"
            >
              <LogOut className="h-4 w-4 mr-2" />
              Logout
            </Button>
          </SidebarFooter>
        </Sidebar>

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          <div className="border-b p-4 bg-background">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <SidebarTrigger />
                <h1 className="text-xl font-semibold">Customer Support Chat</h1>
              </div>
              <div className="flex items-center gap-2">
                <div className="text-sm text-muted-foreground">
                  Welcome back, {userName.split(' ')[0]}!
                </div>
              </div>
            </div>
          </div>
          <div className="flex-1">
            {children}
          </div>
        </div>
      </div>
    </SidebarProvider>
  );
}
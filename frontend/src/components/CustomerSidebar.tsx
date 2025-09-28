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
} from './ui/sidebar';
import { Orders } from './Orders';
import { Profile } from './Profile';
import { Package, User, MessageSquare } from 'lucide-react';

interface CustomerSidebarProps {
  children: React.ReactNode;
}

export function CustomerSidebar({ children }: CustomerSidebarProps) {
  const [activeTab, setActiveTab] = useState<string | null>(null);

  const handleTabClick = (tab: string) => {
    setActiveTab(activeTab === tab ? null : tab);
  };

  return (
    <SidebarProvider>
      <div className="flex h-screen w-full">
        <Sidebar className="border-r">
          <SidebarHeader className="border-b p-4">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-6 w-6" />
              <span>Profile Settings</span>
            </div>
          </SidebarHeader>
          
          <SidebarContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => handleTabClick('orders')}
                  isActive={activeTab === 'orders'}
                  className="w-full justify-start"
                >
                  <Package className="h-4 w-4" />
                  Orders
                </SidebarMenuButton>
              </SidebarMenuItem>
              
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
            </SidebarMenu>

            {/* Tab Content */}
            {activeTab === 'orders' && <Orders />}
            {activeTab === 'profile' && <Profile />}
          </SidebarContent>
        </Sidebar>

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          <div className="border-b p-4 bg-background">
            <div className="flex items-center gap-2">
              <SidebarTrigger />
              <h1>Customer Support Chat</h1>
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
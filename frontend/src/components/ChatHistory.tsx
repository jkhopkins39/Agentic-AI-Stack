import React, { useState } from 'react';
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from './ui/sidebar';
import { Button } from './ui/button';
import { Plus, MessageSquare } from 'lucide-react';

interface ChatSession {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
}

export function ChatHistory() {
  const [chatSessions] = useState<ChatSession[]>([]);

  const handleNewChat = () => {
    // New chat functionality would be implemented here
  };

  return (
    <Sidebar side="right" className="w-80 border-l">
      <SidebarHeader className="border-b p-4">
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            <span>Chat History</span>
          </div>
          <Button onClick={handleNewChat} className="w-full">
            <Plus className="h-4 w-4 mr-2" />
            New Chat
          </Button>
        </div>
      </SidebarHeader>
      
      <SidebarContent>
        <SidebarMenu>
          {chatSessions.length === 0 ? (
            <div className="p-4 text-center text-muted-foreground">
              <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No previous chats</p>
            </div>
          ) : (
            chatSessions.map((session) => (
              <SidebarMenuItem key={session.id}>
                <SidebarMenuButton className="flex flex-col items-start p-3 h-auto">
                  <span className="font-medium truncate w-full">{session.title}</span>
                  <span className="text-xs text-muted-foreground truncate w-full">
                    {session.lastMessage}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {session.timestamp.toLocaleDateString()}
                  </span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))
          )}
        </SidebarMenu>
      </SidebarContent>
    </Sidebar>
  );
}
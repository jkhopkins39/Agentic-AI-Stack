import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Separator } from './ui/separator';

interface Session {
  conversation_id: string;
  user_id: string;
  email: string;
  first_name: string;
  last_name: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

interface SessionDetails {
  session: {
    conversation_id: string;
    user_id: string;
    email: string;
    first_name: string;
    last_name: string;
    phone: string;
    created_at: string;
    updated_at: string;
    context: any;
  };
  messages: Array<{
    id: string;
    query_text: string;
    agent_type: string;
    agent_response: string;
    message_order: number;
    status: string;
    created_at: string;
  }>;
}

interface User {
  id: string;
  email: string;
  first_name: string;
  last_name: string;
  phone: string;
  created_at: string;
  session_count: number;
  total_messages: number;
}

interface Stats {
  stats: {
    total_users: number;
    total_conversations: number;
    total_messages: number;
    total_orders: number;
  };
  agent_distribution: Array<{
    agent_type: string;
    count: number;
  }>;
  recent_activity: {
    recent_messages: number;
  };
}

const AdminDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'sessions' | 'users'>('overview');
  const [sessions, setSessions] = useState<Session[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [selectedSession, setSelectedSession] = useState<SessionDetails | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchData();
  }, [activeTab]);

  const fetchData = async () => {
    setLoading(true);
    try {
      if (activeTab === 'overview') {
        const response = await fetch('http://localhost:8000/api/admin/stats');
        const data = await response.json();
        setStats(data);
      } else if (activeTab === 'sessions') {
        const response = await fetch('http://localhost:8000/api/admin/sessions');
        const data = await response.json();
        setSessions(data.sessions);
      } else if (activeTab === 'users') {
        const response = await fetch('http://localhost:8000/api/admin/users');
        const data = await response.json();
        setUsers(data.users);
      }
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchSessionDetails = async (conversationId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/admin/session/${conversationId}`);
      const data = await response.json();
      setSelectedSession(data);
    } catch (error) {
      console.error('Error fetching session details:', error);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getAgentBadgeColor = (agentType: string) => {
    switch (agentType?.toLowerCase()) {
      case 'order':
        return 'bg-blue-100 text-blue-800';
      case 'email':
        return 'bg-green-100 text-green-800';
      case 'policy':
        return 'bg-purple-100 text-purple-800';
      case 'message':
        return 'bg-orange-100 text-orange-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const renderOverview = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Users</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.stats.total_users || 0}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Sessions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.stats.total_conversations || 0}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Messages</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.stats.total_messages || 0}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Recent Activity (24h)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.recent_activity.recent_messages || 0}</div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Agent Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {stats?.agent_distribution.map((agent, index) => (
              <div key={index} className="flex justify-between items-center">
                <Badge className={getAgentBadgeColor(agent.agent_type)}>
                  {agent.agent_type}
                </Badge>
                <span className="font-medium">{agent.count} messages</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const renderSessions = () => (
    <div className="space-y-4">
      {sessions.map((session) => (
        <Card key={session.conversation_id} className="cursor-pointer hover:shadow-md transition-shadow"
              onClick={() => fetchSessionDetails(session.conversation_id)}>
          <CardHeader>
            <div className="flex justify-between items-start">
              <div>
                <CardTitle className="text-lg">
                  {session.first_name} {session.last_name}
                </CardTitle>
                <p className="text-sm text-gray-600">{session.email}</p>
              </div>
              <Badge variant="outline">{session.message_count} messages</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium">Session ID:</span>
                <p className="text-gray-600 font-mono text-xs">{session.conversation_id}</p>
              </div>
              <div>
                <span className="font-medium">User ID:</span>
                <p className="text-gray-600 font-mono text-xs">{session.user_id}</p>
              </div>
              <div>
                <span className="font-medium">Created:</span>
                <p className="text-gray-600">{formatDate(session.created_at)}</p>
              </div>
              <div>
                <span className="font-medium">Last Updated:</span>
                <p className="text-gray-600">{formatDate(session.updated_at)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );

  const renderUsers = () => (
    <div className="space-y-4">
      {users.map((user) => (
        <Card key={user.id}>
          <CardHeader>
            <CardTitle className="text-lg">
              {user.first_name} {user.last_name}
            </CardTitle>
            <p className="text-sm text-gray-600">{user.email}</p>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium">Phone:</span>
                <p className="text-gray-600">{user.phone || 'N/A'}</p>
              </div>
              <div>
                <span className="font-medium">Sessions:</span>
                <p className="text-gray-600">{user.session_count}</p>
              </div>
              <div>
                <span className="font-medium">Total Messages:</span>
                <p className="text-gray-600">{user.total_messages}</p>
              </div>
              <div>
                <span className="font-medium">Joined:</span>
                <p className="text-gray-600">{formatDate(user.created_at)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );

  const renderSessionDetails = () => {
    if (!selectedSession) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
        <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-hidden">
          <div className="p-6 border-b">
            <div className="flex justify-between items-start">
              <div>
                <h2 className="text-2xl font-bold">
                  {selectedSession.session.first_name} {selectedSession.session.last_name}
                </h2>
                <p className="text-gray-600">{selectedSession.session.email}</p>
              </div>
              <Button onClick={() => setSelectedSession(null)} variant="outline">
                Close
              </Button>
            </div>
          </div>
          
          <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-2">Session Information</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium">Session ID:</span>
                  <p className="text-gray-600 font-mono text-xs">{selectedSession.session.conversation_id}</p>
                </div>
                <div>
                  <span className="font-medium">User ID:</span>
                  <p className="text-gray-600 font-mono text-xs">{selectedSession.session.user_id}</p>
                </div>
                <div>
                  <span className="font-medium">Phone:</span>
                  <p className="text-gray-600">{selectedSession.session.phone || 'N/A'}</p>
                </div>
                <div>
                  <span className="font-medium">Created:</span>
                  <p className="text-gray-600">{formatDate(selectedSession.session.created_at)}</p>
                </div>
              </div>
            </div>

            <Separator className="my-6" />

            <div>
              <h3 className="text-lg font-semibold mb-4">Conversation History</h3>
              <div className="space-y-4">
                {selectedSession.messages.map((message, index) => (
                  <Card key={message.id}>
                    <CardHeader className="pb-2">
                      <div className="flex justify-between items-center">
                        <Badge className={getAgentBadgeColor(message.agent_type)}>
                          {message.agent_type || 'Unknown'}
                        </Badge>
                        <span className="text-sm text-gray-500">
                          Message #{message.message_order}
                        </span>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div>
                        <span className="font-medium text-sm">User Query:</span>
                        <p className="text-gray-700 mt-1">{message.query_text}</p>
                      </div>
                      <div>
                        <span className="font-medium text-sm">Agent Response:</span>
                        <p className="text-gray-700 mt-1">{message.agent_response}</p>
                      </div>
                      <div className="text-xs text-gray-500">
                        {formatDate(message.created_at)}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Admin Dashboard</h1>
          <p className="text-gray-600 mt-2">Monitor user sessions, messages, and system activity</p>
        </div>

        <div className="mb-6">
          <div className="flex space-x-1 bg-gray-200 p-1 rounded-lg w-fit">
            <Button
              variant={activeTab === 'overview' ? 'default' : 'ghost'}
              onClick={() => setActiveTab('overview')}
              className="px-4"
            >
              Overview
            </Button>
            <Button
              variant={activeTab === 'sessions' ? 'default' : 'ghost'}
              onClick={() => setActiveTab('sessions')}
              className="px-4"
            >
              Sessions
            </Button>
            <Button
              variant={activeTab === 'users' ? 'default' : 'ghost'}
              onClick={() => setActiveTab('users')}
              className="px-4"
            >
              Users
            </Button>
          </div>
        </div>

        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="text-lg">Loading...</div>
          </div>
        ) : (
          <>
            {activeTab === 'overview' && renderOverview()}
            {activeTab === 'sessions' && renderSessions()}
            {activeTab === 'users' && renderUsers()}
          </>
        )}

        {renderSessionDetails()}
      </div>
    </div>
  );
};

export default AdminDashboard;

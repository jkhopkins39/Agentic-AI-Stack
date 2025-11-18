import React, { useState, useRef, useEffect } from 'react';
import { useKafkaPublisher, useAgentResponsesConsumer, Message } from './hooks/useKafka';

export const ChatInterface: React.FC = () => {
  const [sessionId] = useState(() => `session-${Date.now()}`);
  const [userId] = useState<string | null>(null);
  const [inputMessage, setInputMessage] = useState('');
  const [userMessages, setUserMessages] = useState<Message[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { publishToIngress, isPublishing, publishError } = useKafkaPublisher(sessionId, userId);
  const { messages: agentMessages, isConnected, error: wsError } = useAgentResponsesConsumer(sessionId);

  const allMessages = [...userMessages, ...agentMessages].sort(
    (a, b) => a.timestamp.getTime() - b.timestamp.getTime()
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [allMessages]);

  const sendMessage = async () => {
    if (!inputMessage.trim() || isPublishing || !isConnected) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content: inputMessage,
      sender: 'user',
      timestamp: new Date(),
      status: 'pending'
    };

    setUserMessages(prev => [...prev, userMessage]);
    const messageToSend = inputMessage;
    setInputMessage('');

    try {
      await publishToIngress(messageToSend);
      setUserMessages(prev =>
        prev.map(msg => msg.id === userMessage.id ? { ...msg, status: 'completed' as const } : msg)
      );
    } catch (error) {
      setUserMessages(prev =>
        prev.map(msg => msg.id === userMessage.id ? { ...msg, status: 'error' as const } : msg)
      );
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b p-4">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-bold">🤖 Multi-Agent AI</h1>
          <div className={`px-3 py-1 rounded-full text-sm ${
            isConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
          }`}>
            {isConnected ? '● Connected' : '○ Disconnected'}
          </div>
        </div>
        {(publishError || wsError) && (
          <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
            {publishError || wsError}
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {allMessages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-xl px-4 py-3 rounded-2xl ${
              msg.sender === 'user' 
                ? 'bg-blue-600 text-white' 
                : 'bg-white text-gray-800 border'
            }`}>
              {msg.agentUsed && (
                <div className="text-xs font-semibold mb-1 px-2 py-0.5 bg-blue-100 text-blue-700 rounded inline-block">
                  {msg.agentUsed} Agent
                </div>
              )}
              <div className="text-sm">{msg.content}</div>
              <div className="text-xs opacity-70 mt-1">
                {msg.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="bg-white border-t p-4">
        <div className="flex space-x-3">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            disabled={isPublishing || !isConnected}
            className="flex-1 px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
          />
          <button
            onClick={sendMessage}
            disabled={isPublishing || !isConnected || !inputMessage.trim()}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
          >
            {isPublishing ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
};
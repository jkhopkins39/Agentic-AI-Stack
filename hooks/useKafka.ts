import { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';
const WS_BASE_URL = 'ws://localhost:8000';

export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  agentUsed?: string;
  timestamp: Date;
  priority?: number;
  status?: 'pending' | 'processing' | 'completed' | 'error';
}

// ============================================================================
// PUBLISHER HOOK
// ============================================================================
export const useKafkaPublisher = (sessionId: string, userId: string | null = null) => {
  const [isPublishing, setIsPublishing] = useState(false);
  const [publishError, setPublishError] = useState<string | null>(null);

  const publishToIngress = useCallback(async (message: string): Promise<void> => {
    setIsPublishing(true);
    setPublishError(null);

    try {
      await axios.post(`${API_BASE_URL}/publish/ingress`, {
        session_id: sessionId,
        user_id: userId,
        query_text: message,
        correlation_id: `corr-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        event_timestamp: Date.now()
      });
      
      console.log('✅ Published to system.ingress');
    } catch (error) {
      const errorMsg = axios.isAxiosError(error) 
        ? error.response?.data?.detail || error.message
        : 'Unknown error';
      setPublishError(errorMsg);
      throw new Error(errorMsg);
    } finally {
      setIsPublishing(false);
    }
  }, [sessionId, userId]);

  return { publishToIngress, isPublishing, publishError };
};

// ============================================================================
// CONSUMER HOOK
// ============================================================================
export const useAgentResponsesConsumer = (sessionId: string) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const wsUrl = `${WS_BASE_URL}/ws/agent-responses/${sessionId}`;
    console.log('Connecting to WebSocket:', wsUrl);
    
    const connectWebSocket = () => {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        console.log('📨 Received message:', event.data);
        try {
          const response = JSON.parse(event.data);
          console.log('Parsed response:', response);
          
          if (response.session_id === sessionId) {
            const newMessage: Message = {
              id: `${response.timestamp}-${response.agent_type}`,
              content: response.message,
              sender: 'assistant',
              agentUsed: response.agent_type,
              timestamp: new Date(response.timestamp),
              priority: response.priority
            };

            console.log('Adding new message:', newMessage);
            setMessages(prev => [...prev, newMessage]);
          } else {
            console.log('Session ID mismatch:', response.session_id, 'vs', sessionId);
          }
        } catch (err) {
          console.error('Error parsing message:', err);
          console.error('Raw message data:', event.data);
        }
      };

      ws.onerror = (event) => {
        console.error('❌ WebSocket error:', event);
        setError('WebSocket error - check console for details');
      };
      
      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        setIsConnected(false);
        
        // Only reconnect if it wasn't a normal closure
        if (event.code !== 1000) {
          console.log('Reconnecting in 3 seconds...');
          setTimeout(() => {
            if (wsRef.current?.readyState === WebSocket.CLOSED) {
              connectWebSocket();
            }
          }, 3000);
        }
      };
    };

    connectWebSocket();

    return () => {
      console.log('Cleaning up WebSocket');
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounting');
      }
    };
  }, [sessionId]);

  return { messages, isConnected, error };
};
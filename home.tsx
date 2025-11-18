import React, { useState } from 'react';
import './Home.css';

const Home: React.FC = () => {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Array<{id: number, text: string, sender: 'user' | 'assistant'}>>([]);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async () => {
    if (!message.trim()) return;

    const userMessage = { id: Date.now(), text: message, sender: 'user' as const };
    setMessages(prev => [...prev, userMessage]);
    setMessage('');
    setIsLoading(true);

    try {
      // Replace with your actual API endpoint
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }),
      });

      const data = await response.json();
      const assistantMessage = { 
        id: Date.now() + 1, 
        text: data.response || 'Hello! I\'m your AI assistant. How can I help you today?', 
        sender: 'assistant' as const 
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = { 
        id: Date.now() + 1, 
        text: 'Sorry, I encountered an error. Please try again.', 
        sender: 'assistant' as const 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="home">
      <header className="header">
        <h1>🤖 Multi-Agent AI Assistant</h1>
        <p>Powered by LangGraph & Claude</p>
      </header>

      <div className="chat-container">
        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome-message">
              <h2>Welcome to your AI Assistant!</h2>
              <p>Ask me anything about:</p>
              <ul>
                <li>📦 Order status and tracking</li>
                <li>📧 Email assistance</li>
                <li>📋 Company policies</li>
                <li>💬 General questions</li>
              </ul>
            </div>
          )}
          
          {messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.sender}`}>
              <div className="message-content">
                <strong>{msg.sender === 'user' ? 'You' : 'Assistant'}:</strong>
                <p>{msg.text}</p>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="message assistant">
              <div className="message-content">
                <strong>Assistant:</strong>
                <p className="typing">Thinking...</p>
              </div>
            </div>
          )}
        </div>

        <div className="input-section">
          <div className="input-container">
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message here... (Press Enter to send)"
              rows={3}
              disabled={isLoading}
            />
            <button 
              onClick={sendMessage} 
              disabled={isLoading || !message.trim()}
              className="send-button"
            >
              {isLoading ? '⏳' : '📤'} Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
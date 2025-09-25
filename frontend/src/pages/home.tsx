import { CustomerSidebar } from '../components/CustomerSidebar';
import { Chat } from '../components/Chat';
import { ChatHistory } from '../components/ChatHistory';

export default function HomePage() {
  return (
    <div className="h-screen">
      <CustomerSidebar>
        <div className="flex h-full">
          <div className="flex-1">
            <Chat />
          </div>
          <ChatHistory />
        </div>
      </CustomerSidebar>
    </div>
  );
}
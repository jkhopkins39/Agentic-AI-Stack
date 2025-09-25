import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/home';
import { ChatInterface } from './components/ChatInterface';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/chat" element={<ChatInterface />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
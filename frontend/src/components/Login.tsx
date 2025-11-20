import React, { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Label } from './ui/label';
import { Alert, AlertDescription } from './ui/alert';
import { Loader2, Mail, Lock, MessageSquare, User, Copy, Check } from 'lucide-react';

interface LoginProps {
  onLogin: (email: string) => void;
}

export function Login({ onLogin }: LoginProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (data.success) {
        onLogin(email);
      } else {
        setError(data.message || 'Login failed');
      }
    } catch (err) {
      setError('Failed to connect to server. Please check if the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleGuestLogin = () => {
    // Guest login - no authentication required
    onLogin('guest@example.com');
  };

  const handleAdminQuickLogin = () => {
    setEmail('admin@example.com');
    setPassword('\\\/LewdBPj4J/8KzKz2K');
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 pt-16">
      {/* Login Card - 1/2 screen width, 1/3 screen height, centered */}
      <div className="w-1/2 h-1/3 min-w-[400px] min-h-[300px] max-w-[800px] max-h-[600px]">
        <div className="bg-white rounded-3xl shadow-2xl border border-gray-200 h-full flex flex-col">
          {/* Header Section */}
          <div className="text-center py-6 px-8 flex-shrink-0">
            <div className="flex justify-center mb-4">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                <MessageSquare className="h-6 w-6 text-blue-600" />
              </div>
            </div>
            <h1 className="text-2xl font-bold text-gray-900 mb-1">Welcome to Agent Stack</h1>
            <p className="text-base text-gray-600">Sign in to access your customer portal</p>
          </div>

          {/* Form Section - Takes remaining space */}
          <div className="flex-1 px-8 pb-6 flex flex-col justify-center">
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Email Field */}
              <div className="space-y-2">
                <Label htmlFor="email" className="text-base font-medium text-gray-700">Email</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
                  <Input
                    id="email"
                    type="email"
                    placeholder="Enter your email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="pl-10 h-10 text-base border border-gray-300 rounded-lg focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    required
                  />
                </div>
              </div>
              
              {/* Password Field */}
              <div className="space-y-2">
                <Label htmlFor="password" className="text-base font-medium text-gray-700">Password</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
                  <Input
                    id="password"
                    type="password"
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="pl-10 h-10 text-base border border-gray-300 rounded-lg focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    required
                  />
                </div>
              </div>

              {/* Error Message */}
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              )}

              {/* Submit Button */}
              <Button 
                type="submit" 
                className="w-full h-10 text-base font-medium bg-blue-600 text-white hover:bg-blue-700 rounded-lg shadow-md" 
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Signing in...
                  </>
                ) : (
                  'Sign In'
                )}
              </Button>
            </form>

            {/* Guest Login Button */}
            <div className="mt-4">
              <Button
                type="button"
                variant="outline"
                onClick={handleGuestLogin}
                className="w-full h-10 text-base border-gray-300 shadow-md"
              >
                <User className="mr-2 h-4 w-4" />
                Continue as Guest
              </Button>
            </div>

            {/* Admin Credentials */}
            <div className="mt-4 pt-4 border-t border-gray-200">
              <p className="text-xs text-gray-500 mb-2 text-center">Admin Credentials</p>
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-gray-500 mb-1">Email:</p>
                    <div className="flex items-center gap-2">
                      <code className="text-xs font-mono text-gray-800 break-all">admin@example.com</code>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0"
                        onClick={() => copyToClipboard('admin@example.com')}
                        title="Copy email"
                      >
                        {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                      </Button>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleAdminQuickLogin}
                    className="ml-2 text-xs"
                  >
                    Fill
                  </Button>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-gray-500 mb-1">Password:</p>
                    <div className="flex items-center gap-2">
                      <code className="text-xs font-mono text-gray-800 break-all">\\\/LewdBPj4J/8KzKz2K</code>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0"
                        onClick={() => copyToClipboard('\\\/LewdBPj4J/8KzKz2K')}
                        title="Copy password"
                      >
                        {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                      </Button>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleAdminQuickLogin}
                    className="ml-2 text-xs"
                  >
                    Fill
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * User Context for managing current user session and data
 */

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { apiService, UserProfileResponse, OrdersResponse } from '../services/api';

interface UserContextType {
  // Current user data
  userProfile: UserProfileResponse | null;
  userOrders: OrdersResponse | null;
  
  // Loading states
  isLoadingProfile: boolean;
  isLoadingOrders: boolean;
  
  // Error states
  profileError: string | null;
  ordersError: string | null;
  
  // Authentication state
  isAuthenticated: boolean;
  
  // Actions
  setCurrentUser: (email: string) => void;
  refreshProfile: () => Promise<void>;
  refreshOrders: () => Promise<void>;
  clearUser: () => void;
  login: (email: string) => void;
  logout: () => void;
  
  // Current user email
  currentUserEmail: string | null;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

interface UserProviderProps {
  children: ReactNode;
}

export function UserProvider({ children }: UserProviderProps) {
  const [userProfile, setUserProfile] = useState<UserProfileResponse | null>(null);
  const [userOrders, setUserOrders] = useState<OrdersResponse | null>(null);
  const [isLoadingProfile, setIsLoadingProfile] = useState(false);
  const [isLoadingOrders, setIsLoadingOrders] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [ordersError, setOrdersError] = useState<string | null>(null);
  const [currentUserEmail, setCurrentUserEmail] = useState<string | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Load user profile
  const loadUserProfile = async (email: string) => {
    setIsLoadingProfile(true);
    setProfileError(null);
    
    try {
      const profile = await apiService.getUserProfile(email);
      setUserProfile(profile);
    } catch (error) {
      console.error('Error loading user profile:', error);
      setProfileError(error instanceof Error ? error.message : 'Failed to load profile');
      // For guest or invalid users, set a minimal profile
      if (email === 'guest@example.com' || error instanceof Error && error.message.includes('404')) {
        const now = new Date().toISOString();
        setUserProfile({
          profile: {
            id: 'guest',
            email: email,
            first_name: 'Guest',
            last_name: 'User',
            phone: null,
            is_admin: false,
            created_at: now,
            updated_at: now
          },
          addresses: [],
          total_orders: 0,
          total_spent: 0
        });
      }
    } finally {
      setIsLoadingProfile(false);
    }
  };

  // Load user orders
  const loadUserOrders = async (email: string) => {
    setIsLoadingOrders(true);
    setOrdersError(null);
    
    try {
      const orders = await apiService.getUserOrders(email);
      setUserOrders(orders);
    } catch (error) {
      console.error('Error loading user orders:', error);
      setOrdersError(error instanceof Error ? error.message : 'Failed to load orders');
      // For guest users, set empty orders
      if (email === 'guest@example.com') {
        setUserOrders({
          orders: [],
          total_count: 0,
          page: 1,
          limit: 10
        });
      }
    } finally {
      setIsLoadingOrders(false);
    }
  };

  // Set current user and load their data
  const setCurrentUser = (email: string) => {
    setCurrentUserEmail(email);
    loadUserProfile(email);
    loadUserOrders(email);
  };

  // Login user
  const login = (email: string) => {
    // Clear previous user data first
    clearUser();
    setIsAuthenticated(true);
    localStorage.setItem('userEmail', email);
    setCurrentUser(email);
  };

  // Logout user
  const logout = () => {
    setIsAuthenticated(false);
    localStorage.removeItem('userEmail');
    clearUser();
  };

  // Refresh profile data
  const refreshProfile = async () => {
    if (currentUserEmail) {
      await loadUserProfile(currentUserEmail);
    }
  };

  // Refresh orders data
  const refreshOrders = async () => {
    if (currentUserEmail) {
      await loadUserOrders(currentUserEmail);
    }
  };

  // Clear user data
  const clearUser = () => {
    setUserProfile(null);
    setUserOrders(null);
    setCurrentUserEmail(null);
    setProfileError(null);
    setOrdersError(null);
  };

  // Check for existing authentication on mount
  useEffect(() => {
    const savedEmail = localStorage.getItem('userEmail');
    if (savedEmail) {
      login(savedEmail);
    }
  }, []);

  const value: UserContextType = {
    userProfile,
    userOrders,
    isLoadingProfile,
    isLoadingOrders,
    profileError,
    ordersError,
    isAuthenticated,
    setCurrentUser,
    refreshProfile,
    refreshOrders,
    clearUser,
    login,
    logout,
    currentUserEmail,
  };

  return (
    <UserContext.Provider value={value}>
      {children}
    </UserContext.Provider>
  );
}

export function useUser() {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
}

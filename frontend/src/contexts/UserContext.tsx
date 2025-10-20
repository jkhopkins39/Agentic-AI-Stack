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
  
  // Actions
  setCurrentUser: (email: string) => void;
  refreshProfile: () => Promise<void>;
  refreshOrders: () => Promise<void>;
  clearUser: () => void;
  
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

  // Auto-load demo user on mount (for testing)
  useEffect(() => {
    // Set a default demo user for testing
    setCurrentUser('john.doe@example.com');
  }, []);

  const value: UserContextType = {
    userProfile,
    userOrders,
    isLoadingProfile,
    isLoadingOrders,
    profileError,
    ordersError,
    setCurrentUser,
    refreshProfile,
    refreshOrders,
    clearUser,
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

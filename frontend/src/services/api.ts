/**
 * API Service for communicating with the backend
 */

const API_BASE_URL = 'http://localhost:8000/api';

export interface UserProfile {
  id: string;
  email: string;
  first_name: string | null;
  last_name: string | null;
  phone: string | null;
  created_at: string;
  updated_at: string;
}

export interface UserAddress {
  id: string;
  address: string;
  city: string;
  state: string | null;
  postal_code: string;
  country: string;
}

export interface OrderItem {
  product_name: string;
  quantity: number;
  unit_price: number;
  total_price: number;
}

export interface Order {
  id: string;
  order_number: string;
  status: string;
  total_amount: number;
  currency: string;
  created_at: string;
  updated_at: string;
  shipped_at: string | null;
  delivered_at: string | null;
  items: OrderItem[];
}

export interface UserProfileResponse {
  profile: UserProfile;
  addresses: UserAddress[];
  total_orders: number;
  total_spent: number;
}

export interface OrdersResponse {
  orders: Order[];
  total_count: number;
  page: number;
  limit: number;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  database: string;
  kafka: string;
}

class ApiService {
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
      },
    };

    const response = await fetch(url, { ...defaultOptions, ...options });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error ${response.status}: ${errorText}`);
    }

    return response.json();
  }

  /**
   * Check system health
   */
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  /**
   * Get user profile with addresses and order statistics
   */
  async getUserProfile(userEmail: string): Promise<UserProfileResponse> {
    return this.request<UserProfileResponse>(`/user/profile?user_email=${encodeURIComponent(userEmail)}`);
  }

  /**
   * Get user's orders with pagination
   */
  async getUserOrders(userEmail: string, page: number = 1, limit: number = 10): Promise<OrdersResponse> {
    const params = new URLSearchParams({
      user_email: userEmail,
      page: page.toString(),
      limit: limit.toString(),
    });
    
    return this.request<OrdersResponse>(`/user/orders?${params}`);
  }

  /**
   * Get detailed order information by order number
   */
  async getOrderDetails(orderNumber: string): Promise<Order> {
    return this.request<Order>(`/orders/${encodeURIComponent(orderNumber)}`);
  }
}

// Export a singleton instance
export const apiService = new ApiService();

import React, { useState } from 'react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Plus, Package, Truck, CheckCircle, RotateCcw, ArrowLeft } from 'lucide-react';

type OrderStatus = 'order_placed' | 'in_transit' | 'delivered' | 'return_processing' | 'returned';

interface Order {
  id: string;
  number: string;
  status: OrderStatus;
  date: Date;
  items: string[];
  total: number;
}

const mockOrders: Order[] = [];

export function Orders() {
  const [orders] = useState<Order[]>(mockOrders);

  const getStatusIcon = (status: OrderStatus) => {
    switch (status) {
      case 'order_placed':
        return <Package className="h-3 w-3" />;
      case 'in_transit':
        return <Truck className="h-3 w-3" />;
      case 'delivered':
        return <CheckCircle className="h-3 w-3" />;
      case 'return_processing':
        return <RotateCcw className="h-3 w-3" />;
      case 'returned':
        return <ArrowLeft className="h-3 w-3" />;
    }
  };

  const getStatusColor = (status: OrderStatus) => {
    switch (status) {
      case 'order_placed':
        return 'bg-blue-100 text-blue-800';
      case 'in_transit':
        return 'bg-yellow-100 text-yellow-800';
      case 'delivered':
        return 'bg-green-100 text-green-800';
      case 'return_processing':
        return 'bg-orange-100 text-orange-800';
      case 'returned':
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusLabel = (status: OrderStatus) => {
    switch (status) {
      case 'order_placed':
        return 'Order Placed';
      case 'in_transit':
        return 'In Transit';
      case 'delivered':
        return 'Delivered';
      case 'return_processing':
        return 'Return Processing';
      case 'returned':
        return 'Returned';
    }
  };

  const handleNewOrder = () => {
    // New order functionality would be implemented here
  };

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2>Your Orders</h2>
        <Button onClick={handleNewOrder} size="sm">
          <Plus className="h-4 w-4 mr-2" />
          New Order
        </Button>
      </div>

      <div className="space-y-3">
        {orders.map((order) => (
          <Card key={order.id} className="p-3">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">{order.number}</p>
                  <p className="text-sm text-muted-foreground">
                    {order.date.toLocaleDateString()}
                  </p>
                </div>
                <Badge className={getStatusColor(order.status)}>
                  {getStatusIcon(order.status)}
                  <span className="ml-1">{getStatusLabel(order.status)}</span>
                </Badge>
              </div>

              <div>
                <p className="text-sm text-muted-foreground mb-1">Items:</p>
                <div className="space-y-1">
                  {order.items.map((item, index) => (
                    <p key={index} className="text-sm">• {item}</p>
                  ))}
                </div>
              </div>

              <div className="flex justify-between items-center pt-2 border-t">
                <span className="text-sm text-muted-foreground">Total:</span>
                <span className="font-medium">${order.total.toFixed(2)}</span>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {orders.length === 0 && (
        <div className="text-center text-muted-foreground py-8">
          <Package className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No orders found</p>
          <Button onClick={handleNewOrder} className="mt-4">
            <Plus className="h-4 w-4 mr-2" />
            Place Your First Order
          </Button>
        </div>
      )}
    </div>
  );
}
import React from 'react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Plus, Package, Truck, CheckCircle, RotateCcw, ArrowLeft, RefreshCw, AlertCircle } from 'lucide-react';
import { useUser } from '../contexts/UserContext';

type OrderStatus = 'pending' | 'in_transit' | 'delivered' | 'return_processing' | 'returned' | 'cancelled';

export function Orders() {
  const { userOrders, isLoadingOrders, ordersError, refreshOrders } = useUser();

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'pending':
        return <Package className="h-3 w-3" />;
      case 'in_transit':
        return <Truck className="h-3 w-3" />;
      case 'delivered':
        return <CheckCircle className="h-3 w-3" />;
      case 'return_processing':
        return <RotateCcw className="h-3 w-3" />;
      case 'returned':
        return <ArrowLeft className="h-3 w-3" />;
      case 'cancelled':
        return <ArrowLeft className="h-3 w-3" />;
      default:
        return <Package className="h-3 w-3" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'pending':
        return 'bg-blue-100 text-blue-800';
      case 'in_transit':
        return 'bg-yellow-100 text-yellow-800';
      case 'delivered':
        return 'bg-green-100 text-green-800';
      case 'return_processing':
        return 'bg-orange-100 text-orange-800';
      case 'returned':
        return 'bg-gray-100 text-gray-800';
      case 'cancelled':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status.toLowerCase()) {
      case 'pending':
        return 'Pending';
      case 'in_transit':
        return 'In Transit';
      case 'delivered':
        return 'Delivered';
      case 'return_processing':
        return 'Return Processing';
      case 'returned':
        return 'Returned';
      case 'cancelled':
        return 'Cancelled';
      default:
        return status.charAt(0).toUpperCase() + status.slice(1);
    }
  };

  const handleNewOrder = () => {
    // New order functionality would be implemented here
  };

  // Loading state
  if (isLoadingOrders) {
    return (
      <div className="p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h2>Your Orders</h2>
          <RefreshCw className="h-4 w-4 animate-spin" />
        </div>
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-32 bg-muted animate-pulse rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  // Error state
  if (ordersError) {
    return (
      <div className="p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h2>Your Orders</h2>
          <Button onClick={refreshOrders} size="sm" variant="outline" className="shadow-md">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
        <Card>
          <CardContent className="flex items-center gap-2 p-6 text-destructive">
            <AlertCircle className="h-5 w-5" />
            <span>Failed to load orders: {ordersError}</span>
          </CardContent>
        </Card>
      </div>
    );
  }

  const orders = userOrders?.orders || [];

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2>Your Orders</h2>
        <div className="flex gap-2">
          <Button onClick={refreshOrders} size="sm" variant="outline" disabled={isLoadingOrders} className="shadow-md">
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoadingOrders ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button onClick={handleNewOrder} size="sm" className="shadow-md">
            <Plus className="h-4 w-4 mr-2" />
            New Order
          </Button>
        </div>
      </div>

      {userOrders && (
        <div className="text-sm text-muted-foreground">
          Showing {orders.length} of {userOrders.total_count} orders
        </div>
      )}

<<<<<<< Updated upstream
=======
      {orders.length === 0 && !showOrderForm && (
        <div className="text-center text-muted-foreground py-8">
          <Package className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No orders found</p>
          <Button onClick={handleNewOrder} className="mt-4 shadow-md">
            <Plus className="h-4 w-4 mr-2" />
            Place Your First Order
          </Button>
        </div>
      )}

>>>>>>> Stashed changes
      <div className="space-y-3">
        {orders.map((order) => (
          <Card key={order.id} className="p-3">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">{order.order_number}</p>
                  <p className="text-sm text-muted-foreground">
                    {new Date(order.created_at).toLocaleDateString()}
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
                    <p key={index} className="text-sm">
                      • {item.product_name} (Qty: {item.quantity}) - ${item.total_price.toFixed(2)}
                    </p>
                  ))}
                </div>
              </div>

              <div className="flex justify-between items-center pt-2 border-t">
                <span className="text-sm text-muted-foreground">Total:</span>
                <span className="font-medium">${order.total_amount.toFixed(2)} {order.currency}</span>
              </div>

              {/* Order dates */}
              <div className="text-xs text-muted-foreground space-y-1">
                {order.shipped_at && (
                  <div>Shipped: {new Date(order.shipped_at).toLocaleDateString()}</div>
                )}
                {order.delivered_at && (
                  <div>Delivered: {new Date(order.delivered_at).toLocaleDateString()}</div>
                )}
              </div>
            </div>
          </Card>
        ))}
      </div>

<<<<<<< Updated upstream
      {orders.length === 0 && (
        <div className="text-center text-muted-foreground py-8">
          <Package className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No orders found</p>
          <Button onClick={handleNewOrder} className="mt-4">
            <Plus className="h-4 w-4 mr-2" />
            Place Your First Order
          </Button>
        </div>
=======
      {/* Inline Order Form */}
      {showOrderForm && (
        <Card className="p-4 mt-4">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Create New Order</h3>
              <Button
                onClick={() => {
                  setShowOrderForm(false);
                  setOrderItems([]);
                  setError(null);
                }}
                variant="ghost"
                size="sm"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
                {error}
              </div>
            )}

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Order Items</Label>
                <Button onClick={addOrderItem} size="sm" variant="outline" className="shadow-md">
                  <Plus className="h-4 w-4 mr-1" />
                  Add Item
                </Button>
              </div>

              {orderItems.length === 0 ? (
                <div className="text-center py-4 text-muted-foreground text-sm">
                  No items added. Click "Add Item" to get started.
                </div>
              ) : (
                <div className="space-y-3">
                  {orderItems.map((item, index) => (
                    <Card key={index} className="p-3">
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Label className="text-sm font-medium">Item {index + 1}</Label>
                          {orderItems.length > 1 && (
                            <Button
                              onClick={() => removeOrderItem(index)}
                              size="sm"
                              variant="ghost"
                              className="h-6 w-6 p-0"
                            >
                              <X className="h-4 w-4" />
                            </Button>
                          )}
                        </div>

                        <div className="space-y-2">
                          <div>
                            <Label htmlFor={`product-${index}`} className="text-xs">Product</Label>
                            <select
                              id={`product-${index}`}
                              value={item.product_id}
                              onChange={(e) => updateOrderItem(index, 'product_id', e.target.value)}
                              className="w-full mt-1 px-3 py-2 border rounded-md text-sm"
                            >
                              {products.map((product) => (
                                <option key={product.id} value={product.id}>
                                  {product.name} - ${product.price.toFixed(2)}
                                </option>
                              ))}
                            </select>
                          </div>

                          <div className="grid grid-cols-2 gap-2">
                            <div>
                              <Label htmlFor={`quantity-${index}`} className="text-xs">Quantity</Label>
                              <Input
                                id={`quantity-${index}`}
                                type="number"
                                min="1"
                                value={item.quantity}
                                onChange={(e) => updateOrderItem(index, 'quantity', parseInt(e.target.value) || 1)}
                                className="mt-1"
                              />
                            </div>
                            <div>
                              <Label htmlFor={`price-${index}`} className="text-xs">Unit Price</Label>
                              <Input
                                id={`price-${index}`}
                                type="number"
                                step="0.01"
                                value={item.unit_price.toFixed(2)}
                                onChange={(e) => updateOrderItem(index, 'unit_price', parseFloat(e.target.value) || 0)}
                                className="mt-1"
                              />
                            </div>
                          </div>

                          <div className="text-sm text-muted-foreground">
                            Subtotal: ${(item.unit_price * item.quantity).toFixed(2)}
                          </div>
                        </div>
                      </div>
                    </Card>
                  ))}
                </div>
              )}
            </div>

            {orderItems.length > 0 && (
              <div className="pt-4 border-t">
                <div className="flex justify-between items-center text-lg font-semibold">
                  <span>Total:</span>
                  <span>${calculateTotal().toFixed(2)}</span>
                </div>
              </div>
            )}

            <div className="flex gap-2 pt-4">
              <Button
                onClick={() => {
                  setShowOrderForm(false);
                  setOrderItems([]);
                  setError(null);
                }}
                variant="outline"
                disabled={isSubmitting}
                className="flex-1 shadow-md"
              >
                Cancel
              </Button>
              <Button
                onClick={handleSubmitOrder}
                disabled={isSubmitting || orderItems.length === 0}
                className="flex-1 shadow-md"
              >
                {isSubmitting ? 'Creating...' : 'Save Order'}
              </Button>
            </div>
          </div>
        </Card>
>>>>>>> Stashed changes
      )}
    </div>
  );
}
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Avatar, AvatarFallback } from './ui/avatar';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { User, Mail, Phone, MapPin, Calendar, RefreshCw, AlertCircle } from 'lucide-react';
import { useUser } from '../contexts/UserContext';

type UserStatus = 'active' | 'premium' | 'inactive';

export function Profile() {
  const { userProfile, isLoadingProfile, profileError, refreshProfile } = useUser();

  const getStatusColor = (status: UserStatus) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'premium':
        return 'bg-purple-100 text-purple-800';
      case 'inactive':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // Loading state
  if (isLoadingProfile) {
    return (
      <div className="p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h2>Profile</h2>
          <RefreshCw className="h-4 w-4 animate-spin" />
        </div>
        <div className="space-y-4">
          <div className="h-32 bg-muted animate-pulse rounded-lg" />
          <div className="h-24 bg-muted animate-pulse rounded-lg" />
          <div className="h-20 bg-muted animate-pulse rounded-lg" />
        </div>
      </div>
    );
  }

  // Error state
  if (profileError) {
    return (
      <div className="p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h2>Profile</h2>
          <Button onClick={refreshProfile} size="sm" variant="outline" className="shadow-md">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
        <Card>
          <CardContent className="flex items-center gap-2 p-6 text-destructive">
            <AlertCircle className="h-5 w-5" />
            <span>Failed to load profile: {profileError}</span>
          </CardContent>
        </Card>
      </div>
    );
  }

  // No data state
  if (!userProfile) {
    return (
      <div className="p-4 space-y-4">
        <h2>Profile</h2>
        <Card>
          <CardContent className="p-6 text-center text-muted-foreground">
            No profile data available
          </CardContent>
        </Card>
      </div>
    );
  }

  const profile = userProfile.profile;
  const addresses = userProfile.addresses;
  const primaryAddress = addresses[0]; // Use first address as primary

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2>Profile</h2>
        <Button onClick={refreshProfile} size="sm" variant="outline" disabled={isLoadingProfile} className="shadow-md">
          <RefreshCw className={`h-4 w-4 mr-2 ${isLoadingProfile ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* User Info Card */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center space-x-4">
            <Avatar className="h-16 w-16">
              <AvatarFallback>
                <User className="h-8 w-8" />
              </AvatarFallback>
            </Avatar>
            <div className="space-y-1">
              <h3 className="text-lg font-semibold">
                {profile.first_name && profile.last_name 
                  ? `${profile.first_name} ${profile.last_name}`
                  : profile.email
                }
              </h3>
              <Badge className={getStatusColor('active')}>
                Active
              </Badge>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Contact Information */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Mail className="h-4 w-4" />
            Contact Information
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm">
              <Mail className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Email:</span>
              <span>{profile.email}</span>
            </div>
            {profile.phone && (
              <div className="flex items-center gap-2 text-sm">
                <Phone className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Phone:</span>
                <span>{profile.phone}</span>
              </div>
            )}
            <div className="flex items-center gap-2 text-sm">
              <Calendar className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Member since:</span>
              <span>{new Date(profile.created_at).toLocaleDateString()}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Address Information */}
      {primaryAddress && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MapPin className="h-4 w-4" />
              Home Address
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1 text-sm">
              <p>{primaryAddress.address}</p>
              <p>{primaryAddress.city}, {primaryAddress.state} {primaryAddress.postal_code}</p>
              <p>{primaryAddress.country}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Order Statistics */}
      <Card>
        <CardHeader>
          <CardTitle>Order Statistics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">{userProfile.total_orders}</div>
              <div className="text-sm text-muted-foreground">Total Orders</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">${userProfile.total_spent.toFixed(2)}</div>
              <div className="text-sm text-muted-foreground">Total Spent</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
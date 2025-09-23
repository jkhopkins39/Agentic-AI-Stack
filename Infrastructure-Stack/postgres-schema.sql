CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(20),
    correlation_id UUID DEFAULT uuid_generate_v4(), -- For end-to-end tracing
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create user_addresses table
CREATE TABLE user_addresses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    address VARCHAR(255) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100),
    postal_code VARCHAR(20) NOT NULL,
    country VARCHAR(100) NOT NULL,
    correlation_id UUID, -- Links to user's correlation_id
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create products table
CREATE TABLE products (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    stock_quantity INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create orders table
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_number VARCHAR(50) NOT NULL UNIQUE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    total_amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    address_id UUID REFERENCES user_addresses(id),
    correlation_id UUID NOT NULL DEFAULT uuid_generate_v4(), -- Unique per order journey
    validation_status VARCHAR(50) DEFAULT 'pending', -- For order validation workflow
    validation_errors TEXT[], -- Array of validation error messages
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    shipped_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ
);

-- Create order_items table
CREATE TABLE order_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES products(id) ON DELETE RESTRICT,
    quantity INT NOT NULL CHECK (quantity > 0),
    unit_price DECIMAL(10,2) NOT NULL,
    correlation_id UUID, -- Links to order's correlation_id
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create queries table for customer support/AI interactions
CREATE TABLE queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    agent_response TEXT,
    correlation_id UUID NOT NULL DEFAULT uuid_generate_v4(), -- Unique per query journey
    related_order_id UUID REFERENCES orders(id), -- Link queries to specific orders
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create email_notifications table for tracking sent emails
CREATE TABLE email_notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID NOT NULL, -- Links to the originating event
    recipient_email VARCHAR(255) NOT NULL,
    email_type VARCHAR(50) NOT NULL, -- order_confirmed, order_shipped, etc.
    subject VARCHAR(255) NOT NULL,
    template_id VARCHAR(100), -- SendGrid template ID
    template_data JSONB, -- Dynamic data for the template
    sendgrid_message_id VARCHAR(255), -- SendGrid's message ID
    status VARCHAR(50) DEFAULT 'pending', -- pending, sent, failed, delivered
    sent_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_correlation_id ON users(correlation_id);
CREATE INDEX idx_user_addresses_user_id ON user_addresses(user_id);
CREATE INDEX idx_user_addresses_correlation_id ON user_addresses(correlation_id);
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_validation_status ON orders(validation_status);
CREATE INDEX idx_orders_correlation_id ON orders(correlation_id);
CREATE INDEX idx_orders_created_at ON orders(created_at);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
CREATE INDEX idx_order_items_correlation_id ON order_items(correlation_id);
CREATE INDEX idx_queries_user_id ON queries(user_id);
CREATE INDEX idx_queries_status ON queries(status);
CREATE INDEX idx_queries_correlation_id ON queries(correlation_id);
CREATE INDEX idx_queries_related_order_id ON queries(related_order_id);
CREATE INDEX idx_queries_created_at ON queries(created_at);
CREATE INDEX idx_email_notifications_correlation_id ON email_notifications(correlation_id);
CREATE INDEX idx_email_notifications_status ON email_notifications(status);
CREATE INDEX idx_email_notifications_email_type ON email_notifications(email_type);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$ language 'plpgsql';

-- Trigger to auto-populate correlation_id in related tables
CREATE OR REPLACE FUNCTION populate_correlation_id()
RETURNS TRIGGER AS $
BEGIN
    -- For user_addresses, use the user's correlation_id
    IF TG_TABLE_NAME = 'user_addresses' AND NEW.correlation_id IS NULL THEN
        SELECT correlation_id INTO NEW.correlation_id FROM users WHERE id = NEW.user_id;
    END IF;
    
    -- For order_items, use the order's correlation_id
    IF TG_TABLE_NAME = 'order_items' AND NEW.correlation_id IS NULL THEN
        SELECT correlation_id INTO NEW.correlation_id FROM orders WHERE id = NEW.order_id;
    END IF;
    
    RETURN NEW;
END;
$ language 'plpgsql';

-- Apply updated_at triggers to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_addresses_updated_at BEFORE UPDATE ON user_addresses 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_products_updated_at BEFORE UPDATE ON products 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_queries_updated_at BEFORE UPDATE ON queries 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_email_notifications_updated_at BEFORE UPDATE ON email_notifications 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Apply correlation_id population triggers
CREATE TRIGGER populate_user_addresses_correlation_id BEFORE INSERT ON user_addresses 
    FOR EACH ROW EXECUTE FUNCTION populate_correlation_id();

CREATE TRIGGER populate_order_items_correlation_id BEFORE INSERT ON order_items 
    FOR EACH ROW EXECUTE FUNCTION populate_correlation_id();

-- Add some sample data for testing with correlation IDs
INSERT INTO users (email, password_hash, first_name, last_name, phone) VALUES
('john.doe@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6MJo/5fF3y', 'John', 'Doe', '555-0123'),
('jane.smith@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6MJo/5fF3y', 'Jane', 'Smith', '555-0124');

INSERT INTO products (name, description, price, stock_quantity) VALUES
('Laptop Computer', 'High-performance laptop for work and gaming', 999.99, 50),
('Wireless Headphones', 'Noise-cancelling wireless headphones', 299.99, 100),
('Coffee Mug', 'Ceramic coffee mug with company logo', 19.99, 200);

-- Grant necessary permissions for replication (needed for Kafka Connect)
ALTER USER AgenticAIStackDB WITH REPLICATION;

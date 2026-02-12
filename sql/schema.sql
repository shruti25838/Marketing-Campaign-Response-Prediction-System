-- Example schema for marketing campaign response data
CREATE TABLE IF NOT EXISTS campaign_data (
    customer_id BIGINT,
    campaign_id BIGINT,
    campaign_channel VARCHAR(50),
    customer_segment VARCHAR(50),
    age INT,
    income NUMERIC(12, 2),
    signup_date DATE,
    purchase_amount NUMERIC(12, 2),
    response INT
);

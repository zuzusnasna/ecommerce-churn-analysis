-- raw transaction table
CREATE TABLE raw_transactions (
    invoice_no     VARCHAR(20),
    customer_id    VARCHAR(20),
    invoice_date   TIMESTAMP,
    quantity       INT,
    unit_price     DECIMAL(10,2),
    country        VARCHAR(50)
);


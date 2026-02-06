WITH customer_orders AS (
    SELECT
        customer_id,
        COUNT(DISTINCT invoice_no) AS frequency,
        SUM(quantity * unit_price) AS monetary,
        MAX(invoice_date) AS last_purchase_date
    FROM raw_transactions
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id
),
reference_date AS (
    SELECT MAX(invoice_date) AS ref_date
    FROM raw_transactions
)

SELECT
    c.customer_id,
    DATEDIFF(r.ref_date, c.last_purchase_date) AS recency,
    c.frequency,
    c.monetary
FROM customer_orders c
CROSS JOIN reference_date r;

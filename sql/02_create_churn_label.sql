-- 고객별 마지막 구매일
WITH last_purchase AS (
    SELECT
        customer_id,
        MAX(invoice_date) AS last_purchase_date
    FROM raw_transactions
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id
),

-- 데이터 기준 마지막 날짜
reference_date AS (
    SELECT
        MAX(invoice_date) AS ref_date
    FROM raw_transactions
)

SELECT
    lp.customer_id,
    lp.last_purchase_date,
    r.ref_date,
    CASE
        WHEN DATEDIFF(r.ref_date, lp.last_purchase_date) >= 60
        THEN 1
        ELSE 0
    END AS churn
FROM last_purchase lp
CROSS JOIN reference_date r;

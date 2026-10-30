
-- Q1. Customers whose completed orders above INR 500 total more than INR 8,000
SELECT
    customer_id,
    SUM(order_amount) AS total_amount
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_amount DESC;


-- Q2. Completed revenue by customer city and product category
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN products AS p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- Q3. Products priced above the average catalog price
SELECT
    product_id,
    product_name,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;


-- Q4. Customers with no completed orders
SELECT
    c.customer_id,
    c.city
FROM customers AS c
LEFT JOIN orders AS o
    ON o.customer_id = c.customer_id
   AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;




-- Q5. Customers whose completed spend is above average
WITH customer_spend AS (
    SELECT
        c.customer_id,
        SUM(o.order_amount) AS total_spend
    FROM customers AS c
    JOIN orders AS o
        ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id
)
SELECT
    customer_id,
    ROUND(total_spend, 2) AS total_spend,
    ROUND((SELECT AVG(total_spend) FROM customer_spend), 2)
        AS average_customer_spend
        
FROM customer_spend
WHERE total_spend > (SELECT AVG(total_spend) FROM customer_spend)
ORDER BY total_spend DESC;





-- Q6. Customer spend rankings within each city
WITH customer_spend AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend
    FROM customers AS c
    JOIN orders AS o
        ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    customer_id,
    city,
    ROUND(total_spend, 2) AS total_spend,
    RANK() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS spend_rank,
    DENSE_RANK() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS dense_spend_rank,
    ROW_NUMBER() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS row_num
FROM customer_spend
ORDER BY city, spend_rank, customer_id;



-- Q7. Daily completed revenue and cumulative revenue
WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS day,
        SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    day,
    ROUND(revenue, 2) AS daily_revenue_inr,
    ROUND(
        SUM(revenue) OVER (
            ORDER BY day
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS cumulative_revenue_inr
FROM daily_revenue
ORDER BY day;

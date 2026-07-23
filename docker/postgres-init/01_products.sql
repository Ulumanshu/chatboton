-- Demo data for the query_postgres tool: a tiny gadget-store product catalog.
-- Runs automatically on the first `docker compose up` (empty data volume).
CREATE TABLE products (
    id         SERIAL PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT NOT NULL,
    price_eur  NUMERIC(10, 2) NOT NULL,
    stock      INTEGER NOT NULL
);

INSERT INTO products (name, category, price_eur, stock) VALUES
    ('Volta Powerbank 20k',   'power',    39.90, 120),
    ('Nimbus Wireless Mouse', 'input',    24.50,  75),
    ('Krakla Mech Keyboard',  'input',    89.00,  30),
    ('Aurix 4K Webcam',       'video',    64.99,  18),
    ('Sonar BT Speaker',      'audio',    54.00,  52),
    ('Piksel USB-C Hub',      'power',    32.75,  90);

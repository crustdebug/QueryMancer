"""Build a demo database for trying QueryMancer against.

Creates a small e-commerce schema with realistic, self-consistent data:
customers place orders, orders contain items, items reference products. The
foreign keys matter as much as the rows - they are what the agent follows when
a question spans tables, and what schema pruning uses to pull in join targets.

Usage:

    python seed_demo.py "postgresql://user:password@host/dbname"

or set DEMO_DATABASE_URL and run it with no arguments. Works against
PostgreSQL, MySQL or SQLite; pass a path ending in .db for SQLite.

Deliberately deterministic (fixed seed), so re-running it produces the same
database and any demo you record stays reproducible.

This DROPS and recreates its tables. It refuses to run against a database that
holds tables it does not recognise, so pointing it at the wrong URL cannot
destroy real data.
"""

import random
import sys
from datetime import date, timedelta

from sqlalchemy import create_engine, inspect, text

SEED = 20240815

# Every table this script owns. Anything else present means the target is not
# a scratch database, and the script stops rather than dropping it.
OWNED_TABLES = ["order_items", "orders", "products", "customers", "suppliers"]

COMPANIES = [
    "Acme Corporation", "Globex", "Initech", "Umbrella Health", "Soylent Foods",
    "Stark Industries", "Wayne Enterprises", "Cyberdyne Systems", "Tyrell Corp",
    "Massive Dynamic", "Vehement Capital", "Wonka Industries", "Gringotts Bank",
    "Duff Brewing", "Bluth Company", "Pied Piper", "Hooli", "Vandelay Imports",
    "Sterling Cooper", "Prestige Worldwide", "Dunder Mifflin", "Genco Olive Oil",
    "Nakatomi Trading", "Weyland Industries", "Aperture Science", "Black Mesa",
    "Oscorp", "Abstergo", "Virtucon", "Rekall",
]

COUNTRIES = [
    ("United States", "USA"), ("Germany", "DEU"), ("United Kingdom", "GBR"),
    ("France", "FRA"), ("Japan", "JPN"), ("Canada", "CAN"), ("Australia", "AUS"),
    ("India", "IND"), ("Brazil", "BRA"), ("Netherlands", "NLD"),
]

SEGMENTS = ["Enterprise", "Mid-Market", "Small Business"]
ORDER_STATUSES = ["delivered", "delivered", "delivered", "shipped", "processing", "cancelled"]

CATEGORIES = {
    "Laptops": [("UltraBook Pro 14", 1899), ("UltraBook Air 13", 1199), ("WorkStation 17", 2799)],
    "Monitors": [("QHD Display 27", 449), ("4K Display 32", 799), ("Portable Display 15", 249)],
    "Peripherals": [("Mechanical Keyboard", 129), ("Wireless Mouse", 59),
                    ("Webcam 4K", 189), ("USB-C Dock", 219)],
    "Audio": [("Noise-Cancelling Headset", 349), ("Conference Speaker", 279),
              ("Studio Microphone", 159)],
    "Storage": [("NVMe Drive 1TB", 139), ("NVMe Drive 2TB", 249),
                ("Portable SSD 1TB", 119), ("NAS Enclosure", 529)],
    "Networking": [("Mesh Router", 299), ("Network Switch 8-Port", 89),
                   ("USB Ethernet Adapter", 39)],
}

SUPPLIERS = [
    ("Pacific Components", "Taiwan"), ("Nordic Supply Chain", "Sweden"),
    ("Shenzhen Electronics", "China"), ("Bavarian Precision", "Germany"),
    ("Great Lakes Distribution", "United States"),
]

# Portable across PostgreSQL, MySQL and SQLite: no SERIAL, no AUTO_INCREMENT,
# no dialect-specific types. Ids are assigned explicitly below.
SCHEMA = [
    """
    CREATE TABLE suppliers (
        supplier_id   INTEGER PRIMARY KEY,
        supplier_name VARCHAR(120) NOT NULL,
        country       VARCHAR(60),
        lead_time_days INTEGER
    )
    """,
    """
    CREATE TABLE customers (
        customer_id   INTEGER PRIMARY KEY,
        company_name  VARCHAR(120) NOT NULL,
        contact_name  VARCHAR(120),
        email         VARCHAR(160),
        country       VARCHAR(60),
        country_code  VARCHAR(3),
        segment       VARCHAR(40),
        signup_date   DATE,
        is_active     INTEGER
    )
    """,
    """
    CREATE TABLE products (
        product_id    INTEGER PRIMARY KEY,
        sku           VARCHAR(20) NOT NULL,
        product_name  VARCHAR(140) NOT NULL,
        category      VARCHAR(60),
        unit_price    DECIMAL(10,2),
        units_in_stock INTEGER,
        discontinued  INTEGER,
        supplier_id   INTEGER REFERENCES suppliers(supplier_id)
    )
    """,
    """
    CREATE TABLE orders (
        order_id      INTEGER PRIMARY KEY,
        customer_id   INTEGER REFERENCES customers(customer_id),
        order_date    DATE,
        shipped_date  DATE,
        status        VARCHAR(30),
        shipping_cost DECIMAL(10,2)
    )
    """,
    """
    CREATE TABLE order_items (
        order_item_id INTEGER PRIMARY KEY,
        order_id      INTEGER REFERENCES orders(order_id),
        product_id    INTEGER REFERENCES products(product_id),
        quantity      INTEGER,
        unit_price    DECIMAL(10,2),
        discount      DECIMAL(4,2)
    )
    """,
]


def build_rows(rng):
    """Generate the whole dataset in memory, consistently."""
    suppliers = [
        {
            "supplier_id": i,
            "supplier_name": name,
            "country": country,
            "lead_time_days": rng.choice([7, 14, 21, 30, 45]),
        }
        for i, (name, country) in enumerate(SUPPLIERS, start=1)
    ]

    products = []
    pid = 1
    for category, items in CATEGORIES.items():
        for name, price in items:
            products.append(
                {
                    "product_id": pid,
                    "sku": f"{category[:3].upper()}-{pid:04d}",
                    "product_name": name,
                    "category": category,
                    "unit_price": price,
                    "units_in_stock": rng.randint(0, 400),
                    # A few discontinued products make "which products are
                    # still sold?" a question with a real answer.
                    "discontinued": 1 if rng.random() < 0.12 else 0,
                    "supplier_id": rng.randint(1, len(suppliers)),
                }
            )
            pid += 1

    today = date.today()
    customers = []
    for i, company in enumerate(COMPANIES, start=1):
        country, code = rng.choice(COUNTRIES)
        customers.append(
            {
                "customer_id": i,
                "company_name": company,
                "contact_name": rng.choice(
                    ["Alice Nguyen", "Ben Carter", "Chloe Dubois", "Daniel Weber",
                     "Elena Rossi", "Farid Haddad", "Grace Okafor", "Hiro Tanaka",
                     "Ines Silva", "Jonas Berg"]
                ),
                "email": f"contact@{company.split()[0].lower().replace(',', '')}.example",
                "country": country,
                "country_code": code,
                "segment": rng.choice(SEGMENTS),
                "signup_date": today - timedelta(days=rng.randint(120, 1400)),
                "is_active": 1 if rng.random() < 0.85 else 0,
            }
        )

    orders, order_items = [], []
    order_id, item_id = 1, 1
    for customer in customers:
        # Enterprise customers order more: this gives group-by questions a
        # result with actual variation rather than noise.
        weight = {"Enterprise": 18, "Mid-Market": 9, "Small Business": 4}[customer["segment"]]
        for _ in range(rng.randint(1, weight)):
            placed = today - timedelta(days=rng.randint(1, 500))
            status = rng.choice(ORDER_STATUSES)
            shipped = None
            if status in ("delivered", "shipped"):
                shipped = placed + timedelta(days=rng.randint(1, 9))
            orders.append(
                {
                    "order_id": order_id,
                    "customer_id": customer["customer_id"],
                    "order_date": placed,
                    "shipped_date": shipped,
                    "status": status,
                    "shipping_cost": round(rng.uniform(5, 85), 2),
                }
            )
            for product in rng.sample(products, rng.randint(1, 5)):
                order_items.append(
                    {
                        "order_item_id": item_id,
                        "order_id": order_id,
                        "product_id": product["product_id"],
                        "quantity": rng.randint(1, 12),
                        "unit_price": product["unit_price"],
                        "discount": rng.choice([0, 0, 0, 0.05, 0.1, 0.15]),
                    }
                )
                item_id += 1
            order_id += 1

    return suppliers, customers, products, orders, order_items


def _existing_tables(engine):
    return set(inspect(engine).get_table_names())


def seed(url: str) -> None:
    engine = create_engine(url)

    present = _existing_tables(engine)
    unknown = present - set(OWNED_TABLES)
    if unknown:
        # The script drops tables, so a target holding anything unfamiliar is
        # treated as a real database and left alone.
        raise SystemExit(
            "Refusing to seed: this database contains tables this script does "
            f"not own ({', '.join(sorted(unknown))}).\n"
            "Point it at an empty database, or drop those tables first."
        )

    rng = random.Random(SEED)
    suppliers, customers, products, orders, order_items = build_rows(rng)

    with engine.begin() as conn:
        # Children first, so foreign keys never block the drop.
        for table in OWNED_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        for statement in SCHEMA:
            conn.execute(text(statement))

        inserts = [
            ("suppliers", suppliers), ("customers", customers), ("products", products),
            ("orders", orders), ("order_items", order_items),
        ]
        for table, rows in inserts:
            if not rows:
                continue
            columns = list(rows[0])
            placeholders = ", ".join(f":{c}" for c in columns)
            statement = text(
                f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            )
            # executemany: one round trip per batch rather than per row, which
            # matters over a network connection to a hosted database.
            conn.execute(statement, rows)

    print("Demo database ready:")
    for table, rows in [
        ("suppliers", suppliers), ("customers", customers), ("products", products),
        ("orders", orders), ("order_items", order_items),
    ]:
        print(f"  {table:<12} {len(rows):>6,} rows")

    revenue = sum(
        i["quantity"] * float(i["unit_price"]) * (1 - float(i["discount"]))
        for i in order_items
    )
    print(f"\nTotal order value in the data: {revenue:,.2f}")
    print("\nTry asking:")
    print('  "Which 5 customers have spent the most?"')
    print('  "What is total revenue by product category?"')
    print('  "How many orders were cancelled, by country?"')
    print('  "Which suppliers have the longest lead times?"')


def main() -> None:
    import os

    url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DEMO_DATABASE_URL", "")
    if not url:
        raise SystemExit(
            "Usage: python seed_demo.py <database-url>\n"
            "  e.g. python seed_demo.py "
            '"postgresql://user:pw@host.neon.tech/neondb?sslmode=require"\n'
            "  or:  python seed_demo.py demo.db        (SQLite)"
        )

    # A bare path is a convenience for local SQLite testing.
    if url.endswith(".db") and "://" not in url:
        url = f"sqlite:///{url}"
    # Neon and most hosted Postgres hand out postgres:// URLs, which
    # SQLAlchemy 2 does not accept.
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    seed(url)


if __name__ == "__main__":
    main()

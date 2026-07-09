"""
Order Lookup Tool
------------------
Simulates a backend order/account lookup system. In production this
would hit Shizaye Multilinks' real database (e.g. the SimShop Firebase
backend). For the capstone, it queries a local CSV.
"""

import pandas as pd

_orders_df = None


def _load_orders(csv_path="data/orders.csv"):
    global _orders_df
    if _orders_df is None:
        _orders_df = pd.read_csv(csv_path, dtype={"phone": str, "order_id": str})
    return _orders_df


def lookup_order(order_id: str = None, phone: str = None, csv_path: str = "data/orders.csv") -> str:
    """
    Look up an order by order ID or customer phone number.

    Args:
        order_id: The order ID (e.g. "ORD1001").
        phone: The customer's phone number, used if order_id is not given.
        csv_path: Path to the orders CSV file.

    Returns:
        A formatted string describing the order, or a not-found message.
    """
    df = _load_orders(csv_path)

    if order_id:
        match = df[df["order_id"].str.upper() == order_id.strip().upper()]
    elif phone:
        match = df[df["phone"] == phone.strip()]
    else:
        return "Please provide an order ID or phone number to look up an order."

    if match.empty:
        return "No matching order found. Please double-check the order ID or phone number."

    row = match.iloc[0]
    return (
        f"Order {row['order_id']} for {row['customer_name']}:\n"
        f"Item: {row['item']}\n"
        f"Status: {row['status']}\n"
        f"Branch: {row['branch']}\n"
        f"Date: {row['date']}"
    )


if __name__ == "__main__":
    print(lookup_order(order_id="ORD1002"))
    print(lookup_order(phone="08031234567"))

#!/usr/bin/env python3
"""
E-Commerce Profitability & Recommendation - single-file runnable script.

How to run:
  python main.py                    # prints profitability summary and sample recommendations
  python main.py --user 1 --top 5   # shows top-5 recommendations for user id 1

If CSVs (products.csv, transactions.csv, users.csv) are not found, sample datasets
will be created automatically so you can run this immediately.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Filenames
PRODUCTS_CSV = "products.csv"
TRANSACTIONS_CSV = "transactions.csv"
USERS_CSV = "users.csv"

def create_sample_data():
    """Creates small sample CSV files if they don't exist."""
    if not os.path.exists(PRODUCTS_CSV):
        products = pd.DataFrame([
            {"product_id": 1, "product_name": "Classic T-Shirt", "cost": 6.00, "price": 20.00, "category": "Apparel"},
            {"product_id": 2, "product_name": "Sneakers", "cost": 25.00, "price": 90.00, "category": "Footwear"},
            {"product_id": 3, "product_name": "Baseball Cap", "cost": 3.50, "price": 15.00, "category": "Accessories"},
            {"product_id": 4, "product_name": "Jeans", "cost": 18.00, "price": 60.00, "category": "Apparel"},
            {"product_id": 5, "product_name": "Hoodie", "cost": 12.00, "price": 45.00, "category": "Apparel"},
        ])
        products.to_csv(PRODUCTS_CSV, index=False)
        print(f"Created sample {PRODUCTS_CSV}")

    if not os.path.exists(USERS_CSV):
        users = pd.DataFrame([
            {"user_id": 1, "user_name": "Aisha"},
            {"user_id": 2, "user_name": "Ben"},
            {"user_id": 3, "user_name": "Carla"},
            {"user_id": 4, "user_name": "Davis"},
        ])
        users.to_csv(USERS_CSV, index=False)
        print(f"Created sample {USERS_CSV}")

    if not os.path.exists(TRANSACTIONS_CSV):
        # transaction_id, user_id, product_id, quantity, timestamp
        tx = pd.DataFrame([
            {"transaction_id": 1, "user_id": 1, "product_id": 1, "quantity": 2, "timestamp": "2025-01-10"},
            {"transaction_id": 2, "user_id": 1, "product_id": 3, "quantity": 1, "timestamp": "2025-01-11"},
            {"transaction_id": 3, "user_id": 2, "product_id": 2, "quantity": 1, "timestamp": "2025-01-12"},
            {"transaction_id": 4, "user_id": 2, "product_id": 1, "quantity": 1, "timestamp": "2025-01-13"},
            {"transaction_id": 5, "user_id": 3, "product_id": 4, "quantity": 2, "timestamp": "2025-01-14"},
            {"transaction_id": 6, "user_id": 3, "product_id": 5, "quantity": 1, "timestamp": "2025-01-15"},
            {"transaction_id": 7, "user_id": 4, "product_id": 2, "quantity": 1, "timestamp": "2025-01-16"},
            {"transaction_id": 8, "user_id": 4, "product_id": 5, "quantity": 2, "timestamp": "2025-01-17"},
            {"transaction_id": 9, "user_id": 2, "product_id": 3, "quantity": 3, "timestamp": "2025-01-18"},
        ])
        tx.to_csv(TRANSACTIONS_CSV, index=False)
        print(f"Created sample {TRANSACTIONS_CSV}")

def load_data():
    """Loads CSVs into DataFrames (expects products, transactions, users)."""
    products = pd.read_csv(PRODUCTS_CSV)
    transactions = pd.read_csv(TRANSACTIONS_CSV)
    users = pd.read_csv(USERS_CSV)
    return products, transactions, users

def profitability_analysis(products: pd.DataFrame, transactions: pd.DataFrame):
    """
    Compute profit per transaction and aggregate profit per product.
    profit_per_unit = price - cost
    total_profit = profit_per_unit * quantity
    """
    merged = transactions.merge(products, on="product_id", how="left")
    merged["profit_per_unit"] = merged["price"] - merged["cost"]
    merged["total_profit"] = merged["profit_per_unit"] * merged["quantity"]

    # Aggregations
    product_profit = merged.groupby(["product_id", "product_name", "category"]).agg(
        total_units_sold = ("quantity", "sum"),
        total_revenue = ("price", lambda s: (s * merged.loc[s.index, "quantity"]).sum()),
        total_cost = ("cost", lambda s: (s * merged.loc[s.index, "quantity"]).sum()),
        total_profit = ("total_profit", "sum")
    ).reset_index()

    # Profit margin for products
    product_profit["profit_margin_percent"] = product_profit["total_profit"] / product_profit["total_revenue"]
    product_profit = product_profit.sort_values("total_profit", ascending=False)
    return product_profit, merged

def build_item_user_matrix(transactions: pd.DataFrame, products: pd.DataFrame, users: pd.DataFrame):
    """
    Build a user x item purchase matrix (counts). Rows: users, Columns: products.
    """
    # pivot table of counts
    pivot = transactions.pivot_table(index="user_id", columns="product_id", values="quantity", aggfunc="sum", fill_value=0)
    # Ensure all products included
    for pid in products["product_id"]:
        if pid not in pivot.columns:
            pivot[pid] = 0
    # Reorder columns by product_id
    pivot = pivot.reindex(sorted(pivot.columns), axis=1).sort_index()
    # Add missing users
    for uid in users["user_id"]:
        if uid not in pivot.index:
            pivot.loc[uid] = [0] * pivot.shape[1]
    pivot = pivot.sort_index()
    return pivot

def item_based_recommender(purchase_matrix: pd.DataFrame, product_meta: pd.DataFrame, user_id: int, top_n=5):
    """
    Simple item-based collaborative filtering:
      - compute cosine similarity between item vectors (columns)
      - score items for user by weighted sum of items they purchased
    """
    # transpose to have items x users for similarity between items (rows)
    item_user = purchase_matrix.T  # items x users
    if item_user.shape[0] == 0:
        return []

    # If an item has zero vector, cosine_similarity may give nan; add small epsilon
    item_user_vals = item_user.values.astype(float) + 1e-9
    sim = cosine_similarity(item_user_vals)  # item x item

    # item ids aligned with item_user.index
    item_ids = list(item_user.index)

    # user vector (products)
    if user_id not in purchase_matrix.index:
        print(f"User {user_id} not found in purchase_matrix (cold start). No personalized recs.")
        return []

    user_vector = purchase_matrix.loc[user_id].values  # product quantities

    # Score each item: sum(similarity(item, j) * user_vector[j])
    scores = sim.dot(user_vector)

    # Build DataFrame of scores
    score_df = pd.DataFrame({
        "product_id": item_ids,
        "score": scores
    }).merge(product_meta[["product_id", "product_name", "category"]], on="product_id", how="left")

    # Remove items user already bought (optional — here we filter those with quantity>0)
    already_bought = set(purchase_matrix.loc[user_id][purchase_matrix.loc[user_id] > 0].index.tolist())
    score_df = score_df[~score_df["product_id"].isin(already_bought)]

    score_df = score_df.sort_values("score", ascending=False).head(top_n)
    # Normalize score for readability
    if score_df["score"].max() > 0:
        score_df["score_normalized"] = score_df["score"] / (score_df["score"].max())
    else:
        score_df["score_normalized"] = 0.0
    return score_df.reset_index(drop=True)

def recommend_for_all_users(purchase_matrix, product_meta, top_n=3):
    """Return top-N recommendations for every user in purchase_matrix."""
    recs = {}
    for uid in purchase_matrix.index:
        rec_df = item_based_recommender(purchase_matrix, product_meta, uid, top_n=top_n)
        recs[uid] = rec_df
    return recs

def print_profitability_summary(product_profit: pd.DataFrame, top_k=5):
    print("\n=== Top products by total profit ===")
    display = product_profit.head(top_k)[["product_id", "product_name", "category", "total_units_sold", "total_revenue", "total_cost", "total_profit", "profit_margin_percent"]]
    # Format numeric columns
    pd.options.display.float_format = '{:,.2f}'.format
    print(display.to_string(index=False))

def main(args):
    create_sample_data()
    products, transactions, users = load_data()
    product_profit, merged = profitability_analysis(products, transactions)

    print_profitability_summary(product_profit, top_k=10)

    # Build pivot
    purchase_matrix = build_item_user_matrix(transactions, products, users)
    print("\nUser x Item purchase matrix (rows=users, cols=product_ids):")
    print(purchase_matrix)

    # Get recommendations for requested user (or sample users)
    target_user = args.user
    top_n = args.top
    if target_user is None:
        # show for all users a sample of top 3
        print("\n===== Sample recommendations for all users (top 3) =====")
        recs = recommend_for_all_users(purchase_matrix, products, top_n=3)
        for uid, df in recs.items():
            print(f"\nUser {uid}:")
            if df is None or df.shape[0] == 0:
                print("  (no recommendations / user cold-start or already bought everything)")
            else:
                print(df[["product_id", "product_name", "category", "score_normalized"]].to_string(index=False))
    else:
        print(f"\n===== Recommendations for user {target_user} (top {top_n}) =====")
        df = item_based_recommender(purchase_matrix, products, target_user, top_n=top_n)
        if df is None or df.shape[0] == 0:
            print("No recommendations found (cold-start or not enough data).")
        else:
            print(df[["product_id", "product_name", "category", "score_normalized"]].to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-Commerce profitability and recommender demo.")
    parser.add_argument("--user", type=int, help="user_id to get recommendations for (optional)")
    parser.add_argument("--top", type=int, default=5, help="top-N recommendations to return")
    args = parser.parse_args()
    main(args)

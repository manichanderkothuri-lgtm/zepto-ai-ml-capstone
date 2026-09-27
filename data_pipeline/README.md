# Module 1 — Data Pipeline

Run `python pipeline.py`. The script scrapes at least 60 books from Books to Scrape, cleans the required fields, converts GBP to INR at the fixed assignment rate **1 GBP = 105.50 INR**, writes `books_clean.csv`, creates normalized SQLite tables `categories` and `books`, and writes SQL/pandas outputs.

Cleaning decisions: numeric parsing failures in `price_gbp` and `rating` are median-imputed; rows missing essential title/category fields are dropped. `in_stock` is parsed from the presence of “In stock”.

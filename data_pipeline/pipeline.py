import re, sqlite3, logging
from pathlib import Path
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE='https://books.toscrape.com/'
OUT=Path(__file__).parent
DB=OUT/'books.db'
CSV=OUT/'books_clean.csv'
RATE=105.50
RATING_MAP={'One':1,'Two':2,'Three':3,'Four':4,'Five':5}
logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_soup(url):
    r=requests.get(url, timeout=20); r.raise_for_status(); return BeautifulSoup(r.text,'html.parser')

def scrape(min_books=60):
    rows=[]; url=BASE
    while url and len(rows)<min_books:
        soup=get_soup(url)
        for card in soup.select('article.product_pod'):
            title=card.h3.a.get('title','').strip()
            price=card.select_one('.price_color').get_text(strip=True)
            availability=card.select_one('.availability').get_text(' ',strip=True)
            rating_class=next((c for c in card.get('class',[]) if c in RATING_MAP),None)
            # rating is on the p.star-rating element, not article class
            sr=card.select_one('p.star-rating')
            rating_text=next((c for c in sr.get('class',[]) if c in RATING_MAP),'One') if sr else 'One'
            cat=card.select_one('ul.breadcrumb li:nth-of-type(3) a')
            category=cat.get_text(strip=True) if cat else 'Unknown'
            # category on listing page is not always in DOM; visit product page only when needed
            href=card.h3.a.get('href')
            if category=='Unknown':
                ps=get_soup(requests.compat.urljoin(url,href)); bc=ps.select('ul.breadcrumb li a')
                category=bc[-1].get_text(strip=True) if bc else 'Unknown'
            rows.append({'title':title,'price':price,'star_rating':rating_text,'availability':availability,'category':category})
            if len(rows)>=min_books: break
        nxt=soup.select_one('li.next a')
        url=requests.compat.urljoin(url,nxt['href']) if nxt and len(rows)<min_books else None
    return pd.DataFrame(rows)

def clean(df):
    out=df.copy()
    out['price_gbp']=pd.to_numeric(out['price'].str.replace('£','',regex=False),errors='coerce')
    out['rating']=out['star_rating'].map(RATING_MAP)
    out['in_stock']=out['availability'].str.contains('In stock',case=False,na=False)
    for c in ['price_gbp','rating']:
        out[c]=out[c].fillna(out[c].median())
    out=out.dropna(subset=['title','category'])
    out['price_inr']=out['price_gbp']*RATE
    return out[['title','price_gbp','price_inr','rating','in_stock','category']]

def load_sqlite(df):
    con=sqlite3.connect(DB); con.execute('PRAGMA foreign_keys=ON')
    con.executescript('''DROP TABLE IF EXISTS books; DROP TABLE IF EXISTS categories;
    CREATE TABLE categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE NOT NULL);
    CREATE TABLE books(book_id INTEGER PRIMARY KEY, title TEXT NOT NULL, price_gbp REAL, price_inr REAL, rating INTEGER, in_stock INTEGER, category_id INTEGER NOT NULL REFERENCES categories(category_id));''')
    cats=pd.DataFrame({'category_name':sorted(df.category.unique())})
    cats.to_sql('categories',con,if_exists='append',index=False)
    cmap=pd.read_sql_query('SELECT * FROM categories',con)
    b=df.merge(cmap,left_on='category',right_on='category_name').drop(columns=['category','category_name'])
    b['in_stock']=b['in_stock'].astype(int); b.to_sql('books',con,if_exists='append',index=False)
    queries={
      'q1_select_where':"SELECT title, price_gbp FROM books WHERE price_gbp > 20 ORDER BY price_gbp DESC;",
      'q2_order_limit':"SELECT title, rating, price_gbp FROM books ORDER BY rating DESC, price_gbp DESC LIMIT 10;",
      'q3_distinct':"SELECT DISTINCT category_name FROM categories ORDER BY category_name;",
      'q4_between':"SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 30 ORDER BY price_gbp;",
      'q5_join':"SELECT c.category_name, b.title, b.rating, b.price_inr FROM books b JOIN categories c ON b.category_id=c.category_id ORDER BY b.rating DESC, b.price_inr DESC LIMIT 10;"}
    lines=[]
    for name,q in queries.items():
        res=pd.read_sql_query(q,con); lines += [f'-- {name}\n{q}\n{res.to_string(index=False)}\n']
    (OUT/'sql_outputs.txt').write_text('\n'.join(lines),encoding='utf-8')
    join_sql=pd.read_sql_query(queries['q5_join'],con)
    join_pd=df.merge(cmap,left_on='category',right_on='category_name').rename(columns={'category_name':'category_name'})
    join_pd=join_pd[['category_name','title','rating','price_inr']].sort_values(['rating','price_inr'],ascending=[False,False]).head(10).reset_index(drop=True)
    (OUT/'pandas_join_output.txt').write_text('pd.read_sql result:\n'+join_sql.to_string(index=False)+'\n\npd.merge result:\n'+join_pd.to_string(index=False),encoding='utf-8')
    con.close()

if __name__=='__main__':
    raw=scrape(60); clean_df=clean(raw); clean_df.to_csv(CSV,index=False); load_sqlite(clean_df)
    logging.info('Completed: %s rows, %s categories. GBP->INR rate %.2f',len(clean_df),clean_df.category.nunique(),RATE)

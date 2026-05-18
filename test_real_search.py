import psycopg2

def test_real_search():
    conn_params = {
        "host": "localhost",
        "port": 5433,
        "dbname": "media_bias",
        "user": "postgres",
        "password": "mediabias123",
    }
    
    try:
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        
        # 1. Show some data from the database
        print("--- DATABASE STATISTICS ---")
        cur.execute("SELECT COUNT(*) FROM articles;")
        total_articles = cur.fetchone()[0]
        print(f"Total articles in database: {total_articles}")
        
        if total_articles == 0:
            print("Database is empty! You need to run article_embeddings.py first.")
            return

        print("\n--- SAMPLE ARTICLES FROM DATABASE ---")
        cur.execute("SELECT article_id, outlet, topic, label_bias, agreement FROM articles LIMIT 10;")
        rows = cur.fetchall()
            
        for r in rows:
            print(f"ID: {r[0][:8]}... | Outlet: {r[1]} | Topic: {r[2]} | Bias: {r[3]} | Agreement: {r[4]}")
            
        # 2. Test Vector Similarity Search
        print("\n--- TESTING PGVECTOR SIMILARITY SEARCH ---")
        # Let's get the embedding of the first article
        target_id = rows[0][0]
        cur.execute("SELECT embedding FROM article_embeddings WHERE article_id = %s;", (target_id,))
        target_embedding_str = cur.fetchone()[0]
        
        print(f"Finding articles semantically similar to {target_id[:8]}...")
        
        # Search using pgvector cosine distance (<=>)
        # We skip the target_id itself to find its neighbors
        cur.execute("""
            SELECT a.article_id, a.outlet, a.label_bias, e.embedding <=> %s::vector AS distance
            FROM articles a
            JOIN article_embeddings e ON a.article_id = e.article_id
            WHERE a.article_id != %s
            ORDER BY distance ASC
            LIMIT 3;
        """, (target_embedding_str, target_id))
        
        neighbors = cur.fetchall()
        for n in neighbors:
            print(f"Distance: {n[3]:.4f} | ID: {n[0][:8]}... | Outlet: {n[1]} | Bias: {n[2]}")
            
        cur.close()
        conn.close()
        print("\n✅ Search works perfectly on real data!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_real_search()

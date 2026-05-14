import chromadb
import json

def check_database(db_path="./vector_db", collection_name="article_embeddings"):
    print(f"--- Connecting to ChromaDB at {db_path} ---")
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        print(f"SUCCESS: Found collection '{collection_name}' with {count} articles.")
        
        if count > 0:
            print("\n--- Showing the first 3 articles in the DB ---")
            results = collection.peek(limit=3)
            for i in range(len(results['ids'])):
                print(f"\n[ID]: {results['ids'][i]}")
                print(f"[Snippet]: {results['documents'][i][:100]}...")
                
                # Parse the emotion scores we saved as a JSON string
                scores = json.loads(results['metadatas'][i]['emotion_scores'])
                print(f"[Emotions]: {scores}")
        else:
            print("\nThe database is currently empty. Run article_embeddings.py to fill it!")
            
    except Exception as e:
        print(f"ERROR: Could not find or read the collection. {e}")

if __name__ == "__main__":
    check_database()

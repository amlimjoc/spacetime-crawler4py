import json
import math
import time
from nltk.stem import PorterStemmer

from scraper import tokenize_text

stemmer = PorterStemmer()

INDEX_PATH = "inverted_index.json"
ID_MAP_PATH = "doc_id_map.json"

def load_data():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        inverted_index = json.load(f)
    with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
        doc_id_map = json.load(f)
    return inverted_index, doc_id_map

def binary_search_intersection(query: str, index: dict, doc_map: dict):

    query_tokens = tokenize_text(query)
    
    query_tokens = list(set(query_tokens)) 
    
    if not query_tokens:
        return []

    total_docs = len(doc_map)

    query_tokens = sorted(query_tokens, key=lambda t: len(index.get(t, [])))
    
    first_token = query_tokens[0]
    if first_token not in index:
        return []
        
    valid_doc_ids = set(posting[0] for posting in index[first_token])
    
    for token in query_tokens[1:]:
        if token not in index:
            return []
        current_doc_ids = set(posting[0] for posting in index[token])
        valid_doc_ids = valid_doc_ids.intersection(current_doc_ids)
        
        if not valid_doc_ids:
            return []

    doc_scores = {doc_id: 0.0 for doc_id in valid_doc_ids}
    
    for token in query_tokens:
        postings = index[token]
        df = len(postings)
        idf = math.log10(total_docs / df) if df > 0 else 0
        
        for doc_id, tf, is_imp in postings:
            if doc_id in valid_doc_ids:
                weight_multiplier = 1.5 if is_imp == 1 else 1.0
                
                score = tf * idf * weight_multiplier
                doc_scores[doc_id] += score

    ranked_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    
    top_5_results = []
    for doc_id, score in ranked_docs[:5]:
        url = doc_map[str(doc_id)]
        top_5_results.append((url, score))
        
    return top_5_results

def main():
    try:
        index, doc_map = load_data()
    except FileNotFoundError:
        return

    while True:
        query = input("Enter search query: ").strip()
        
        if not query:
            continue

        results = binary_search_intersection(query, index, doc_map)

        if not results:
            print("No matching documents found")
        else:
            for i, (url, score) in enumerate(results, 1):
                print(f"{i}. {url} Score: {score:.4f}")

if __name__ == "__main__":
    main()
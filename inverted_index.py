import json
import os
from scraper import tokenize_text

INDEX_PATH = "inverted_index.json"
# Put the DEV folder with all the web pages in the same directory as this script
DEV_FOLDER_PATH = "DEV"

def build_inverted_index(document_generator):
    '''
    creates an inverted index where each token maps to every doc it appears
    '''
    inverted_index = {}
    doc_count = 0

    for doc_id, content in document_generator:
        doc_count += 1
        tokens = tokenize_text(content)

        tf = calculate_tf(tokens)

        for token, score in tf.items():
            if token not in inverted_index:
                inverted_index[token] = []

            inverted_index[token].append((doc_id, score))

    return inverted_index, doc_count

def save_index(index, path=INDEX_PATH):
    '''saves the index to a json file'''
    with open(path, "w", encoding='utf-8') as f:
        json.dump(index, f)


def calculate_tf(doc_tokens):
    '''
    make dictionary containing each token and its tf value.
    '''
    tf = {}
    total_tokens = len(doc_tokens)
    
    if total_tokens == 0:
        return {}

    for token in doc_tokens:
        tf[token] = tf.get(token, 0) + 1

    for token in tf:
        tf[token] /= total_tokens

    return tf


def get_index_stats(index, doc_count):
    return {
        "num_documents": doc_count,
        "num_unique_tokens": len(index),
        "index_size_kb": os.path.getsize(INDEX_PATH) / 1024 if os.path.exists(INDEX_PATH) else 0
    }

def load_documents(dev_path):
    """
    Walks through the directory and yields (doc_id, content) for each JSON file.
    """
    for root, dirs, files in os.walk(dev_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                        doc_id = file_path
                        
                        content = data.get("content", "")
                        
                        if content:
                            yield doc_id, content
                            
                except Exception as e:
                    print(f"{e}")

def print_index_stats():
    '''build, save, and print stats'''
    document_generator = load_documents(DEV_FOLDER_PATH)

    index, doc_count = build_inverted_index(document_generator)
    
    save_index(index)

    stats = get_index_stats(index, doc_count)

    print(f"Indexed Documents: {stats['num_documents']}")
    print(f"Unique Tokens: {stats['num_unique_tokens']}")
    print(f"Index Size (KB): {stats['index_size_kb']:.2f}")

if __name__ == "__main__":
    print_index_stats()
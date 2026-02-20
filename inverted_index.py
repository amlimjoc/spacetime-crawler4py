import json
import os
from nltk.stem import PorterStemmer

INDEX_PATH = "inverted_index.json"
# Put the DEV folder with all the web pages in the same directory as this script
DEV_FOLDER_PATH = "DEV"
stemmer = PorterStemmer()

def tokenize_text(text: str):
    tokens = []
    word = []

    for ch in text:
        if ch.isalnum() and ch.isascii():
            word.append(ch.lower())
        elif ch == "'" and word:
            continue
        else:
            if word:
                tok = "".join(word)
                # don't add 1-letter tokens
                if 30 > len(tok) > 1:
                    # Apply stemming
                    stemmed_token = stemmer.stem(tok)
                    tokens.append(stemmed_token)
                word.clear()
    
    if word:
        tok = "".join(word)
        if 30 > len(tok) > 1:
            stemmed_token = stemmer.stem(tok)
            tokens.append(stemmed_token)
    
    return tokens

def calculate_tf(doc_tokens):
    tf = {}
    total_tokens = len(doc_tokens)
    if total_tokens == 0: return {}
    for token in doc_tokens:
        tf[token] = tf.get(token, 0) + 1
    return {token: count/total_tokens for token, count in tf.items()}

def build_inverted_index(document_generator):
    inverted_index = {}
    doc_id_map = {}
    current_doc_id = 0

    for url, content in document_generator:
        current_doc_id += 1
        
        if current_doc_id % 1000 == 0:
            print(f"{current_doc_id} docs")

        doc_id_map[current_doc_id] = url
        
        tokens = tokenize_text(content)
        tf_scores = calculate_tf(tokens)

        for token, score in tf_scores.items():
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append((current_doc_id, round(score, 5)))

    return inverted_index, doc_id_map

def load_documents(dev_path):
    for root, _, files in os.walk(dev_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        yield data.get("url", file_path), data.get("content", "")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

def print_index_stats():    
    document_generator = load_documents(DEV_FOLDER_PATH)
    index, doc_id_map = build_inverted_index(document_generator)
    
    with open(INDEX_PATH, "w", encoding='utf-8') as f:
        json.dump(index, f)

    idx_size_kb = os.path.getsize(INDEX_PATH) / 1024

    print(f"Number of indexed documents: {len(doc_id_map)}")
    print(f"Number of unique tokens: {len(index)}")
    print(f"Total size of index on disk: {idx_size_kb:.2f} KB")

if __name__ == "__main__":
    print_index_stats()
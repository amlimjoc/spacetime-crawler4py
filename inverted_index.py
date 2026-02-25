import json
import os
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

INDEX_PATH = "inverted_index.json"
ID_MAP_PATH = "doc_id_map.json"
DEV_FOLDER_PATH = "DEV"
stemmer = PorterStemmer()

def get_tokens_from_soup(soup):
    """
    extracts tokens and identifies important ones.
    """
    all_text = soup.get_text()
    all_tokens = tokenize_text(all_text)
    
    important_text = []
    for tag in soup.find_all(['title', 'h1', 'h2', 'h3', 'b', 'strong']):
        important_text.append(tag.get_text())
    
    important_tokens = set(tokenize_text(" ".join(important_text)))
    
    return all_tokens, important_tokens

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
                # ignore digits and very long junk
                if not tok.isdigit() and 1 < len(tok) < 30:
                    tokens.append(stemmer.stem(tok))
                word.clear()
    if word:
        tok = "".join(word)
        if not tok.isdigit() and 1 < len(tok) < 30:
            tokens.append(stemmer.stem(tok))
    return tokens

def build_inverted_index(document_generator):
    inverted_index = {}
    doc_id_map = {}
    current_doc_id = 0

    for url, html_content in document_generator:
        current_doc_id += 1
        if current_doc_id % 500 == 0:
            print(f"{current_doc_id} docs")

        doc_id_map[current_doc_id] = url
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        all_tokens, important_tokens = get_tokens_from_soup(soup)
        
        total_tokens = len(all_tokens)
        if total_tokens == 0: continue
        
        counts = {}
        for t in all_tokens:
            counts[t] = counts.get(t, 0) + 1

        for token, count in counts.items():
            tf_score = count / total_tokens
            
            is_imp = 1 if token in important_tokens else 0
            
            if token not in inverted_index:
                inverted_index[token] = []
            
            inverted_index[token].append((current_doc_id, round(tf_score, 5), is_imp))

    return inverted_index, doc_id_map

def load_documents(dev_path):
    for root, _, files in os.walk(dev_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        yield data.get("url"), data.get("content", "")
                except:
                    continue

def print_index_stats():    
    idx, id_map = build_inverted_index(load_documents(DEV_FOLDER_PATH))
    
    with open(INDEX_PATH, "w") as f: json.dump(idx, f)
    with open(ID_MAP_PATH, "w") as f: json.dump(id_map, f)

    print(f"\nDocuments: {len(id_map)}")
    print(f"Unique Tokens: {len(idx)}")
    print(f"Index Size: {os.path.getsize(INDEX_PATH) / 1024:.2f} KB")

if __name__ == "__main__":
    print_index_stats()
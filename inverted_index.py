import json
import os
from scraper import tokenize_text

INDEX_PATH = "inverted_index.json"

def build_inverted_index(documents):
    '''
    Takes a list of documents, creates an inverted index where each token maps to every doc it appears.

    :param documents: A dictionary of documents where a doc_id maps to it's content.

    Returns:
        A dictionary with each token and their respective docs.
    '''
    inverted_index = {}

    # for every document, get the tokens, get their respective tf value, create an inverted index
    for doc_id, content in documents.items():
        tokens = tokenize_text(content)

        tf = calculate_tf(tokens)

        for token, score in tf.items():
            if token not in inverted_index:
                inverted_index[token] = []

            inverted_index[token].append((doc_id, score))

        for token in inverted_index:
            inverted_index[token].sort(key=lambda x: x[0])
            
    return inverted_index

def save_index(index, path=INDEX_PATH):
    '''
    Docstring for save_index
    
    :param index: Description
    :param path: Description
    '''
    with open(path, "w") as f:
        json.dump(index, f)


def load_index(path=INDEX_PATH):
    '''
    Docstring for load_index
    
    :param path: Description
    '''
    with open(path, "r") as f:
        return json.load(f)


def calculate_tf(doc_tokens):
    '''
    Given a list of tokens from a document, return a dictionary containing each token and it's tf value.
    
    :param doc_tokens: A list of tokens.
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


def get_index_stats(index, documents):
    '''
    Docstring for get_index_stats
    
    :param index: Description
    :param documents: Description
    '''
    return {
        "num_documents": len(documents),
        "num_unique_tokens": len(index),
        "index_size_kb": os.path.getsize(INDEX_PATH) / 1024 if os.path.exists(INDEX_PATH) else 0
    }


def print_index_stats():
    '''
    Docstring for print_index_stats
    '''
    # Example placeholder documents loader
    documents = load_documents()   # must return {doc_id: content}

    index = build_inverted_index(documents)
    save_index(index)

    stats = get_index_stats(index, documents)

    print(f"Indexed Documents: {stats['num_documents']}")
    print(f"Unique Tokens: {stats['num_unique_tokens']}")
    print(f"Index Size (KB): {stats['index_size_kb']:.2f}")


def load_documents():
    """
    Replace with your dataset loader.
    Must return: {doc_id: content}
    """
    return {}


if __name__ == "__main__":
    print_index_stats()
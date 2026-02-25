from collections import Counter

from scraper import tokenize_text

import bisect


from inverted_index import load_index, build_inverted_index, get_index_stats

def search_query(query: str, inverted_index: dict):
    '''
    Finds relevant document IDs based on a search query.

    Arguments:
        query (str)
        inverted_index (dict)

    Returns:
        A set containing relevant documents to the query.
    '''
    query_tokens = tokenize_text(query)
    query_tokens = sorted(query_tokens, key=lambda token: len(inverted_index.get(token, [])))

    relevant_docs = set(inverted_index.get(query_tokens[0], []))
    
    for token in query_tokens[1:]:
        token_postings = inverted_index.get(token, [])
        
        relevant_docs = set(binary_search_intersection(list(relevant_docs), token_postings))
        
        if not relevant_docs:
            break
    
    return relevant_docs

def binary_search_intersection(list1, list2):
    '''
    Finds common document IDs between two sorted postings lists using binary search.

    Arguments:
        list1 (list): A sorted list of document IDs and term frequencies for a given term.
        list2 (list): A sorted list of document IDs and term frequencies for another term.

    Returns:
        list: A list containing document IDs that appear in both postings lists.
    '''
    result = []
    for item in list1:
        idx = bisect.bisect_left(list2, item)
        if idx < len(list2) and list2[idx] == item:
            result.append(item)
    return result
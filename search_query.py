import json
import math
import time
from urllib.parse import urlparse
from flask import Flask, request, render_template_string
from nltk.stem import PorterStemmer

from scraper import tokenize_text

app = Flask(__name__)
stemmer = PorterStemmer()

INDEX_PATH = "inverted_index.json"
INDEX_META_PATH = "inverted_index_meta.json"
ID_MAP_PATH = "doc_id_map.json"
PAGERANK_PATH = "pagerank.json"

index_meta = {}
doc_id_map = {}
pagerank_map = {}
pagerank_min = 1.0
index_file = None


def load_data():
    global index_meta, doc_id_map, pagerank_map, index_file
    
    with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
        index_meta = json.load(f)
    with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
        doc_id_map = json.load(f)
        
    try:
        with open(PAGERANK_PATH, "r", encoding="utf-8") as f:
            pagerank_map = json.load(f)
        pagerank_min = min(pagerank_map.values()) if pagerank_map else 1.0
    except FileNotFoundError:
        pagerank_map = {}
        pagerank_min = 1.0

    # keep the file handle open for fast O(1) seeks during web requests
    index_file = open(INDEX_PATH, "r", encoding="utf-8")


def _read_postings_for_token(token):
    info = index_meta.get(token)
    if not info:
        return 0, []

    offset, df = info
    try:
        index_file.seek(offset)
        line = index_file.readline()
        if not line:
            return 0, []

        stored_token, stored_df, postings = json.loads(line)
        return stored_df, postings
    except Exception:
        return 0, []


def _augment_query_tokens_with_ngrams(tokens):
    ngrams = []
    for i in range(len(tokens) - 1):
        ngrams.append(tokens[i] + " " + tokens[i + 1])
    for i in range(len(tokens) - 2):
        ngrams.append(tokens[i] + " " + tokens[i + 1] + " " + tokens[i + 2])
    return ngrams


def retrieve_and_rank(query: str):
    start_time = time.time()
    
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return [], 0.0
        
    ngrams = _augment_query_tokens_with_ngrams(query_tokens)
    unigrams_set = set(query_tokens)
    all_scoring_tokens = query_tokens + ngrams

    total_docs = len(doc_id_map)

    # soft conjunction
    token_info = {}  # token is (w_qt, idf, postings)
    query_counts = {}
    for t in all_scoring_tokens:
        query_counts[t] = query_counts.get(t, 0) + 1

    q_norm_sq = 0.0

    for token in set(all_scoring_tokens):
        df, postings = _read_postings_for_token(token)
        if df == 0:
            continue

        idf = math.log10(total_docs / df) if df > 0 else 0.0
        is_ngram = " " in token
        denominator = max(1, len(query_tokens) - token.count(" ")) if is_ngram else len(query_tokens)
        tf_q = query_counts.get(token, 0) / denominator
        w_qt = tf_q * idf

        # emphasize n-grams directly in the query vector
        if is_ngram:
            w_qt *= 2.0

        if w_qt == 0.0:
            continue

        q_norm_sq += w_qt * w_qt
        token_info[token] = (w_qt, idf, postings)

    if q_norm_sq == 0.0:
        return [], 0.0

    # cosine normalization
    doc_scores = {}
    doc_norm_sq = {}
    docs_with_unigram = set()

    for token, (w_qt, idf, postings) in token_info.items():
        for posting in postings:
            doc_id = posting[0]
            tf = posting[1]
            is_imp = posting[2]

            weight_multiplier = 1.5 if is_imp == 1 else 1.0
            w_dt = tf * idf * weight_multiplier

            if w_dt == 0.0:
                continue

            if token in unigrams_set:
                docs_with_unigram.add(doc_id)

            prev_score = doc_scores.get(doc_id, 0.0)
            doc_scores[doc_id] = prev_score + w_dt * w_qt

            prev_norm = doc_norm_sq.get(doc_id, 0.0)
            doc_norm_sq[doc_id] = prev_norm + w_dt * w_dt

    if not docs_with_unigram:
        return [], 0.0

    q_norm = math.sqrt(q_norm_sq)

    # combine with pagerank
    ranked_docs = []
    for doc_id, dot_product in doc_scores.items():
        if doc_id not in docs_with_unigram:
            continue

        d_norm_sq = doc_norm_sq.get(doc_id, 0.0)
        if d_norm_sq == 0.0:
            continue

        d_norm = math.sqrt(d_norm_sq)
        cosine_score = dot_product / (q_norm * d_norm)

        url = doc_id_map[str(doc_id)]
        
        pr_score = pagerank_map.get(url, pagerank_min)
        
        # log of page rank + tf idf
        final_score = cosine_score * math.log10(10 + pr_score) 
        
        ranked_docs.append((url, final_score))

    #top 20 ranked docs
    ranked_docs = sorted(ranked_docs, key=lambda item: item[1], reverse=True)[:20]
    
    elapsed_time = (time.time() - start_time) * 1000 # convert to ms
    return ranked_docs, elapsed_time



HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px auto; max-width: 800px; }
        .search-container { text-align: center; margin-bottom: 30px; }
        input[type="text"] { width: 70%; padding: 10px; font-size: 16px; border-radius: 5px; border: 1px solid #ccc; }
        input[type="submit"] { padding: 10px 20px; font-size: 16px; border-radius: 5px; background: #0066cc; color: white; border: none; cursor: pointer; }
        .result-item { margin-bottom: 20px; }
        .result-url { color: #1a0dab; font-size: 18px; text-decoration: none; }
        .result-url:hover { text-decoration: underline; }
        .result-score { color: #006621; font-size: 14px; }
        .stats { color: #808080; font-size: 14px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="search-container">
        <h1>Assignment 3</h1>
        <form method="GET" action="/">
            <input type="text" name="q" value="{{ query }}" placeholder="Search the ICS domain..." required>
            <input type="submit" value="Search">
        </form>
    </div>

    {% if query %}
        <div class="stats">
            Found {{ results|length }} results in {{ "%.2f"|format(time) }} milliseconds.
        </div>
        
        {% if results %}
            {% for url, score in results %}
            <div class="result-item">
                <a class="result-url" href="{{ url }}" target="_blank">{{ url }}</a><br>
                <span class="result-score">Score: {{ "%.4f"|format(score) }}</span>
            </div>
            {% endfor %}
        {% else %}
            <p>No matching documents found for "<b>{{ query }}</b>".</p>
        {% endif %}
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    query = request.args.get("q", "").strip()
    results = []
    elapsed_time = 0.0

    if query:
        results, elapsed_time = retrieve_and_rank(query)

    return render_template_string(HTML_TEMPLATE, query=query, results=results, time=elapsed_time)


if __name__ == "__main__":
    load_data()
    app.run(host="127.0.0.1", port=5000, debug=False)
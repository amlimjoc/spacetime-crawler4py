import json
import math
import time
from flask import Flask, request, render_template_string
from nltk.stem import PorterStemmer

from inverted_index import tokenize_text

app = Flask(__name__)
stemmer = PorterStemmer()

INDEX_PATH = "inverted_index.json"
INDEX_META_PATH = "inverted_index_meta.json"
ID_MAP_PATH = "doc_id_map.json"

index_meta = {}
doc_id_map = {}
index_file = None


def load_data():
    global index_meta, doc_id_map, index_file
    
    with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
        index_meta = json.load(f)
    with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
        doc_id_map = json.load(f)

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

def _count_ngram_occurrences(ngram_tokens, term_pos_dict):
    if not all(t in term_pos_dict for t in ngram_tokens):
        return 0
        
    occurrences = 0
    first_token = ngram_tokens[0]
    
    for pos in term_pos_dict[first_token]:
        is_match = True
        for i in range(1, len(ngram_tokens)):
            if (pos + i) not in term_pos_dict[ngram_tokens[i]]:
                is_match = False
                break
        
        if is_match:
            occurrences += 1
            
    return occurrences


def retrieve_and_rank(query: str):
    start_time = time.time()
    
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return [], 0.0

    bigrams = []
    for i in range(len(query_tokens) - 1):
        bigrams.append([query_tokens[i], query_tokens[i+1]])
        
    trigrams = []
    for i in range(len(query_tokens) - 2):
        trigrams.append([query_tokens[i], query_tokens[i+1], query_tokens[i+2]])

    total_docs = len(doc_id_map)

    token_info = {} 
    query_counts = {}
    for t in query_tokens:
        query_counts[t] = query_counts.get(t, 0) + 1

    q_norm_sq = 0.0

    for token in set(query_tokens):
        df, postings = _read_postings_for_token(token)
        if df == 0:
            continue

        idf = math.log10(total_docs / df) if df > 0 else 0.0
        tf_q = query_counts[token] / len(query_tokens)
        w_qt = tf_q * idf

        if w_qt == 0.0:
            continue

        q_norm_sq += w_qt * w_qt
        token_info[token] = (w_qt, idf, postings)

    if q_norm_sq == 0.0:
        return [], 0.0

    doc_scores = {}
    doc_term_positions = {}
    docs_with_unigram = set()

    for token, (w_qt, idf, postings) in token_info.items():
        for posting in postings:
            doc_id = posting[0]
            tf = posting[1]
            is_imp = posting[2]
            positions = posting[3] 

            weight_multiplier = 1.5 if is_imp == 1 else 1.0
            w_dt = tf * idf * weight_multiplier

            if w_dt == 0.0:
                continue

            docs_with_unigram.add(doc_id)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (w_dt * w_qt)

            if doc_id not in doc_term_positions:
                doc_term_positions[doc_id] = {}
            doc_term_positions[doc_id][token] = set(positions)

    if not docs_with_unigram:
        return [], 0.0

    q_norm = math.sqrt(q_norm_sq)
    ranked_docs = []
    
    for doc_id, dot_product in doc_scores.items():
        doc_info = doc_id_map.get(str(doc_id))
        if not doc_info:
            continue
            
        url = doc_info[0]
        d_norm = doc_info[1] 

        if d_norm <= 0.0:
            continue

        ngram_boost = 0.0
        term_pos_dict = doc_term_positions[doc_id]
        
        for bg in bigrams:
            occurrences = _count_ngram_occurrences(bg, term_pos_dict)
            ngram_boost += occurrences * 0.5 
            
        for tg in trigrams:
            occurrences = _count_ngram_occurrences(tg, term_pos_dict)
            ngram_boost += occurrences * 1.0 

        # add ngram boost to the cosine similarity 
        final_score = (dot_product / (d_norm * q_norm)) + ngram_boost
        ranked_docs.append((url, final_score))

    ranked_docs = sorted(ranked_docs, key=lambda item: item[1], reverse=True)[:20]
    
    elapsed_time = (time.time() - start_time) * 1000
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
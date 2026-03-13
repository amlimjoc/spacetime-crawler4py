import json
import os
import sys
from urllib.parse import urljoin
import math

from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

INDEX_PATH = "inverted_index.json"
INDEX_META_PATH = "inverted_index_meta.json"
ID_MAP_PATH = "doc_id_map.json"
DEV_FOLDER_PATH = "DEV"
PARTIAL_INDEX_PREFIX = "partial_index_"
PARTIAL_INDEX_THRESHOLD = 10_000
ANCHOR_MAP_PATH = "anchor_tokens.json"

stemmer = PorterStemmer()

def get_tokens_from_soup(soup):
    """
    extracts tokens and identifies important ones.
    """
    all_text = soup.get_text()
    all_tokens = tokenize_text(all_text)

    important_text = []
    for tag in soup.find_all(["title", "h1", "h2", "h3", "b", "strong"]):
        important_text.append(tag.get_text(separator=" "))

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
                if not tok.isdigit() and 1 < len(tok) < 30:
                    tokens.append(stemmer.stem(tok))
                word.clear()
    if word:
        tok = "".join(word)
        if not tok.isdigit() and 1 < len(tok) < 30:
            tokens.append(stemmer.stem(tok))
    return tokens


def _compute_simhash(token_positions):
    """
    64bit simhash
    """
    vector = [0] * 64
    for token, positions in token_positions.items():
        weight = len(positions)
        h = 0
        for ch in token:
            h = (h * 131 + ord(ch)) & ((1 << 64) - 1)

        for bit in range(64):
            if h & (1 << bit):
                vector[bit] += weight
            else:
                vector[bit] -= weight
    fingerprint = 0
    for bit in range(64):
        if vector[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def _calculate_similarity(hash_a, hash_b):
    """
    calculate similarity between two simhashes
    """
    xor = hash_a ^ hash_b
    diff_count = 0
    while xor:
        xor &= xor - 1
        diff_count += 1
    same_bits = 64 - diff_count
    return same_bits / 64.0


def _write_partial_index(index_dict, partial_idx):
    """
    write partial index to file
    """
    if not index_dict:
        return
    filename = f"{PARTIAL_INDEX_PREFIX}{partial_idx}.json"
    with open(filename, "w", encoding="utf-8") as f:
        for token in sorted(index_dict.keys()):
            line_obj = [token, index_dict[token]]
            f.write(json.dumps(line_obj))
            f.write("\n")


def _build_anchor_index(anchor_map, doc_id_map):
    """
    build anchor index from anchor map and doc id map
    """
    if not anchor_map:
        return {}
    # reverse lookup: url to doc_id
    url_to_doc_id = {url: int(doc_id) for doc_id, url in doc_id_map.items()}

    anchor_index = {}

    for target_url, tokens in anchor_map.items():
        doc_id = url_to_doc_id.get(target_url)
        if doc_id is None:
            continue
        if not tokens:
            continue
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        total_tokens = len(tokens)
        if total_tokens == 0:
            continue
        for token, count in counts.items():
            tf_score = count / total_tokens
            is_imp = 1 
            posting = [doc_id, round(tf_score, 5), is_imp, []]
            postings = anchor_index.setdefault(token, [])
            postings.append(posting)
    return anchor_index


def _merge_partial_indexes_to_disk(partial_files, anchor_map, doc_id_map):
    """
    merge of sorted partial index files, writing directly to INDEX_PATH
    and recording per-term offsets and document frequencies.
    """
    anchor_index = _build_anchor_index(anchor_map, doc_id_map)
    if anchor_index:
        anchor_idx = len(partial_files)
        _write_partial_index(anchor_index, anchor_idx)
        partial_files.append(f"{PARTIAL_INDEX_PREFIX}{anchor_idx}.json")

    # open all partials and read first line from each
    active = []
    for path in partial_files:
        try:
            f = open(path, "r", encoding="utf-8")
        except OSError:
            continue
        line = f.readline()
        if not line:
            f.close()
            continue
        token, postings = json.loads(line)
        active.append(
            {
                "file": f,
                "current_token": token,
                "current_postings": postings,
                "path": path,
            }
        )

    if not active:
        return

    term_metadata = {}

    with open(INDEX_PATH, "w", encoding="utf-8") as out_f:
        while active:
            min_token = min(item["current_token"] for item in active)
            merged_postings = []
            next_active = []
            for item in active:
                if item["current_token"] == min_token:
                    merged_postings.extend(item["current_postings"])
                    # advance this file
                    line = item["file"].readline()
                    if line:
                        token, postings = json.loads(line)
                        item["current_token"] = token
                        item["current_postings"] = postings
                        next_active.append(item)
                    else:
                        item["file"].close()
                else:
                    next_active.append(item)
            merged_postings.sort(key=lambda p: p[0])
            # record where this term starts for O(1) seek lookups
            offset = out_f.tell()
            df = len(merged_postings)
            term_metadata[min_token] = [offset, df]

            out_f.write(json.dumps([min_token, df, merged_postings]))
            out_f.write("\n")
            active = next_active

    # clean up partial files
    for item in active:
        try:
            item["file"].close()
        except OSError:
            pass

    for path in partial_files:
        try:
            os.remove(path)
        except OSError:
            pass

    # write term offsets and document frequencies for fast lookup
    with open(INDEX_META_PATH, "w", encoding="utf-8") as meta_f:
        json.dump(term_metadata, meta_f)


def build_inverted_index(document_generator):
    inverted_index = {}
    doc_id_map = {}
    anchor_map = {}
    fingerprints = []

    current_doc_id = 0
    partial_index_counter = 0
    docs_since_flush = 0

    for url, html_content in document_generator:
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        all_tokens, important_tokens = get_tokens_from_soup(soup)

        total_tokens = len(all_tokens)
        if total_tokens == 0:
            continue

        # positions for unigrams
        token_positions = {}
        for idx, tok in enumerate(all_tokens):
            positions = token_positions.setdefault(tok, [])
            positions.append(idx)

        # add 2-grams
        for idx in range(len(all_tokens) - 1):
            bigram = all_tokens[idx] + " " + all_tokens[idx + 1]
            positions = token_positions.setdefault(bigram, [])
            positions.append(idx)

        # add 3-grams
        for idx in range(len(all_tokens) - 2):
            trigram = (
                all_tokens[idx]
                + " "
                + all_tokens[idx + 1]
                + " "
                + all_tokens[idx + 2]
            )
            positions = token_positions.setdefault(trigram, [])
            positions.append(idx)

        simhash = _compute_simhash(token_positions)
        is_duplicate = False
        
        SIM_THRESHOLD = 0.95 

        for prev in fingerprints:
            if _calculate_similarity(simhash, prev) >= SIM_THRESHOLD:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        fingerprints.append(simhash)

        current_doc_id += 1
        docs_since_flush += 1

        if current_doc_id % 500 == 0:
            print(f"{current_doc_id} docs")

        doc_id_map[current_doc_id] = url

        # anchor text collection
        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href")
            if not href:
                continue
            try:
                target_url = urljoin(url, href).split("#")[0]
            except ValueError:
                continue
            anchor_text = anchor.get_text(separator=" ", strip=True)
            if not anchor_text:
                continue
            anchor_tokens = tokenize_text(anchor_text)
            if not anchor_tokens:
                continue
            bucket = anchor_map.setdefault(target_url, [])
            bucket.extend(anchor_tokens)

        # add postings with positions for each token / ngrams
        for token, positions in token_positions.items():
            tf_weighted = 1 + math.log10(len(positions))
            tf_score = tf_weighted / total_tokens
            
            is_imp = 1 if any(word in important_tokens for word in token.split()) else 0
            postings = inverted_index.setdefault(token, [])
            postings.append([current_doc_id, round(tf_score, 5), is_imp, positions])

        # partial index flush
        if docs_since_flush >= PARTIAL_INDEX_THRESHOLD:
            _write_partial_index(inverted_index, partial_index_counter)
            partial_index_counter += 1
            inverted_index.clear()
            docs_since_flush = 0

    if inverted_index:
        _write_partial_index(inverted_index, partial_index_counter)
    partial_files = [
        f
        for f in os.listdir(".")
        if f.startswith(PARTIAL_INDEX_PREFIX) and f.endswith(".json")
    ]

    # merge to final
    _merge_partial_indexes_to_disk(partial_files, anchor_map, doc_id_map)

    # persist anchor map
    try:
        with open(ANCHOR_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(anchor_map, f)
    except OSError as e:
        print(f"Failed to write anchor map: {e}", file=sys.stderr)

    return doc_id_map

def load_documents(dev_path):
    for root, _, files in os.walk(dev_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        yield data.get("url"), data.get("content", "")
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON {file_path}", file=sys.stderr)
                except FileNotFoundError:
                    print(f"File not found {file_path}", file=sys.stderr)
                except UnicodeDecodeError:
                    print(f"Encoding issue in {file_path}", file=sys.stderr)
                except Exception as e:
                    print(f"{file_path}: {e}", file=sys.stderr)

def print_index_stats():    
    id_map = build_inverted_index(load_documents(DEV_FOLDER_PATH))

    with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id_map, f)

    unique_tokens = 0
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            for _ in f:
                unique_tokens += 1
    except FileNotFoundError:
        unique_tokens = 0

    print(f"\nDocuments: {len(id_map)}")
    print(f"Unique Tokens: {unique_tokens}")
    print(f"Index Size: {os.path.getsize(INDEX_PATH) / 1024:.2f} KB")

if __name__ == "__main__":
    print_index_stats()
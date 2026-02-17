import re
from collections import Counter, defaultdict
from urllib.parse import urlparse, urljoin, parse_qs
from nltk.stem import PorterStemmer

from bs4 import BeautifulSoup

# global crawl stats
UNIQUE_URLS = set()
LONGEST_PAGE = {"url": None, "token_count": 0}
WORD_COUNTS = Counter()
SUBDOMAIN_PAGE_COUNTS = defaultdict(int)
CONTENT_SIGNATURES = {}

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else",
    "for", "on", "in", "at", "by", "to", "of", "from", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those",
    "it", "its", "as", "so", "we", "you", "he", "she", "they",
    "them", "his", "her", "their", "our", "us",
    "about", "into", "over", "under", "up", "down", "out", "off",
}

def tokenize_text(text: str):
    tokens = []
    word = []
    stemmer = PorterStemmer()  # Initialize the stemmer

    for ch in text:
        if ch.isalnum() and ch.isascii():
            word.append(ch.lower())
        elif ch == "'" and word:
            continue
        else:
            if word:
                tok = "".join(word)
                # don't add 1-letter tokens
                if len(tok) > 1:
                    # Apply stemming
                    stemmed_token = stemmer.stem(tok)
                    tokens.append(stemmed_token)
                word.clear()
    
    if word:
        tok = "".join(word)
        if len(tok) > 1:
            stemmed_token = stemmer.stem(tok)
            tokens.append(stemmed_token)
    
    return tokens

def scraper(url, resp):
    links = extract_next_links(url, resp)
    return [link for link in links if is_valid(link)]

def extract_next_links(url, resp):
    #Safety so it won't crash on bad responses
    if resp.status != 200 or not resp.raw_response or not resp.raw_response.content:
        return []
    # Implementation required.
    # url: the URL that was used to get the page
    # resp.url: the actual url of the page
    # resp.status: the status code returned by the server. 200 is OK, you got the page. Other numbers mean that there was some kind of problem.
    # resp.error: when status is not 200, you can check the error here, if needed.
    # resp.raw_response: this is where the page actually is. More specifically, the raw_response has two parts:
    #         resp.raw_response.url: the url, again
    #         resp.raw_response.content: the content of the page!
    # Return a list with the hyperlinks (as strings) scrapped from resp.raw_response.content
    if resp is None or not hasattr(resp, "status"):
        return []
    if resp.status != 200:
        return []
    if not hasattr(resp, "raw_response") or resp.raw_response is None:
        return []
    if not hasattr(resp.raw_response, "content") or not resp.raw_response.content:
        return []

    content = resp.raw_response.content

    # html filter trap
    if len(content) > 2_000_000:
        return []

    soup = BeautifulSoup(content, 'html.parser')
    text = soup.get_text(separator=" ")
    tokens = tokenize_text(text)
    token_count = len(tokens)
    page_url = resp.url.split('#')[0] if hasattr(resp, "url") else url.split('#')[0]
    parsed_page = urlparse(page_url)
    hostname = parsed_page.netloc.lower()

    # low information page trap
    if token_count < 20:
        return []

    if page_url not in UNIQUE_URLS:
        UNIQUE_URLS.add(page_url)

        if token_count > LONGEST_PAGE["token_count"]:
            LONGEST_PAGE["token_count"] = token_count
            LONGEST_PAGE["url"] = page_url

        # common words tracking
        for tok in tokens:
            if tok not in STOP_WORDS:
                WORD_COUNTS[tok] += 1

        if hostname.endswith(".uci.edu") or hostname == "uci.edu":
            SUBDOMAIN_PAGE_COUNTS[hostname] += 1

    # content trap uses first 50 tokens to identify duplicates
    token_prefix = tuple(tokens[:50])
    signature_key = (hostname, parsed_page.path, token_prefix)
    first_seen_url = CONTENT_SIGNATURES.get(signature_key)
    if first_seen_url is not None and first_seen_url != page_url:
        return []
    CONTENT_SIGNATURES[signature_key] = page_url

    links = []

    for anchor in soup.find_all('a', href=True):
        link = anchor['href']
        if link.startswith(('mailto:', 'javascript:', 'tel:')):
            continue
        
        try:
            absolute_url = urljoin(page_url, link).split('#')[0]
        except ValueError:
            continue
        # not adding duplicates and link to self
        if absolute_url not in links and absolute_url != page_url:
            links.append(absolute_url)

    return links


def is_valid(url):
    # Decide whether to crawl this url or not. 
    # If you decide to crawl it, return True; otherwise return False.
    # There are already some conditions that return False.
    try:
        parsed = urlparse(url)
        if parsed.scheme not in set(["http", "https"]):
            return False
        
        # avoid calendar links/trap using keywords in path or query
        trap_keywords = ['calendar', 'date', 'year', 'month', 'day', 'time', 'ical', 'outlook-ical', 'tribe-bar-date']
        combined = (parsed.path + " " + parsed.query).lower()
        if any(keyword in combined for keyword in trap_keywords):
            return False

        # super trap list
        query_lower = parsed.query.lower()

        query_traps = [
            "view=",
            "expanded=",
            "filter",
            "affiliation",
            "do=",
            "idx=",
            "ns=",
            "tab_",
            "rev=",
            "action=",
            'sort', 
            'order', 
            'orderby', 
            'search', 
            'filter', 
            'limit', 
            'page', 
            'p', 
            'skip', 
            'take',
        ]

        if any(trap in query_lower for trap in query_traps):
            return False

        # path repetition / numeric traps
        segments = [seg for seg in parsed.path.split("/") if seg]
        
        if len(segments) > 8:
            return False

        # too many purely numeric segments (often calendars or IDs)
        numeric_segments = [seg for seg in segments if seg.isdigit()]
        if len(numeric_segments) >= 3:
            return False

        # path repetition detection
        if segments:
            # adjacent duplicate segments
            for i in range(len(segments) - 1):
                if segments[i] == segments[i + 1]:
                    return False

            # same segment repeated many times
            if any(segments.count(seg) >= 3 for seg in set(segments)):
                return False

            # /a/b/a/b
            if len(segments) >= 4 and len(segments) % 2 == 0:
                half = len(segments) // 2
                if segments[:half] == segments[half:]:
                    return False

        # session id trap
        query_params = parse_qs(parsed.query)
        host = (parsed.hostname or "").lower()
        path_lower = parsed.path.lower()

        # Filter out eppstein/pix directory
        if '/eppstein/pix/' in path_lower or path_lower.startswith('/~eppstein/pix/'):
            return False
            
        qkeys = {k.lower() for k in query_params.keys()}

        #wiki: rej query variants
        if host in {"wiki.ics.uci.edu", "swiki.ics.uci.edu"} and parsed.query:
            return False 
        
        #gitlab: rej large browsing areas
        git_traps = ['commit', 'tree', 'blob', 'diff', 'blame', 'compare']
        if any(trap in parsed.path.lower() for trap in git_traps):
            return False
        #autoindex sort trap
        if "c" in qkeys and "o" in qkeys:
            return False 

        #trap from version/format parameter
        if "version" in qkeys:
            return False

        if "format" in qkeys:
            return False

        host = (parsed.hostname or "").lower()
        
        session_like_keys = {
            "sessionid", "sid", "phpsessid", "jsessionid",
            "asp-session-id", "aspsessionid",
        }
        for key in query_params.keys():
            key_lower = key.lower()
            if key_lower in session_like_keys or "session" in key_lower:
                return False

        # dokuwiki trap
        if "doku.php" in path_lower:
            return False
        
        # redirector/social sharing traps
        if "r.php?next=" in url.lower() or "facebook.com" in url.lower():
            return False

        #diff mode
        if "action=diff" in query_lower:
            return False
        
        #attachments trap
        if any(seg in path_lower for seg in ["raw-attachment", "attachment", "zip-attachment"]):
            return False

        # date-pattern regex blocking (calendar/date traps)

        date_regexes = [
            r'\d{4}-\d{2}-\d{2}',   # YYYY-MM-DD
            r'\d{4}/\d{2}/\d{2}',   # YYYY/MM/DD
            r'\d{4}\.\d{2}\.\d{2}', # YYYY.MM.DD
            r'\d{2}-\d{2}-\d{4}',   # MM-DD-YYYY
            r'\d{2}/\d{2}/\d{4}',   # MM/DD/YYYY
            r'\d{4}-\d{2}',         # YYYY-MM
            r'\d{4}/\d{2}',         # YYYY/MM
            r'\d{2}-\d{4}',         # MM-YYYY
            r'\d{2}/\d{4}',         # MM/YYYY
            r'\d{4}\.\d{2}',        # YYYY.MM
        ]

        for pattern in date_regexes:
            if re.search(pattern, path_lower):
                return False

        # very long query strings / too many parameters
        if len(parsed.query) > 100 or parsed.query.count("&") > 2:
            return False
        
        allowed_domains = ['ics.uci.edu', 'cs.uci.edu', 'informatics.uci.edu', 'stat.uci.edu']

        #old was:
        # domain = parsed.netloc
        # if not any(domain.endswith(allowed_domain) for allowed_domain in allowed_domains):
        #     return False
        # netloc can have ports and the endwith filtering also allows bad lookalikes to pass through. new one is stricterr

        domain = (parsed.hostname or "").lower()
        if not any(domain == d or domain.endswith("." + d) for d in allowed_domains):
            return False
        
        return not re.match(
            r".*\.(css|js|bmp|gif|jpe?g|ico"
            + r"|png|tiff?|mid|mp2|mp3|mp4"
            + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
            + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
            + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
            + r"|epub|dll|cnf|tgz|sha1"
            + r"|thmx|mso|arff|rtf|jar|csv"
            + r"|rm|smil|wmv|swf|wma|zip|rar|gz)$", parsed.path.lower())

    except TypeError:
        print ("TypeError for ", parsed)
        raise


def get_crawl_stats():
    """
    answers to questions
    """
    top_words = WORD_COUNTS.most_common(50)
    subdomains_sorted = dict(sorted(SUBDOMAIN_PAGE_COUNTS.items()))
    return {
        "unique_page_count": len(UNIQUE_URLS),
        "longest_page": LONGEST_PAGE,
        "top_words": top_words,
        "subdomains": subdomains_sorted,
    }


def print_crawl_stats():
    """Print crawl stats"""
    stats = get_crawl_stats()
    print(f"Unique pages: {stats['unique_page_count']}")
    lp = stats['longest_page']
    print(f"Longest page: {lp['token_count']} tokens -> {lp['url']}")
    print("\nTop 50 words:")
    for word, count in stats['top_words']:
        print(f"  {word}: {count}")
    print("\nSubdomains (pages):")
    for host, count in stats['subdomains'].items():
        print(f"  {host}: {count}")

if __name__ == "__main__":
    print_crawl_stats()

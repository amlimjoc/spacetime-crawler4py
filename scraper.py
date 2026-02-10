import re
from urllib.parse import urlparse
UNIQUE_URLS = set()
LONGEST_PAGE = {"url": None, "token_count": 0}

from bs4 import BeautifulSoup
from urllib.parse import urljoin

def tokenize_text(text: str):
    tokens = []
    word = []

    for ch in text:
        if ch.isalnum() and ch.isascii():
            word.append(ch.lower())
            #ignore apostrophes inside words
        elif ch == "'" and word:
            continue
        else:
            if word:
                tok = "".join(word)
                #don't add 1-letter tokens
                if len(tok) > 1:
                    tokens.append(tok)
                word.clear()
    if word:
        tok = "".join(word)
        if len(tok) > 1:
            tokens.append(tok)
    
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
    soup = BeautifulSoup(resp.raw_response.content, 'html.parser')
    text = soup.get_text(separator=" ")
    tokens = tokenize_text(text)
    token_count = len(tokens)
    page_url = resp.url.split('#')[0] if hasattr(resp, "url") else url.split('#')[0]

    if page_url not in UNIQUE_URLS:
        UNIQUE_URLS.add(page_url)

        if token_count > LONGEST_PAGE["token_count"]:
            LONGEST_PAGE["token_count"] = token_count
            LONGEST_PAGE["url"] = page_url
    links = []

    for anchor in soup.find_all('a', href=True):
        link = anchor['href']
        if link.startswith(('mailto:', 'javascript:', 'tel:')):
            continue

        absolute_url = urljoin(page_url, link).split('#')[0]
        # not adding duplicates and link to self
        if absolute_url not in links and absolute_url != page_url:
            # remove fragment
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
        
        # avoid calendar links/trap using keywords in queries
        trap_keywords = ['calendar', 'date', 'year', 'month', 'day', 'time', 'ical', 'outlook-ical', 'tribe-bar-date']
        if any(keyword in parsed.path.lower() for keyword in trap_keywords):
            return False
        # avoid traps with too many parameters
        if parsed.query.count("&") > 2:
            return False
        allowed_domains = ['ics.uci.edu', 'cs.uci.edu', 'informatics.uci.edu', 'stat.uci.edu']
        domain = parsed.netloc
        if not any(domain.endswith(allowed_domain) for allowed_domain in allowed_domains):
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

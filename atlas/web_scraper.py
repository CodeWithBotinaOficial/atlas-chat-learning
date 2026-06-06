import requests
import urllib.robotparser
from urllib.parse import urlparse
import sys
import os

from . import text_processor

def check_robots_txt(url: str) -> bool:
    """
    Checks if scraping the given URL is allowed by its robots.txt file.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if scraping is allowed, False otherwise.
    """
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    robots_url = f"{base_url}/robots.txt"
    
    rp = urllib.robotparser.RobotFileParser()
    
    try:
        # Fetch robots.txt content using requests for better error handling
        # Use a generic User-Agent for fetching robots.txt itself
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}
        robots_response = requests.get(robots_url, headers=headers, timeout=5)
        robots_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        # Parse the content directly
        rp.parse(robots_response.text.splitlines())
        print(f"[✓] Successfully fetched and parsed robots.txt from {robots_url}.")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"[!] robots.txt not found at {robots_url}. Assuming scraping is allowed.")
            return True # If robots.txt is not found, assume allowed
        else:
            print(f"[✗] HTTP error fetching robots.txt from {robots_url}: {e}", file=sys.stderr)
            print("Assuming scraping is disallowed for safety.", file=sys.stderr)
            return False
    except requests.exceptions.RequestException as e:
        print(f"[✗] Network error fetching robots.txt from {robots_url}: {e}", file=sys.stderr)
        print("Assuming scraping is disallowed for safety.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[✗] Error parsing robots.txt from {robots_url}: {e}", file=sys.stderr)
        print("Assuming scraping is disallowed for safety.", file=sys.stderr)
        return False

    # Check for specific user-agent 'AtlasBot' first, then general '*'
    user_agents_to_check = ['AtlasBot', '*'] 
    
    for ua in user_agents_to_check:
        if not rp.can_fetch(ua, url):
            print(f"[✗] robots.txt check failed for User-agent '{ua}'. Scraping of {url} is disallowed.")
            # urllib.robotparser.RobotFileParser does not expose the exact rule that caused the disallowance
            # easily. For more detailed output, a custom parser would be needed.
            return False
    
    print(f"[✓] robots.txt check passed. Scraping of {url} is allowed.")
    return True

def scrape_article_text(url: str) -> str:
    """
    Fetches the HTML content of a URL and extracts the main article text.

    Args:
        url (str): The URL to scrape.

    Returns:
        str: The cleaned and sanitized main text content of the page.

    Raises:
        requests.exceptions.RequestException: If there's a network error.
        Exception: For other scraping or processing errors.
    """
    try:
        headers = {
            'User-Agent': 'AtlasBot/1.0 (+https://github.com/CodeWithBotinaOficial/atlas-chat-learning)' # Updated User-Agent
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        raw_html = response.text
        
        cleaned_text = text_processor.clean_text(raw_html)
        return cleaned_text
    except requests.exceptions.RequestException as e:
        print(f"Network error while fetching {url}: {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error during scraping or text processing for {url}: {e}", file=sys.stderr)
        raise

if __name__ == '__main__':
    # Example usage
    # Test with a URL that should be allowed (e.g., a blog post)
    test_allowed_url = "https://blog.codewithbotina.com/es/posts/recreacion-moderna-de-pac-man-aprende-a-desarrollar-juegos-cross-platform-con-net-y-avalonia-ui"
    print(f"\n--- Testing robots.txt for ALLOWED URL: {test_allowed_url} ---")
    if check_robots_txt(test_allowed_url):
        print("Scraping allowed. Attempting to scrape...")
        try:
            text = scrape_article_text(test_allowed_url)
            print("\n--- Scraped Text Sample (first 500 chars) ---")
            print(text[:500])
            print("...")
        except Exception as e:
            print(f"Failed to scrape: {e}")
    else:
        print("Scraping disallowed.")

    # Test with a URL that might be disallowed (e.g., a common admin path)
    # Note: This might not actually be disallowed on example.com, it's just for demonstration
    test_disallowed_url = "https://www.example.com/admin/" 
    print(f"\n--- Testing robots.txt for POTENTIALLY DISALLOWED URL: {test_disallowed_url} ---")
    if check_robots_txt(test_disallowed_url):
        print("Scraping allowed.")
    else:
        print("Scraping disallowed (as expected for this test case).")

    # Test with a URL where robots.txt might be missing (e.g., a very simple site)
    # Using a URL that is likely to return a 404 for robots.txt
    test_no_robots_url = "http://www.nonexistentdomain12345.com/some-page" 
    print(f"\n--- Testing robots.txt for URL with potentially MISSING robots.txt: {test_no_robots_url} ---")
    if check_robots_txt(test_no_robots_url):
        print("Scraping allowed (due to missing robots.txt or other reason).")
    else:
        print("Scraping disallowed.")
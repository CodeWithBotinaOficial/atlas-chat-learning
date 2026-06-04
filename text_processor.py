import re
import sys

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Error: 'beautifulsoup4' library not found.", file=sys.stderr)
    print("Please install it using: pip install beautifulsoup4", file=sys.stderr)
    sys.exit(1)

def clean_text(raw_html: str) -> str:
    """
    Parses raw HTML, extracts the main textual content, and sanitizes it.

    This function attempts to find the main content of a webpage by looking
    for common HTML tags and attributes associated with articles or main content.
    It removes script and style tags, normalizes whitespace, and strips
    leading/trailing whitespace.

    Args:
        raw_html (str): The raw HTML content of a webpage.

    Returns:
        str: The cleaned and sanitized main text content.
    """
    if not raw_html:
        return ""

    soup = BeautifulSoup(raw_html, 'lxml') # Using 'lxml' parser for better performance and robustness

    # Remove script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    # Try to find the main content area
    main_content_tags = ['article', 'main', 'div', 'section']
    possible_main_content = None

    # Prioritize tags like <article> or <main>
    for tag_name in ['article', 'main']:
        tag = soup.find(tag_name)
        if tag:
            possible_main_content = tag
            break
    
    # If not found, look for common class/id names in div/section
    if not possible_main_content:
        for tag_name in ['div', 'section']:
            for attr_name in ['class', 'id']:
                for keyword in ['content', 'post-body', 'article-body', 'main-content', 'entry-content']:
                    tag = soup.find(tag_name, {attr_name: re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)})
                    if tag:
                        possible_main_content = tag
                        break
                if possible_main_content:
                    break
            if possible_main_content:
                break

    # Fallback: if no specific main content found, extract all text from paragraph tags
    if not possible_main_content:
        paragraphs = soup.find_all('p')
        text_parts = [p.get_text(separator=' ', strip=True) for p in paragraphs]
        text = '\n\n'.join(text_parts)
    else:
        text = possible_main_content.get_text(separator='\n', strip=True)

    # Normalize whitespace: replace multiple spaces/newlines with a single space/newline
    # and then strip leading/trailing whitespace from each line
    cleaned_lines = []
    for line in text.split('\n'):
        line = re.sub(r'\s+', ' ', line).strip()
        if line: # Only add non-empty lines
            cleaned_lines.append(line)
    
    # Join lines with double newline for paragraph separation, then normalize again
    final_text = '\n\n'.join(cleaned_lines)
    final_text = re.sub(r'\n\s*\n', '\n\n', final_text).strip() # Remove excessive blank lines

    return final_text

if __name__ == '__main__':
    # Example usage with a dummy HTML string
    dummy_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
        <script>var x = 1;</script>
        <style>.a { color: red; }</style>
    </head>
    <body>
        <header>
            <h1>Website Title</h1>
            <nav>
                <a href="#">Home</a>
                <a href="#">About</a>
            </nav>
        </header>
        <main id="main-content">
            <article>
                <h2>Article Title</h2>
                <p>This is the first paragraph of the article. It contains some important information.</p>
                <p>Here is the second paragraph. It also has some content.   With  extra spaces.</p>
                <div>
                    <p>A paragraph inside a div within the article.</p>
                </div>
                <ul>
                    <li>List item 1</li>
                    <li>List item 2</li>
                </ul>
            </article>
            <aside>
                <h3>Related Posts</h3>
                <p>Ad content here.</p>
            </aside>
        </main>
        <footer>
            <p>&copy; 2023 My Website</p>
        </footer>
    </body>
    </html>
    """
    
    print("--- Original HTML (truncated) ---")
    print(dummy_html[:500])
    print("...")

    cleaned = clean_text(dummy_html)
    print("\n--- Cleaned Text ---")
    print(cleaned)

    # Test with a more complex structure (simulating a blog post)
    blog_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>My Awesome Blog Post</title>
        <style>body { font-family: sans-serif; }</style>
        <script>console.log('hello');</script>
    </head>
    <body>
        <nav class="navbar">
            <a href="/">Home</a>
            <a href="/blog">Blog</a>
        </nav>
        <div id="wrapper">
            <header>
                <h1>Welcome to My Blog</h1>
            </header>
            <main class="article-body">
                <article>
                    <h2>The Wonders of Python</h2>
                    <p>Python is a versatile language. It's used everywhere from web development to data science. Its simplicity makes it a great choice for beginners.</p>
                    <p>One of Python's strengths is its extensive ecosystem of libraries. Libraries like NumPy, Pandas, and Scikit-learn are indispensable for data analysis.</p>
                    <h3>Getting Started</h3>
                    <p>To start with Python, you typically install it from python.org. Then, you can use pip to install packages.</p>
                    <pre><code>pip install requests beautifulsoup4</code></pre>
                    <p>This is a final thought on the topic.</p>
                </article>
                <aside class="sidebar">
                    <h4>About the Author</h4>
                    <p>John Doe is a Python enthusiast.</p>
                    <div class="ad-container">
                        <p>Buy our premium course!</p>
                    </div>
                </aside>
            </main>
            <footer>
                <p>Copyright &copy; 2023</p>
            </footer>
        </div>
    </body>
    </html>
    """
    print("\n--- Blog HTML (truncated) ---")
    print(blog_html[:500])
    print("...")
    blog_cleaned = clean_text(blog_html)
    print("\n--- Cleaned Blog Text ---")
    print(blog_cleaned)

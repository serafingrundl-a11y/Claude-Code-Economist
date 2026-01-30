"""
Local preview server for the static DACA Results website.
Run: python serve.py
Then open http://localhost:8000 in your browser.
"""

import http.server
from pathlib import Path

PORT = 8000

if __name__ == "__main__":
    directory = str(Path(__file__).resolve().parent)
    print(f"Serving at http://localhost:{PORT}")
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=directory, **kwargs)
    server = http.server.HTTPServer(("", PORT), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()

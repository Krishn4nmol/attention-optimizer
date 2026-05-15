import http.server
import socketserver
import threading
import os

def start_video_server(video_path, port=8502):
    directory = os.path.dirname(os.path.abspath(video_path))

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, format, *args):
            pass  # suppress logs

    try:
        server = socketserver.TCPServer(("", port), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        return server
    except OSError:
        pass  # server already running on that port
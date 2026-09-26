import http.server
import socketserver
import os
import glob

TASKS_DIR = r"C:\Users\Suman Patari\.gemini\antigravity-ide\brain\599e623b-e4ad-4b2b-a894-aa35a9138945\.system_generated\tasks"
PORT = 8080

def get_latest_log_file():
    list_of_files = glob.glob(os.path.join(TASKS_DIR, '*.log'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

class LogHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        log_file = get_latest_log_file()
        
        try:
            if log_file:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[-200:]
                    log_content = "".join(lines)
            else:
                log_content = "No log files found yet..."
        except Exception as e:
            log_content = f"Error reading log: {e}"

        html = f"""
        <html>
        <head>
            <title>Live Training Log</title>
            <meta http-equiv="refresh" content="5">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ background-color: #0d1117; color: #00ff00; font-family: monospace; padding: 20px; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; font-size: 14px; line-height: 1.5; }}
            </style>
            <script>
                window.onload = function() {{
                    window.scrollTo(0, document.body.scrollHeight);
                }}
            </script>
        </head>
        <body>
            <h2>Neural Network Training Log - Live Updates</h2>
            <p>Auto-refreshes every 5 seconds... (Reading newest log: {os.path.basename(str(log_file))})</p>
            <hr style="border-color:#333;">
            <pre>{log_content}</pre>
        </body>
        </html>
        """
        self.wfile.write(html.encode('utf-8'))

with socketserver.TCPServer(("", PORT), LogHandler) as httpd:
    print(f"Serving log at port {PORT}")
    httpd.serve_forever()

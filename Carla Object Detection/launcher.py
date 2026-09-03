import os
import re
import subprocess
import sys
import threading
import time

# Define the port matching your perception server's configuration
PORT = int(os.getenv("PORT", "8000"))

def run_server():
    """Runs the FastAPI server using a subprocess."""
    print(f"[*] Starting Perception Server on port {PORT}...")
    # This executes your script which internally calls uvicorn.run
    subprocess.run([sys.executable, "perception_server.py"])

def run_cloudflared():
    """Runs the Cloudflare tunnel and parses stderr for the public URL."""
    print("[*] Starting Cloudflare Quick Tunnel...")
    
    # Launch cloudflared tunnel pointing to our local server
    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Regex to capture the dynamically generated TryCloudflare URL
    url_regex = re.compile(r"https://[-a-zA-Z0-9]+\.trycloudflare\.com")
    url_found = False

    # Cloudflared streams logs to stderr
    while True:
        line = process.stderr.readline()
        if not line and process.poll() is not None:
            break
        
        if line:
            match = url_regex.search(line)
            if match and not url_found:
                print("\n" + "="*65)
                print(f"🚀 PUBLIC URL READY: {match.group(0)}")
                print("="*65 + "\n")
                url_found = True

if __name__ == "__main__":
    # Ensure cloudflared is installed and accessible in the system PATH
    try:
        subprocess.run(["cloudflared", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: 'cloudflared' is not installed or not in PATH.")
        print("Please install it from: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
        sys.exit(1)

    # Start the local FastAPI server in a background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait a few seconds to let the server bind to the port
    time.sleep(3)

    # Start the Cloudflare tunnel
    try:
        run_cloudflared()
    except KeyboardInterrupt:
        print("\n[*] Shutting down server and tunnel...")
        sys.exit(0)
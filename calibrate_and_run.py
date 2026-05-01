import os
import sys
import subprocess
import urllib.request
import urllib.error
import time
from pathlib import Path

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[94m",  # Blue
        "SUCCESS": "\033[92m", # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
        "RESET": "\033[0m"
    }
    print(f"{colors.get(status, '')}[{status}] {message}{colors['RESET']}")

def check_python_version():
    print_status("Checking Python version...", "INFO")
    if sys.version_info < (3, 8):
        print_status("Python 3.8+ is required.", "ERROR")
        sys.exit(1)
    print_status(f"Python {sys.version.split()[0]} detected.", "SUCCESS")

def install_dependencies():
    print_status("Checking and installing dependencies...", "INFO")
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print_status("requirements.txt not found!", "ERROR")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
        print_status("Dependencies installed.", "SUCCESS")
    except subprocess.CalledProcessError:
        print_status("Failed to install dependencies.", "ERROR")
        return False
    return True

def check_env_file():
    print_status("Checking .env file...", "INFO")
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print_status(".env file not found!", "WARNING")
        # Check for template
        template_path = Path(__file__).parent / ".env.example" 
        # (Assuming we might want to create one, but for now just warn)
        print_status("Please ensure you have a .env file with your GEMINI_API_KEY.", "WARNING")
        return False
    print_status(".env file found.", "SUCCESS")
    return True

def check_ollama():
    print_status("Checking Ollama connection...", "INFO")
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/", timeout=2) as response:
            if response.status == 200:
                print_status("Ollama is running.", "SUCCESS")
                return True
    except (urllib.error.URLError, ConnectionRefusedError):
        pass
    
    print_status("Ollama is NOT running.", "WARNING")
    print_status("You may need to run 'ollama serve' in a separate terminal for LLM features.", "WARNING")
    return False

def main():
    print_status("Starting System Calibration...", "INFO")
    
    check_python_version()
    
    if not install_dependencies():
        sys.exit(1)
        
    check_env_file()
    check_ollama()
    
    print_status("Calibration Complete! Launching App...", "SUCCESS")
    print("-" * 50)
    
    # Run the main app
    app_path = Path(__file__).parent / "app.py"
    try:
        subprocess.run([sys.executable, str(app_path)])
    except KeyboardInterrupt:
        print_status("\nApplication stopped by user.", "INFO")

if __name__ == "__main__":
    main()

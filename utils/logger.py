import os
import sys
import datetime

class SimpleLogger:
    def __init__(self, log_file=None):
        self.log_file = log_file
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def _log(self, level, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        
        # Print to console
        print(formatted)
        sys.stdout.flush()
        
        # Save to file if enabled
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(formatted + "\n")

    def info(self, message):
        self._log("INFO", message)

    def warning(self, message):
        self._log("WARNING", message)

    def error(self, message):
        self._log("ERROR", message)

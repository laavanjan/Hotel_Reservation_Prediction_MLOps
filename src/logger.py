import logging
import os
from datetime import datetime

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")
##filename = log_24-02-2025.log

logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
##what should be wrtten inside the LOG_FILE

def get_logger(name):
    """
    Creates and configures a logger with a specific name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

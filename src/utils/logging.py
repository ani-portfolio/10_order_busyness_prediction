import sys
from pathlib import Path
import logging
from datetime import datetime
from src import config

script_name = Path(sys.argv[0]).stem 
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

handlers = [logging.StreamHandler()]  

if config.logger_path:  
    log_file = f'{config.logger_path}{script_name}_{timestamp}.log'
    handlers.append(logging.FileHandler(log_file))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
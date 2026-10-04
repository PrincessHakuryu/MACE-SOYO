import logging
import os
import sys



lab_info = "MACE-SOYO training"



class GPUFilter(logging.Filter):
    """Allow log records from distributed rank 0 only."""
    def __init__(self, dist_id,local_id):
        super().__init__()
        self.dist_id = dist_id
        self.local_id = local_id

    def filter(self, record):
        return self.dist_id == 0

class GPUHandler(logging.FileHandler):
    """File handler for per-rank training logs."""
    def __init__(self, dist_id,local_id, log_dir='./logs'):
        # Create a separate log file for each distributed rank.
        filename = os.path.join(log_dir, f"gpu_{dist_id}.debug")
        super().__init__(filename=filename,mode="w")  # Initialize FileHandler.

        # Set the log format.
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        self.setFormatter(formatter)

class GpuLogger(logging.Logger):
    """Logger subclass for per-rank training logs."""
    def __init__(self, dist_id=0, local_id=0, log_level=logging.DEBUG, log_dir='./logs/debug'):
        super().__init__(name=f"gpu_{dist_id}", level=log_level)
        self.dist_id = dist_id
        self.local_id = local_id

        # Add handlers and filters only during initial configuration.
        if not self.hasHandlers():
            os.makedirs(log_dir, exist_ok=True)

            # Create the console handler.
            console_handler = logging.StreamHandler(sys.stdout)  # Console output.
            console_handler.setLevel(logging.INFO)  # Log INFO and higher levels.
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)

            # Create the file handler.
            debug_handler = GPUHandler(dist_id,local_id, log_dir)  # File output.
            debug_handler.setLevel(logging.DEBUG)  # Log DEBUG and higher levels.

            self.addHandler(console_handler)
            self.addHandler(debug_handler)

            # Add the rank filter.
            gpu_filter = GPUFilter(dist_id,local_id)
            console_handler.addFilter(gpu_filter)  # Filter console output.
            #debug_handler.addFilter(gpu_filter)  # Optionally filter file output too.
        self.info(lab_info)

if __name__ == "__main__":
    # Logging smoke test.
    gpu_id = 0  # Set to 1 to test nonzero-rank filtering.
    logger = GpuLogger(gpu_id)

    logger.info("This is an info message1.")
    logger.debug("This is a debug message2.")

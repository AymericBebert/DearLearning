"""
Default configuration file. Put here all the variables you will need for the code.

If you want to locally override some of them, make a `local.py` file next to this one,
and change their values in it.
"""

# Log config
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-8s %(threadName)s/%(module)s: %(message)s"
LOG_DATE_FMT = "%H:%M:%S"

# Path
DATA_PATH = "Data"
DATASET_PATH = "Data/nsynth-train"
CACHE_PATH = "Cache"

# Spectrogram params
BLOCK_DURATION = 100  # block size in milliseconds
STEP_FRACTION = 8     # window move 1/this_number right each step
FFT_SIZE = 256        # width of FFT window
FFT_RATIO = 8         # 1/<this> of the total FFT will be kept (2 for full real FFT)
SAMPLE_DURATION = 2   # <this> second sound (padding or crop)

# Dataset filtering
DATASET_EXCLUDE_FAMILIES = []
DATASET_EXCLUDE_SOURCES = ["synthetic", "electronic"]

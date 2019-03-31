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

# Spectrogram params
WINDOW_SECOND_FRACTION = 10  # 0.1s window
STEP_FRACTION = 8  # window move 1/this_number right each step
FFT_SIZE = 256  # Width of FFT window
SAMPLE_DURATION = 2  # 2s sound (padding or crop)

#!/usr/bin/env python3

"""
Show a text-mode spectrogram using live microphone data.
"""

import logging
from typing import Dict, List, Callable, TypeVar, Generic

import numpy as np
from scipy.signal import windows as ssw
from scipy.fftpack import fft
from sklearn.metrics import accuracy_score, confusion_matrix
import sounddevice as sd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical, plot_model

import config
from src.utils import resolve_path
from src.sound_actions import wave_file_to_time_data


# local_data_store = {0: [resolve_path("Data", "french.wav")], 1: [resolve_path("Data", "english.wav")]}
# local_classes_i2n = {0: "french", 1: "english"}

local_data_store = {
    0: [resolve_path("Data", "grave.wav")],
    1: [resolve_path("Data", "normal.wav")],
    2: [resolve_path("Data", "haute.wav")],
}
local_classes_i2n = {
    0: "grave",
    1: "normal",
    2: "haute",
}

# local_data_store = {
#     0: [resolve_path("Data", "alestorm.wav")],
#     1: [resolve_path("Data", "amon_amarth.wav")],
#     2: [resolve_path("Data", "nightwish.wav")],
#     # 3: [resolve_path("Data", "normal.wav")],
# }
# local_classes_i2n = {
#     0: "alestorm",
#     1: "amon_amarth",
#     2: "nightwish",
#     # 3: "voix",
# }


def give_many_fft(fs: int, time_data: np.ndarray, threshold: float = 0.01,
                  sample_duration: int = config.SAMPLE_DURATION,
                  block_duration: int = config.BLOCK_DURATION,
                  step_fraction: int = config.STEP_FRACTION,
                  fft_size: int = config.FFT_SIZE,
                  fft_ratio: int = config.FFT_RATIO,
                  ):
    """From frequency sampling and content of a wav file, make the spectrogram"""
    sample_size = fs * sample_duration

    window_size = int(fs * block_duration / 1000)
    window = ssw.blackman(window_size)

    step = window_size // step_fraction
    n_steps = sample_size // step

    padding_left = np.zeros(step)
    padding_right = np.zeros(max(0, sample_size - len(time_data) - step))
    time_data_2 = np.concatenate((padding_left, time_data, padding_right))[:sample_size]

    for k in range(n_steps - step_fraction):
        sound_part = time_data_2[k * step:k * step + window_size]
        sound_windowed = window * sound_part
        if max(abs(sound_windowed)) < threshold:
            continue
        yield abs(fft(sound_windowed, fft_size * fft_ratio)[:fft_size])


def make_labels_and_fft(corr: Dict[int, List[str]], file_access: Callable[[str], str] = lambda x: x, **kwargs):
    """Make the fft of some samples from each file in the data dict
    :returns: Lists for ids, labels, spectrograms; lookups {class_index -> class_name}, {class_name -> class_index}
    """
    labels: List[int] = []
    data: List[np.ndarray] = []

    for k, file_names in corr.items():
        logging.info(f"Making {k} data...")
        for fn in file_names:
            file_full_name = file_access(fn)
            for out in give_many_fft(*wave_file_to_time_data(file_full_name), **kwargs):
                labels.append(k)
                data.append(out)
        logging.info(f"- Total: {len(labels)} entries...")

    return labels, data


def make_model(evaluate=False, **kwargs):
    logging.info("Making labels and ffts...")
    labels, ffts = make_labels_and_fft(local_data_store, **kwargs)

    # Shuffle dataset
    logging.info("Shuffling...")
    indices = np.arange(len(labels))
    np.random.shuffle(indices)

    labels = np.asarray(labels)[indices]
    ffts = np.asarray(ffts)[indices]

    labels = to_categorical(np.asarray(labels))
    num_classes = labels.shape[1]

    logging.info(f"{labels.shape[0]} entries, {labels.shape[1]} classes, ffts shape {ffts.shape}")

    # Model
    logging.info("Making model...")
    model = Sequential()

    model.add(Dense(256, input_shape=(256,), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    plot_model(model, to_file=resolve_path("Data", "model_3.png"), show_shapes=True, show_layer_names=True)

    if evaluate:
        logging.info("Model evaluation...")

        # Cut dataset into train and validation sets
        cut = int(labels.shape[0] * 0.8)

        labels_train, labels_valid = labels[:cut], labels[cut:]
        ffts_train, ffts_valid = ffts[:cut], ffts[cut:]

        # Train the model on train data
        model.fit(ffts_train, labels_train, batch_size=64, epochs=20, validation_data=(ffts_valid, labels_valid))

        logging.info(f"Done training model for evaluation")

        labels_pred = model.predict(ffts_valid)

        labels_valid_flat = np.argmax(labels_valid, axis=1)
        labels_pred_flat = np.argmax(labels_pred, axis=1)

        acc = accuracy_score(labels_valid_flat, labels_pred_flat)
        cm = confusion_matrix(labels_valid_flat, labels_pred_flat)
        logging.info(f"Got {acc:.5f} accuracy.\nConfusion matrix:\n{cm}")

    logging.info("Model real train...")

    model.fit(ffts, labels, batch_size=64, epochs=30)

    logging.info(f"Done training model")
    return model


T = TypeVar('T')


class MedianLastN(Generic[T]):
    def __init__(self, n: int, value: T = -1):
        self.n: int = n
        self.last: List[T] = [value] * n
        self.cur: int = 0

    def register(self, value: T):
        self.last[self.cur] = value
        self.cur = (self.cur + 1) % self.n

    def get_median(self) -> T:
        c: Dict[T, int] = {}
        for item in self.last:
            c[item] = c.get(item, 0) + 1
        med = max(c.items(), key=lambda x: x[1])[0]
        return med


if __name__ == "__main__":
    LIST_DEVICES = False  # list audio devices and exit
    DEVICE = None  # input device (numeric ID or substring)

    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT, datefmt=config.LOG_DATE_FMT)

    _model = make_model(evaluate=True)

    try:
        if LIST_DEVICES:
            logging.info(f"Devices: {sd.query_devices()}")
            exit(0)

        _samplerate = sd.query_devices(DEVICE, "input")["default_samplerate"]
        _block_size = int(_samplerate * config.BLOCK_DURATION / 1000)
        _window = ssw.blackman(_block_size)
        _fft_size = config.FFT_SIZE

        mln = MedianLastN(7, -1)

        def _callback(in_data, _, __, ___):
            sound = in_data.T[0]
            if max(sound) < 0.01:
                return
            # logging.info(f"in_data: {len(in_data.flatten())}")
            _this_fft = abs(fft(_window * sound, _fft_size * config.FFT_RATIO)[:_fft_size])
            _this_set = _this_fft[None, :]
            _this_labels = _model.predict(_this_set)
            label_pred = np.argmax(_this_labels, axis=1)[0]

            mln.register(label_pred)
            med = mln.get_median()

            logging.info(f"Predicted: {local_classes_i2n.get(med)}")

        with sd.InputStream(device=DEVICE, channels=1, callback=_callback,
                            blocksize=_block_size,
                            samplerate=_samplerate):
            while True:
                response = input()
                if response in ("", "q", "Q"):
                    break

    except KeyboardInterrupt:
        exit("Interrupted by user")
    except Exception as e:
        exit(type(e).__name__ + ": " + str(e))

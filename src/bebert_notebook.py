"""
<center><h1>DearLearning - Can you ear that?</h1></center>

Project idea: use Deep Learning to ear the difference between instruments
"""
# %%
# import sys

# CWD = "/Users/aymeric/Cours/DeepLearning/DearLearning"
# sys.path.insert(0, CWD)
# __file__ = 'Tests.ipynb'

# % load_ext autoreload
# % autoreload 2
#
# % matplotlib inline
# %load_ext Cython

# Built-in imports
import os
import time
import json

# Installed libraries
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal import windows as ssw

# Project imports
from src.utils import resolve_path, pretty_duration

# %%
data_path = resolve_path("Data")
dataset_path = resolve_path("Data/nsynth-valid")


# %%
def file_to_fft(filename, plot=False):
    """Get frequency sampling and FFT of the wav file"""
    fs, data = wavfile.read(filename)

    # Get one soundtrack if stereo sound
    if len(data.shape) == 2:
        data = data.T[0]

    # Scale [-1; 1]
    scale = max(-min(data), max(data))
    b = data.astype(float) / scale

    # Compute fft
    c = fft(b, 256)

    # Show result
    rv = abs(c[:len(c) // 2])
    if plot:
        plt.plot(rv, 'r')

    return fs, rv


# %%
_fs, _data = wavfile.read(os.path.join(dataset_path, "audio", "guitar_acoustic_030-076-127.wav"))
print(f"Frequency sampling: {_fs}, data shape: {_data.shape}, data sample: {_data[:5]}...")


# %%
def wave_file_to_time_data(fname):
    """Get content of the wav file, normalize the highest peak to +/- 1.0"""
    fs, data = wavfile.read(fname)

    if len(data.shape) == 2:
        data = data.T[0]

    scale = max(-min(data), max(data))
    td = data.astype(float) / scale
    return fs, td


WINDOW_SECOND_FRACTION = 10  # 0.1s window
STEP_FRACTION = 8  # window move 1/this_number right each step
FFT_SIZE = 256  # Width of FFT window
SAMPLE_DURATION = 2  # 2s sound (padding or crop)


def make_spectrogram(fs, time_data, plot=False):
    """From frequency sampling and content of a wav file, make the spectrogram"""
    window_size = fs // WINDOW_SECOND_FRACTION + ((fs // WINDOW_SECOND_FRACTION) % 2 == 0)
    window = ssw.blackman(window_size)

    sample_size = fs * SAMPLE_DURATION
    step = window_size // STEP_FRACTION
    n_steps = sample_size // step

    padding_left = np.zeros(step)
    padding_right = np.zeros(max(0, sample_size - len(time_data) - step))
    time_data_2 = np.concatenate((padding_left, time_data, padding_right))[:sample_size]

    spectrogram = np.zeros((n_steps, FFT_SIZE), dtype=float)
    for k in range(n_steps - STEP_FRACTION):
        sound_part = time_data_2[k * step:k * step + window_size]
        sound_windowed = window * sound_part
        ft = abs(fft(sound_windowed, FFT_SIZE * 2)[:FFT_SIZE])
        spectrogram[k, :] = ft

    if plot:
        print(f"Window size: {window_size}, time_data size: {len(time_data_2)}, spectrogram: {spectrogram.shape}")
        plt.plot(window)

        plt.imshow(spectrogram, interpolation='nearest')
        plt.show()

    return spectrogram


def make_file_spectrogram(fname, plot=False):
    """Make the spectrogram of the wav file"""
    return make_spectrogram(*wave_file_to_time_data(fname), plot=plot)


# %%
make_file_spectrogram(os.path.join(dataset_path, "audio", "guitar_acoustic_030-076-127.wav"), plot=True)
make_file_spectrogram(os.path.join(dataset_path, "audio", "guitar_acoustic_030-077-075.wav"), plot=True)
make_file_spectrogram(os.path.join(dataset_path, "audio", "guitar_acoustic_030-078-050.wav"), plot=True)
make_file_spectrogram(os.path.join(dataset_path, "audio", "mallet_acoustic_047-094-127.wav"), plot=True)
make_file_spectrogram(os.path.join(data_path, "broken_clarinet.wav"), plot=True)
make_file_spectrogram(os.path.join(data_path, "brah.wav"), plot=True)
print("Done")
# %%

# %%
_fs, _ft = file_to_fft(os.path.join(dataset_path, "audio", "guitar_acoustic_030-076-127.wav"), plot=True)
# %%
_fs, _ft = file_to_fft(os.path.join(dataset_path, "audio", "guitar_acoustic_030-077-075.wav"), plot=True)
# %%
_fs, _ft = file_to_fft(os.path.join(dataset_path, "audio", "guitar_acoustic_030-078-050.wav"), plot=True)
# %%
_fs, _ft = file_to_fft(os.path.join(dataset_path, "audio", "mallet_acoustic_047-094-127.wav"), plot=True)
# %%
_fs, _ft = file_to_fft(os.path.join(dataset_path, "audio", "bass_synthetic_134-097-127.wav"), plot=True)
# %%
_fs, _ft = file_to_fft(os.path.join(data_path, "broken_clarinet.wav"), plot=True)
# %%

# %%
with open(os.path.join(dataset_path, "examples.json"), "r") as f:
    datastore = json.load(f)
# %%
len(datastore.keys())
# %%
test_sample = 'brass_acoustic_059-036-075'

print(datastore[test_sample])
# %%
make_file_spectrogram(os.path.join(dataset_path, "audio", test_sample + ".wav"), plot=True)
print("Done")
# %%

# %%
# Performance
t0 = time.perf_counter()

a = make_file_spectrogram(os.path.join(dataset_path, "audio", test_sample + ".wav"))

t1 = time.perf_counter()
print(f"Spectrogram made in {pretty_duration(t1 - t0)}")
# %%
corr = {}
for s in datastore.values():
    family = s["instrument_family_str"]
    name = s["note_str"]
    if s["instrument_source_str"] == "acoustic":
        if family not in corr:
            corr[family] = []
        corr[family].append(name)

classes_i2n = {i: k for i, k in enumerate(corr.keys())}
classes_n2i = {k: i for i, k in classes_i2n.items()}
num_classes = len(classes_i2n)
print(classes_n2i)
# %%
labels = []
spectrograms = []

for k, file_names in corr.items():
    print(f"Making {k} data...")
    for fn in file_names:
        labels.append(classes_n2i[k])
        spectrograms.append(make_file_spectrogram(os.path.join(dataset_path, "audio", fn + ".wav")))
# %%
print(spectrograms[1000])
print(labels[1000])
# %%
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
# from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
# from keras.utils import np_utils
#
# model_m = Sequential()
# model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
# model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
# model_m.add(Conv1D(100, 10, activation='relu'))
# model_m.add(MaxPooling1D(3))
# model_m.add(Conv1D(160, 10, activation='relu'))
# model_m.add(Conv1D(160, 10, activation='relu'))
# model_m.add(GlobalAveragePooling1D())
# model_m.add(Dropout(0.5))
# model_m.add(Dense(num_classes, activation='softmax'))
# print(model_m.summary())
# %%
# from keras.utils import plot_model
#
# plot_model(model, to_file=resolve_path('Data', f'model_img.png', show_shapes=True, show_layer_names=True)
# %%

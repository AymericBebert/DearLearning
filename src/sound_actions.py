"""
Functions on sound
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal import windows as ssw

import config


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
        plt.title(os.path.basename(filename))
        plt.show()

    return fs, rv


def wave_file_to_time_data(fname):
    """Get content of the wav file, normalize the highest peak to +/- 1.0"""
    fs, data = wavfile.read(fname)

    if len(data.shape) == 2:
        data = data.T[0]

    scale = max(-min(data), max(data))
    td = data.astype(float) / scale
    return fs, td


def make_spectrogram(fs, time_data, plot=False):
    """From frequency sampling and content of a wav file, make the spectrogram"""
    window_size = fs // config.WINDOW_SECOND_FRACTION
    window_size += window_size % 2 == 0
    window = ssw.blackman(window_size)

    sample_size = fs * config.SAMPLE_DURATION
    step = window_size // config.STEP_FRACTION
    n_steps = sample_size // step

    padding_left = np.zeros(step)
    padding_right = np.zeros(max(0, sample_size - len(time_data) - step))
    time_data_2 = np.concatenate((padding_left, time_data, padding_right))[:sample_size]

    fft_size = config.FFT_SIZE
    spectrogram = np.zeros((n_steps, fft_size), dtype=float)
    for k in range(n_steps - config.STEP_FRACTION):
        sound_part = time_data_2[k * step:k * step + window_size]
        sound_windowed = window * sound_part
        ft = abs(fft(sound_windowed, fft_size * 2)[:fft_size])
        spectrogram[k, :] = ft

    if plot:
        plt.title(f"fs: {fs}, time_data size: {len(time_data_2)}, spectrogram: {spectrogram.shape}")
        # plt.plot(window)

        plt.imshow(spectrogram, interpolation='nearest')
        plt.show()

    return spectrogram


def make_file_spectrogram(fname, plot=False):
    """Make the spectrogram of the wav file"""
    return make_spectrogram(*wave_file_to_time_data(fname), plot=plot)

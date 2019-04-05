"""
Functions on sound
"""

import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal import windows as ssw

import config


def file_to_fft(filename: str, plot: bool = False, desc: str = '',
                fft_size: int = config.FFT_SIZE,
                fft_ratio: int = config.FFT_RATIO,
                ) -> Tuple[int, np.ndarray]:
    """Get frequency sampling and FFT of the wav file"""
    fs, data = wavfile.read(filename)

    # Get one soundtrack if stereo sound
    if len(data.shape) == 2:
        data = data.T[0]

    # Scale [-1; 1]
    scale = max(-min(data), max(data))
    b = data.astype(float) / scale

    # Compute fft
    c = fft(b, fft_size * fft_ratio)

    # Show result
    rv = abs(c[:fft_size])
    if plot:
        title = desc + ' - ' if desc else ''
        plt.title(f"{title}{fs} - {data.shape}")

        plt.plot(rv, 'r')
        plt.title(os.path.basename(filename))
        plt.show()

    return fs, rv


def wave_file_to_time_data(fname: str):
    """Get content of the wav file, normalize the highest peak to +/- 1.0"""
    fs, data = wavfile.read(fname)

    if len(data.shape) == 2:
        data = data.T[0]

    scale = max(-min(data), max(data))
    td = data.astype(float) / scale
    return fs, td


def make_spectrogram(fs: int, time_data: np.ndarray, plot: bool = False, desc: str = '',
                     sample_duration: int = config.SAMPLE_DURATION,
                     block_duration: int = config.BLOCK_DURATION,
                     step_fraction: int = config.STEP_FRACTION,
                     fft_size: int = config.FFT_SIZE,
                     fft_ratio: int = config.FFT_RATIO,
                     ) -> np.ndarray:
    """From frequency sampling and content of a wav file, make the spectrogram"""
    window_size = int(fs * block_duration / 1000)
    window = ssw.blackman(window_size)

    sample_size = fs * sample_duration
    step = window_size // step_fraction
    n_steps = sample_size // step

    padding_left = np.zeros(step)
    padding_right = np.zeros(max(0, sample_size - len(time_data) - step))
    time_data_2 = np.concatenate((padding_left, time_data, padding_right))[:sample_size]

    spectrogram = np.zeros((n_steps, fft_size), dtype=float)
    for k in range(n_steps - step_fraction):
        sound_part = time_data_2[k * step:k * step + window_size]
        sound_windowed = window * sound_part
        ft = abs(fft(sound_windowed, fft_size * fft_ratio)[:fft_size])
        spectrogram[k, :] = ft

    if plot:
        title = desc + ' - ' if desc else ''
        plt.title(f"{title}{fs} - {spectrogram.shape}")

        plt.imshow(spectrogram, interpolation='nearest')
        plt.show()

    return spectrogram


def make_file_spectrogram(fname: str, plot: bool = False, **kwargs) -> np.ndarray:
    """Make the spectrogram of the wav file"""
    return make_spectrogram(*wave_file_to_time_data(fname), plot=plot, desc=os.path.basename(fname), **kwargs)


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

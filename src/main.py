#!/usr/bin/env python3

import os
import time
import logging

import config
from src.utils import resolve_path, pretty_duration
from src.sound_actions import make_file_spectrogram, file_to_fft
from src.dataset_actions import load_dataset_json, extract_tracks_by_family, make_spectrograms_and_labels


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT, datefmt=config.LOG_DATE_FMT)

    data_path = resolve_path(config.DATA_PATH)
    dataset_path = resolve_path(config.DATASET_PATH)

    # # FFT of some files
    # file_to_fft(os.path.join(dataset_path, "audio", "guitar_acoustic_030-076-127.wav"), plot=True)
    # file_to_fft(os.path.join(dataset_path, "audio", "guitar_acoustic_030-077-075.wav"), plot=True)
    # file_to_fft(os.path.join(dataset_path, "audio", "guitar_acoustic_030-078-050.wav"), plot=True)
    # file_to_fft(os.path.join(dataset_path, "audio", "mallet_acoustic_047-094-127.wav"), plot=True)
    # file_to_fft(os.path.join(dataset_path, "audio", "bass_synthetic_134-097-127.wav"), plot=True)
    # file_to_fft(os.path.join(data_path, "broken_clarinet.wav"), plot=True)

    # # Spectrograms of some files
    # make_file_spectrogram(os.path.join(dataset_path, "audio", "guitar_acoustic_030-076-127.wav"), plot=True)
    # make_file_spectrogram(os.path.join(dataset_path, "audio", "guitar_acoustic_030-077-075.wav"), plot=True)
    # make_file_spectrogram(os.path.join(dataset_path, "audio", "guitar_acoustic_030-078-050.wav"), plot=True)
    # make_file_spectrogram(os.path.join(dataset_path, "audio", "mallet_acoustic_047-094-127.wav"), plot=True)
    # make_file_spectrogram(os.path.join(data_path, "broken_clarinet.wav"), plot=True)
    # make_file_spectrogram(os.path.join(data_path, "brah.wav"), plot=True)

    dataset_json = load_dataset_json(os.path.join(dataset_path, "examples.json"))

    test_sample = 'brass_acoustic_059-036-075'
    make_file_spectrogram(os.path.join(dataset_path, "audio", test_sample + ".wav"), plot=True)

    # Performance
    t0 = time.perf_counter()

    a = make_file_spectrogram(os.path.join(dataset_path, "audio", test_sample + ".wav"))

    t1 = time.perf_counter()
    logging.info(f"Spectrogram made in {pretty_duration(t1 - t0)}")

    data_store = extract_tracks_by_family(dataset_json)

    # Make all the spectrograms. This can take some time
    ids, labels, spectrograms, classes_i2n, classes_n2i = make_spectrograms_and_labels(data_store)

    num_classes = len(classes_i2n)
    logging.info(f"{num_classes} classes: {classes_n2i}")

    logging.info(spectrograms[1000])
    logging.info(labels[1000])
    logging.info("All done")

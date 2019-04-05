#!/usr/bin/env python3

import os
import time
import pickle
import hashlib
import logging

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, GlobalMaxPooling1D
from keras.utils import to_categorical, plot_model

import config
from src.utils import resolve_path, pretty_duration
from src.sound_actions import make_file_spectrogram, file_to_fft
from src.dataset_actions import load_dataset_json, extract_tracks_by_family, make_labels_and_spectrograms
from src.metrics import plot_confusion_matrix


if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT, datefmt=config.LOG_DATE_FMT)

    data_path = resolve_path(config.DATA_PATH)
    dataset_path = resolve_path(config.DATASET_PATH)
    cache_path = resolve_path(config.CACHE_PATH)

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

    # test_sample = 'brass_acoustic_059-036-075'
    # make_file_spectrogram(os.path.join(dataset_path, "audio", test_sample + ".wav"), plot=True)

    # # Performance
    # t0 = time.perf_counter()
    #
    # a = make_file_spectrogram(os.path.join(dataset_path, "audio", test_sample + ".wav"))
    #
    # t1 = time.perf_counter()
    # logging.info(f"Spectrogram made in {pretty_duration(t1 - t0)}")

    # Extract data from dataset json
    exclude_families = config.DATASET_EXCLUDE_FAMILIES
    exclude_sources = config.DATASET_EXCLUDE_SOURCES
    dataset_json = load_dataset_json(os.path.join(dataset_path, "examples.json"))

    def file_access(fn):
        return resolve_path(config.DATASET_PATH, "audio", fn + ".wav")

    data_store, classes_n2i = extract_tracks_by_family(dataset_json,
                                                       exclude_families=exclude_families,
                                                       exclude_source=exclude_sources)

    # Make all the spectrograms. This can take some time. Put data into cache for faster reload
    os.makedirs(cache_path, exist_ok=True)
    dataset_hash = hashlib.sha1(f"{dataset_path}-{exclude_families}-{exclude_sources}".encode('utf-8')).hexdigest()
    cache_file_path = os.path.join(cache_path, dataset_hash + ".pickle")

    try:
        with open(cache_file_path, "rb") as cf:
            logging.info("Loading data from cache...")
            ids, labels, spectrograms = pickle.load(cf)
    except Exception as e:
        logging.info(f"Could not load cache, making data from source... ({e})")
        ids, labels, spectrograms = make_labels_and_spectrograms(data_store, file_access)
        logging.info("Saving data to cache...")
        with open(cache_file_path, "wb") as cf:
            pickle.dump((ids, labels, spectrograms), cf)

    num_classes = len(classes_n2i)
    logging.info(f"{num_classes} classes: {classes_n2i}")

    # Shuffle dataset
    indices = np.arange(len(ids))
    np.random.shuffle(indices)
    ids = np.asarray(ids)[indices]
    labels = np.asarray(labels)[indices]
    spectrograms_2d = np.asarray(spectrograms)[indices]

    labels = to_categorical(np.asarray(labels))
    spectrograms_3d = np.expand_dims(spectrograms_2d, axis=3)

    # Cut dataset into train and validation sets
    cut = int(ids.shape[0] * 0.8)

    ids_train, ids_valid = ids[:cut], ids[cut:]
    labels_train, labels_valid = labels[:cut], labels[cut:]
    spectrograms_2d_train, spectrograms_2d_valid = spectrograms_2d[:cut], spectrograms_2d[cut:]
    spectrograms_3d_train, spectrograms_3d_valid = spectrograms_3d[:cut], spectrograms_3d[cut:]

    logging.info("Dataset ready.")

    #
    # Make and train the model
    #
    # # Model 01
    # model = Sequential()
    #
    # model.add(Conv2D(64, 5, input_shape=(160, 256, 1), activation='relu'))
    # model.add(MaxPooling2D())
    # model.add(Flatten())
    # model.add(Dropout(0.25))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax'))
    #
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # plot_model(model, to_file=os.path.join(cache_path, "model_1.png"), show_shapes=True, show_layer_names=True)
    #
    # model.fit(spectrograms_3d_train,
    #           labels_train,
    #           batch_size=64,
    #           epochs=50,
    #           validation_data=(spectrograms_3d_valid, labels_valid))

    # Model 02
    model = Sequential()

    model.add(Conv1D(64, 5, input_shape=(160, 256), activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    plot_model(model, to_file=os.path.join(cache_path, "CNN_02.png"), show_shapes=True, show_layer_names=True)

    N_EPOCH = 30
    t0 = time.perf_counter()

    model.fit(spectrograms_2d_train,
              labels_train,
              batch_size=64,
              epochs=50,
              validation_data=(spectrograms_2d_valid, labels_valid))

    logging.info(f"Done training model in {pretty_duration(time.perf_counter() - t0)} ({N_EPOCH} epochs)")

    #
    # Evaluate results
    #
    # Predict on validation set
    labels_pred = model.predict(spectrograms_2d_valid)

    logging.info(f"labels_valid.shape: {labels_valid.shape}")
    logging.info(f"labels_pred.shape: {labels_pred.shape}")

    labels_valid_flat = np.argmax(labels_valid, axis=1)
    labels_pred_flat = np.argmax(labels_pred, axis=1)

    # Scores
    logging.info(f"Accuracy:{skm.accuracy_score(labels_valid_flat, labels_pred_flat):.5f}")
    logging.info(f"Confusion matrix:\n{skm.confusion_matrix(labels_valid_flat, labels_pred_flat)}")

    # Pretty confusion matrix
    class_names = [k.split("_")[0] for k, v in sorted(classes_n2i.items(), key=lambda x: x[1])]
    np.set_printoptions(precision=2)

    plot_confusion_matrix(labels_valid_flat, labels_pred_flat, classes=class_names)
    plot_confusion_matrix(labels_valid_flat, labels_pred_flat, classes=class_names, normalize=True)
    plt.show()

    logging.info("All done")

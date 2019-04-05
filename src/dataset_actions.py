"""
Functions to manage dataset
"""

import json
import logging
from typing import Dict, List, Tuple, Any, Callable

import numpy as np

from src.sound_actions import make_file_spectrogram, give_many_fft, wave_file_to_time_data


def load_dataset_json(json_path: str):
    """From the json annotating all the wav files, get useful info"""
    with open(json_path, "r") as jf:
        return json.load(jf)


def extract_tracks_by_family(dataset_json: Dict[str, Any], exclude_families=None,
                             exclude_source=None) -> Tuple[Dict[int, List[str]], Dict[str, int]]:
    """From the dict coming from nsynth json, group instruments by family.
    :returns: {class_index -> files} and {class_name -> class_index}
    """
    exclude_families = [] if exclude_families is None else exclude_families
    exclude_source = [] if exclude_source is None else exclude_source

    classes_n2i: Dict[str, int] = {}

    corr = {}
    for s in dataset_json.values():
        # instrument family
        family = s["instrument_family_str"]
        if family in exclude_families:
            continue

        # track name and instrument source
        name = s["note_str"]
        source = s["instrument_source_str"]

        # register track name under index of {family}_{source} key
        if source not in exclude_source:
            fs = f"{family}_{source}"
            if fs not in classes_n2i:
                classes_n2i[fs] = len(classes_n2i)
                corr[classes_n2i[fs]] = []
            corr[classes_n2i[fs]].append(name)
    return corr, classes_n2i


def make_labels_and_spectrograms(corr: Dict[int, List[str]], file_access: Callable[[str], str] = lambda x: x):
    """Make the spectrogram of each file in the data dict
    :returns: Lists for ids, labels, spectrograms
    """
    ids: List[str] = []
    labels: List[int] = []
    spectrograms: List[np.ndarray] = []

    for k, file_names in corr.items():
        logging.info(f"Making {k} data...")
        for fn in file_names:
            ids.append(fn)
            labels.append(k)
            spectrograms.append(make_file_spectrogram(file_access(fn)))

    return ids, labels, spectrograms


def make_labels_and_fft(corr: Dict[int, List[str]], file_access: Callable[[str], str] = lambda x: x, **kwargs) -> Tuple[List[int], List[np.ndarray], List[str]]:
    """Make the fft of some samples from each file in the data dict
    :returns: Lists for ids, labels, spectrograms; lookups {class_index -> class_name}, {class_name -> class_index}
    """
    labels: List[int] = []
    data: List[np.ndarray] = []
    source: List[str] = []

    for k, file_names in corr.items():
        logging.info(f"Making {k} data...")
        for fn in file_names:
            file_full_name = file_access(fn)
            for out in give_many_fft(*wave_file_to_time_data(file_full_name), **kwargs):
                labels.append(k)
                data.append(out)
                source.append(fn)
        logging.info(f"- Total: {len(labels)} entries...")

    return labels, data, source

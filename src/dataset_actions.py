"""
Functions to manage dataset
"""

import json
import logging
from typing import Dict, List, Any

import numpy as np

import config
from src.utils import resolve_path
from src.sound_actions import make_file_spectrogram


def load_dataset_json(json_path: str):
    """From the json annotating all the wav files, get useful info"""
    with open(json_path, "r") as jf:
        return json.load(jf)


def extract_tracks_by_family(dataset_json: Dict[str, Any],
                             exclude_families=None, exclude_source=None) -> Dict[str, List[str]]:
    """From the dict coming from nsynth json, group instruments by family"""
    exclude_families = [] if exclude_families is None else exclude_families
    exclude_source = [] if exclude_source is None else exclude_source

    corr = {}
    for s in dataset_json.values():
        # instrument family
        family = s["instrument_family_str"]
        if family in exclude_families:
            continue

        # track name and instrument source
        name = s["note_str"]
        source = s["instrument_source_str"]

        # register track name under {family}_{source} key
        if source not in exclude_source:
            fs = f"{family}_{source}"
            if fs not in corr:
                corr[fs] = []
            corr[fs].append(name)
    return corr


def make_spectrograms_and_labels(corr: Dict[str, List[str]]):
    """Make the spectrogram of each file in the data dict
    :returns: Lists for ids, labels, spectrograms; lookups {class_index -> class_name}, {class_name -> class_index}
    """
    classes_i2n: Dict[int, str] = {i: k for i, k in enumerate(corr.keys())}
    classes_n2i: Dict[str, int] = {k: i for i, k in classes_i2n.items()}

    ids: List[str] = []
    labels: List[int] = []
    spectrograms: List[np.ndarray] = []

    for k, file_names in corr.items():
        logging.info(f"Making {k} data...")
        for fn in file_names:
            ids.append(fn)
            labels.append(classes_n2i[k])
            spectrograms.append(make_file_spectrogram(resolve_path(config.DATASET_PATH, "audio", fn + ".wav")))

    return ids, labels, spectrograms, classes_i2n, classes_n2i

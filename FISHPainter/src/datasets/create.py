import warnings
warnings.filterwarnings(
    "ignore",
    message="Argument fill/fillcolor is not supported for Tensor input. Fill value is zero",
    category=UserWarning
)

import os
import yaml
import numpy as np
from numbers import Number
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from . import ids as ID
from ..utils.BackgroundTransformer import BackgroundTransformer
<<<<<<< HEAD
from ... import FISHPainter_home
from ..signals import create_FISH
=======
from pathlib import Path
from ..signals import create_FISH
from tqdm import tqdm
import os
from ...config import FISHPainter_home
from typing import Union, Literal
>>>>>>> 27a9c29e9d31394bca6130fdc94f68f7a201365b

DEFAULT_BACKGROUNDS = FISHPainter_home / "cell_backgrounds_128.h5"

POSSIBLE_TYPES = [ID.copynumber, ID.alt]  # here add ALT optional

POSSIBLE_ARGS = {
    ID.copynumber: [
        'number',
        'num_red', 'num_red_cluster', 'red_cluster_size',
        'num_green', 'num_green_cluster', 'green_cluster_size',
        'signal_size', 'signal_intensity',
        'target_class'
    ],
    ID.alt: [
        # tbd
    ]
}


def extract_parameters(parameters, possible_args):
    return {key: parameters.pop(key, None) for key in possible_args}


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


<<<<<<< HEAD
# ------------------------------
# Parallel helpers (process-wide)
# ------------------------------
_GLOBAL_BG = None  # opened once per worker process


def _init_worker(bg_path_str):
    """Initializer: open the BackgroundTransformer once per worker process."""
    global _GLOBAL_BG
    _GLOBAL_BG = BackgroundTransformer(bg_path_str)


def _make_one_image(task_tuple):
    """
    Worker function that generates one RGB patch + metadata.
    Receives a tuple to keep pickling simple.
    """
    (parameters, signal_size, signal_intensity, alpha, sigma, merge_bboxes, target_class, seed) = task_tuple

    # Ensure this task has a unique RNG state for everything that uses np.random
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Sample params and a random background
    instance_parameters = define_instance_parameters(parameters, rng=rng)
=======
def create_dataset(config_file, FISH_type: Union[Literal[ID.copynumber], Literal[ID.alt]] = ID.copynumber, background_path=None, verbose=False):
>>>>>>> 27a9c29e9d31394bca6130fdc94f68f7a201365b
    
    idx = rng.integers(0, len(_GLOBAL_BG))
    instance_background, instance_mask = _GLOBAL_BG[idx]

    rgb_patch = create_FISH(
        instance_background,
        instance_mask,
        **instance_parameters,
        signal_size=signal_size,
        signal_intensity=signal_intensity,
        return_as_dict=True,
        alpha=alpha,
        sigma=sigma,
        merge_bboxes=merge_bboxes
    )["patch"]

    return (
        rgb_patch,
        instance_mask,
        parameters_tolist(instance_parameters, parameters),
        target_class
    )


def create_dataset(
    config_file,
    FISH_type,
    background_path=None,
    verbose=False,
    alpha=20,
    sigma=2,
    merge_bboxes=False,
    base_seed=None,
    parallel=True# optional for reproducibility
):
    assert FISH_type in POSSIBLE_TYPES, f"{FISH_type} is not implemented! Chose one of {POSSIBLE_TYPES}"

    config = load_config(config_file)
    print(f"Found the following {len(config.keys())} conditions: " + ", ".join(k for k in config.keys()))

    # Always pass a path string; workers will open the dataset themselves
    bg_path = str(background_path) if background_path is not None else str(DEFAULT_BACKGROUNDS)
    if background_path is not None:
        assert Path(background_path).exists(), "Provided path to background file does not exist!"

    if FISH_type == ID.copynumber:
        dataset = create_copynumber_dataset(
            config,
            bg_path,
            alpha,
            sigma,
            verbose=verbose,
            merge_bboxes=merge_bboxes,
            base_seed=base_seed,
            parallel=parallel
        )
    elif FISH_type == ID.alt:
        print("NOT YET IMPLEMENTED")
        dataset = {}
    return dataset


def create_copynumber_dataset(config, background_path, alpha, sigma, verbose=False, merge_bboxes=False, base_seed=None, parallel=True):
    results = {}
    
    max_workers = os.cpu_count() if parallel else 1

    # SeedSequence to guarantee DIFFERENT seeds per image across processes
    # If base_seed is None, it uses fresh entropy; otherwise it is reproducible.
    master_ss = np.random.SeedSequence(base_seed)

    for condition, parameters in config.items():
        
        parameters = extract_parameters(parameters, POSSIBLE_ARGS[ID.copynumber])
        target_class = parameters.pop("target_class")
        signal_size = parameters.pop("signal_size")
        signal_intensity = parameters.pop('signal_intensity')
        n_images = parameters.pop("number")

        results[condition] = dict(rgb_patches=[], masks=[], parameters=[], target_classes=[])

        # Pre-generate unique child seeds for each image
        child_seqs = master_ss.spawn(n_images)
        seeds = [int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in child_seqs]

        # Bundle constants for each task; vary only the seed
        common_args = (parameters, signal_size, signal_intensity, alpha, sigma, merge_bboxes, target_class)

        # Create a process pool where each worker opens the HDF5 once
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(str(background_path),)
        ) as ex:
            # Submit all tasks first
            futures = [
                ex.submit(_make_one_image, (*common_args, s))
                for s in seeds
            ]

            # Advance the bar on COMPLETION (not submission)
            iterator = as_completed(futures)
            if verbose:
                iterator = tqdm(iterator, total=n_images, position=0, desc=f"Working on {condition}")

            for fut in iterator:
                rgb_patch, instance_mask, param_list, tgt = fut.result()
                results[condition]["rgb_patches"].append(rgb_patch)
                results[condition]["masks"].append(instance_mask)
                results[condition]["parameters"].append(param_list)
                results[condition]["target_classes"].append(tgt)
                

    return results


def parameters_tolist(instance_parameters, possible_parameters):
    # possible_parameters can be a dict; iterating preserves key order (Py3.7+)
    return [instance_parameters[pp] if pp in instance_parameters else 0 for pp in possible_parameters]


def define_instance_parameters(parameters, rng=None):
    """Sample instance parameters; uses provided rng for determinism per image."""
    if rng is None:
        rng = np.random.default_rng()
        
    instance_parameters = {}
    for key, value in parameters.items():
        
        if isinstance(value, Number):
            instance_parameters[key] = value
            
        elif isinstance(value, list):
            instance_parameters[key] = int(rng.integers(value[0], value[1] + 1))
    
    return instance_parameters
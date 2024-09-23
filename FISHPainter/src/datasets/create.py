import yaml
from . import ids as ID
import numpy as np
from numbers import Number
from ..utils.BackgroundTransformer import BackgroundTransformer
from pathlib import Path
from ..signals import create_FISH
from tqdm import tqdm
import os
from ...config import FISHPainter_home
from typing import Union, Literal

DEFAULT_BACKGROUNDS = FISHPainter_home / "cell_backgrounds_128.h5"

POSSIBLE_TYPES = [ID.copynumber, ID.alt] #here add ALT optional

POSSIBLE_ARGS = {
    
    ID.copynumber: [
        'number', 
        'num_red', 'num_red_cluster', 'red_cluster_size',
        'num_green', 'num_green_cluster','green_cluster_size',
        'signal_size', 
        'target_class'],
    
    ID.alt: [
        #tbd
        ]

    }

def extract_parameters(parameters, possible_args):
    
    return {key: parameters.pop(key, None) for key in possible_args}


def load_config(config_file):

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        
    return config


def create_dataset(config_file, FISH_type: Union[Literal[ID.copynumber], Literal[ID.alt]] = ID.copynumber, background_path=None, verbose=False):
    
    assert FISH_type in POSSIBLE_TYPES, f"{FISH_type} is not implemented! Chose one of {POSSIBLE_TYPES}"
    
    config = load_config(config_file)
    
    print(f"Found the following {len(config.keys())} conditions: " + ", ".join(k for k in config.keys()))
    
    if background_path is not None:
        assert Path(background_path).exists(), "Provided path to background file does not exist!"
        background_dataset = BackgroundTransformer(background_path)
    
    else:
        background_dataset = BackgroundTransformer(DEFAULT_BACKGROUNDS)
        

    if FISH_type == ID.copynumber:
        
        dataset = create_copynumber_dataset(config, background_dataset, verbose=verbose)
    
    elif FISH_type == ID.alt:
        print("NOT YET IMPLEMENTED")
        
    return dataset


def create_copynumber_dataset(config, background_dataset, verbose=False):
    
    # config_bar = tqdm(config.items(), position=0) if verbose else config.items()
    config_bar = config.items()

    results = {}
    for condition, parameters in config_bar:
        
        parameters = extract_parameters(parameters, POSSIBLE_ARGS[ID.copynumber])
        target_class = parameters.pop("target_class")
        signal_size = parameters.pop("signal_size")
        n_images = parameters.pop("number")
        
        number_bar = tqdm(range(n_images), position=0, desc=f"Working on {condition}") if verbose else range(n_images)

        results[condition] = dict(rgb_patches=[], masks=[], parameters=[], target_classes=[])
        for n in number_bar:
            
            instance_parameters = define_instance_parameters(parameters)
            instance_background, instance_mask = background_dataset[np.random.randint(0, len(background_dataset))]
        
            rgb_patch = create_FISH(
                instance_background, 
                instance_mask, 
                **instance_parameters, 
                signal_size=signal_size, 
                return_as_dict=True)["patch"]
                        
            results[condition]["rgb_patches"].append(rgb_patch)
            results[condition]["masks"].append(instance_mask)
            results[condition]["parameters"].append(parameters_tolist(instance_parameters, parameters))
            results[condition]["target_classes"].append(target_class)

    return results
            
            
def parameters_tolist(instance_parameters, possible_parameters):
    
    return [instance_parameters[pp] if pp in instance_parameters else 0 for pp in possible_parameters]     
        

def define_instance_parameters(parameters):
    
    instance_parameters = {}
    for key, value in parameters.items():
    
        if isinstance(value, Number):
            instance_parameters[key] = value
        
        elif isinstance(value, list):
            instance_parameters[key] = np.random.randint(value[0], value[1]+1)
            
    return instance_parameters
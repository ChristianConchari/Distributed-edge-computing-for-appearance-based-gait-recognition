"""
This module contains the create_dir function, which creates a directory at the specified path.
"""
from os import makedirs
from shutil import rmtree

def create_dir(folder: str, force: bool = True, verbose: bool = False) -> None:
    """
    Create a directory if it doesn't exist at the specified path.

    Parameters:
    folder (str): The path of the directory to be created.
    force (bool): If True, deletes the directory if it already 
        exists before creating a new one. Default is True.
    verbose (bool): If True, prints additional information 
        during the directory creation process. Default is False.
    """
    try:
        makedirs(folder)
        if verbose:
            print(f'Directory {folder} created succesfully.')
    except FileExistsError as e:
        if force:
            if verbose:
                print(e)
            rmtree(folder)
            makedirs(folder)
        else:
            if verbose:
                print(f'Directory {folder} already exists.')
        
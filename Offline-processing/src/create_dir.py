"""
This module contains the create_dir function, which creates a directory at the specified path.
"""
from os import makedirs
from shutil import rmtree
from argparse import ArgumentParser

def create_dir(folder: str, force: bool = True, verbose: bool = False) -> None:
    """
    Creates a directory at the specified path.

    Parameters:
    folder (str): The path of the directory to be created.
    force (bool): If True, deletes the directory if it already exists before creating a new one. Default is True.
    verbose (bool): If True, prints additional information during the directory creation process. Default is False.
    """
    try:
        makedirs(folder)
        
    except FileExistsError as e:
        if verbose:
            print(e)
        if force:
            rmtree(folder)
            makedirs(folder)

def main() -> None:
    """
    Main function that creates a directory based on the provided path.

    Args:
        None

    Returns:
        None
    """
    parser = ArgumentParser(description="Create a directory.")
    parser.add_argument("path", help="The path of the directory to be created.")
    parser.add_argument("--force", action='store_true', help="If set, deletes the directory if it already exists before creating a new one.")
    parser.add_argument("--verbose", action='store_true', help="If set, prints additional information during the directory creation process.")
    args = parser.parse_args()
    
    create_dir(args.path, force=args.force, verbose=args.verbose)

if __name__ == '__main__':
    main()
    
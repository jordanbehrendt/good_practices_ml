import argparse
import json
import os
from typing import Dict, List
import yaml
import shutil

def rename_folder(DATA_PATH: str, old_folder_name: str, new_folder_name: str) -> None:
    """Renames a folder in {DATA_PATH}/geoguessr.

    Args:
        DATA_PATH (str): Path to data folder.
        old_folder_name (str): The current name of the folder.
        new_folder_name (str): The new name of the folder.

    Returns:
        None
    """
    try:
        os.rename(os.path.join(DATA_PATH, "geoguessr", old_folder_name),
              os.path.join(DATA_PATH, "geoguessr", new_folder_name))
    except FileNotFoundError:
        print(f"Can't Rename: Folder '{old_folder_name}' not found.")


def remove_folder(DATA_PATH: str, folder_name: str) -> None:
    """Deletes a folder in {DATA_PATH}/geoguessr.

    Args:
        DATA_PATH (str): Path to data folder.
        folder_name (str): The name of the folder.

    Returns:
        None
    """
    try:
        shutil.rmtree(os.path.join(DATA_PATH, "geoguessr", folder_name))
    except FileNotFoundError:
        print(f"Can't Remove: Folder '{folder_name}' not found.")


def prepare_geoguessr_data(REPO_PATH: str, DATA_PATH: str) -> None:
    """Renames and deletes folders in {DATA_PATH}/geoguessr.
    Uses the data in changes.json as described in C.1.2 in the paper.

    Args:
        REPO_PATH (str): Path to project folder.
        DATA_PATH (str): Path to data folder.

    Returns:
        None
    """
    changes_file = open("{}/data_collection/geoguessr/changes.json".format(REPO_PATH))
    changes_json = json.load(changes_file)

    # Rename folders
    folders_to_rename: Dict[str, str] = changes_json["rename"]
    for old_folder_name, new_folder_name in folders_to_rename.items():
        rename_folder(DATA_PATH, old_folder_name, new_folder_name)

    # Delete folders
    folders_to_delete: List[str] = changes_json["delete"]
    for folder_name in folders_to_delete:
        remove_folder(DATA_PATH, folder_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare the geoguessr folders')
    parser.add_argument('--yaml_path', metavar='str', required=True, help='The path to the yaml file with the stored paths')
    args = parser.parse_args()
    with open(args.yaml_path) as file:
        paths = yaml.safe_load(file)
        DATA_PATH = paths['data_path']
        REPO_PATH = paths['repo_path']
        prepare_geoguessr_data(REPO_PATH, DATA_PATH)

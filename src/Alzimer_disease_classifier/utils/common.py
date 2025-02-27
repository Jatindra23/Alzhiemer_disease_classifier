import os
from box.exceptions import BoxValueError
import yaml
from Alzimer_disease_classifier.logger import logging
from Alzimer_disease_classifier.exception import RenelException
import sys
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise RenelException(e, sys)


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        logging.info(f"jason file saved at: {path}")
    except Exception as e:
        raise RenelException(e, sys)


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    try:
        with open(path, "rb") as f:
            content = json.load(f)
            return ConfigBox(content)
    except Exception as e:
        raise RenelException(e, sys)


@ensure_annotations
def save_bin(data: Any, path: Path):
    try:
        joblib.dump(value=data, filename=path)
    except Exception as e:
        raise RenelException(e, sys)


@ensure_annotations
def load_bin(path: Path) -> Any:
    try:
        return joblib.load(path)
    except Exception as e:
        raise RenelException(e, sys)


@ensure_annotations
def get_size(path: Path) -> str:
    """
    path: str
    """
    size_in_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
    return f"{size_in_mb} MB"


@ensure_annotations
def encode_to_base64(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode("utf-8")


@ensure_annotations
# def decodeImage(imgstring, filename):
#     imgdata = base64.b64decode(imgstring)
#     with open(filename, "wb") as f:
#         f.write(imgdata)
#         f.close()
def decodeImage(image_file, filename):
    """
    Saves the uploaded image to the specified filename.

    Args:
        image_file (FileStorage): The uploaded image file.
        filename (str): The path where the image will be saved.
    """
    try:
        image_file.save(filename)
        logging.info(f"Image saved successfully at {filename}.")
    except Exception as e:
        logging.error(f"Error saving image: {e}")
        raise


@ensure_annotations
def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


import shutil
import os


@ensure_annotations
def copy_model(src, dst):
    """
    Copy a model file or directory from src to dst.

    Parameters:
    - src (str): Source path (file or directory).
    - dst (str): Destination path (file or directory).

    Raises:
    - ValueError: If the source path does not exist.
    - FileExistsError: If the destination directory exists when copying a directory.
    """
    # Check if source exists
    if not os.path.exists(src):
        logging.error(f"Source path '{src}' does not exist.")

    # Check if destination is a directory
    dst_dir = os.path.dirname(dst)
    if not os.path.isdir(dst):
        os.makedirs(dst)
        logging.info((f"Destination directory '{dst_dir}' created."))

    # If source is a file, copy the file
    if os.path.isfile(src):
        shutil.copy(src, dst)
        logging.info(f"File copied successfully from '{src}' to '{dst}'.")

    # If source is a directory, copy the entire directory
    # elif os.path.isdir(src):
    #     shutil.copytree(src, dst)
    #     print(f"Directory copied successfully from '{src}' to '{dst}'.")

    else:
        logging.info(f"Source path '{src}' is neither a file nor a directory.")


# Example usage:
# copy_model('path/to/source/model.h5', 'path/to/destination/model.h5')  # For a file
# copy_model('path/to/source/directory/', 'path/to/destination/directory/')  # For a directory

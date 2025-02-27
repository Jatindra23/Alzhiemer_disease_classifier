import os 
from Alzimer_disease_classifier.logger import logging
from Alzimer_disease_classifier.exception import RenelException
import sys
from Alzimer_disease_classifier.entity.config_entity import DataIngestionConfig
from pathlib import Path
import shutil


class DataIngestion:
    def __init__(self, config):
        self.config = config

    def get_data_from_folder(self, data_path: Path):
        """
        Copies the entire directory structure (including subfolders like `mild` and `none`)
        from `data_path` to `config.root_dir`, preserving folder hierarchy.
        """
        logging.info(f"Copying data from {data_path} to {self.config.root_dir}")
        dest_dir = Path(self.config.root_dir)

        try:
            # Ensure source directory exists
            if not data_path.exists():
                raise RenelException(f"Source path {data_path} does not exist.", sys)
            
            # Create root destination directory
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Walk through the source directory
            for root, dirs, files in os.walk(data_path):
                # Get relative path from source to current directory
                relative_path = os.path.relpath(root, data_path)
                logging.info(f"Relative path: {relative_path}")
                dest_subdir = dest_dir / relative_path
                logging.info(f"Destination subdirectory: {dest_subdir}")

                # Create corresponding subdirectory in destination
                dest_subdir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory: {dest_subdir}")

                # Copy files to the destination subdirectory
                for file in files:
                    src_file = Path(root) / file
                    dest_file = dest_subdir / file

                    if not dest_file.exists():
                        shutil.copy(src_file, dest_subdir)
                        # logging.info(f"Copied: {src_file} → {dest_file}")
                    else:
                        logging.warning(f"Skipped (already exists): {dest_file}")

            logging.info(f"Data copied successfully to {dest_dir}")
            return str(dest_dir)
        
        except Exception as e:
            logging.error(f"Error: {e}")
            raise RenelException(e, sys)
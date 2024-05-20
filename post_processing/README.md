
# New MorphVQ Point-to-Point Correspondence Post-Processing

This repository contains a Python script for post-processing MorphVQ point-to-point correspondences.
The script reads and processes landmark files, VTK shapes, and JSONL correspondence files to recover estimated landmarks from point-to-point correspondences.

## Prerequisites

Ensure you have the following Python package installed:

- `tqdm`

You can install the `tqdm` package using pip:

```sh
pip install tqdm
```

## Usage

The preprocessing script `postprocessing.py` takes three arguments:

- `--landmark_dir`: The directory containing the ground-truth landmark files.
- `--vtk_shapes_dir`: The directory containing the ply shape files.
- `--jsonl_file_path`: The path to the JSONL file containing the correspondences saved after inference.

### Running the Script

To run the script, use the following command in your terminal:

```sh
python3 postprocessing.py --landmark_dir /path/to/landmark_dir --vtk_shapes_dir /path/to/vtk_shapes_dir --jsonl_file_path /path/to/jsonl_file
```

## Script Details

### `postprocessing.py`

This script performs the following steps:

1. **Read VTK Shapes**: Reads the VTK shape files from the specified directory.
2. **Read JSONL Correspondences**: Reads the JSONL correspondence file with a progress bar.
3. **Process Landmark Files**: Reads .fcsv landmark files from the specified directory.
4. **Update Landmarks and Correspondences**: Updates and filters the landmarks and correspondences based on the VTK shapes and the JSONL file.

### Function Definitions

- `read_ply_folder(vtk_shapes_dir)`: Reads the VTK shape files from the specified directory.
- `read_jsonl_to_dict_with_progress(jsonl_file_path)`: Reads the JSONL file and returns a dictionary of correspondences with a progress bar.
- `process_fcsv_folder(landmark_dir)`: Processes the landmark files from the specified directory.
- `update_landmarks_and_correspondences(vtk_shapes, correspondences, landmarks)`: Updates and filters the landmarks and correspondences.

## License

This project is licensed under the MIT License

## Acknowledgments

- [tqdm](https://github.com/tqdm/tqdm) for the progress bar functionality.
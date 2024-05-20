# PLYprep Slicer Class Implementation

This README file provides instructions for using the `PLYprep` class in the Slicer Interactor.
This code prepares PLY models for training given `volumes`, `labels`, and `.fcsv` ground-truth `landmark` files.
Samples of these files can be found in the `volume_and_labels`, and `landmark folders`.
This class ensuring that the volumes and labels are correctly scaled, and that the generated polygon meshes are cleaned,
appropriately re-meshed, and repaired.

## Prerequisites

Before running the code, ensure you have `pymeshfix` installed in the interactor using pip:

```bash
pip_install("pymeshfix")
```

## Usage

Once pymeshfix is installed, copy the code in the `model_exporter.py` file and paste it into the slicer python interactor.

### PLYprep Class

The PLYprep class processes sets of volume, segmentation, and landmark files. It applies a series of transformations, 
smoothing, and remeshing steps to prepare the PLY models.

### Initialization

Once the class is loaded into the python interactor creating a class with the folder path parameters 
(and, optionally the number of triangles desired) will begin the batch cleaning process.

```bash
prep = PLYprep(volume_label_folder, landmarks_folder, save_folder, num_triangles=24000)
```

- `volume_label_folder`: Path to the folder containing volume and label files.
- `landmarks_folder`: Path to the folder containing landmarks files.
- `save_folder`: Path to the folder where processed files will be saved.
- `num_triangles` (optional): Desired number of triangles for the remeshed model (default is 24000).

### Main Methods

- `process_file_sets`: Processes all valid file sets in the provided folders.
- `import_single_file_set`: Loads a single set of volume, label, and landmarks files.
- `correct_scale`: Applies scaling and transformation to the nodes.
- `applyIslandFilter`: Removes small islands or keeps the largest island in the segmentation.
- `applySmoothingClosingFillHoles`: Applies smoothing to fill holes in the segmentation.
- `applyWrapSolidifyToModel`: Applies the Wrap Solidify effect to the model.
- `flipModelFaces`: Flips the faces of the model to correct orientation.
- `cleanAndRemeshModel`: Cleans and remeshes the model to the desired number of triangles.
- `repairModelNodeWithPyMeshFix`: Repairs the model using PyMeshFix to ensure it is manifold.
- `exportNodes`: Exports the processed nodes to the specified directory.

## Example Usage
```bash
volume_label_folder = "/path/to/volume_label_folder"
landmarks_folder = "/path/to/landmarks_folder"
save_folder = "/path/to/save_folder"
num_triangles = 24000

prep = PLYprep(volume_label_folder, landmarks_folder, save_folder, num_triangles)
prep.process_file_sets()
```

This will process all valid file sets in the specified folders, apply the necessary transformations, 
and save the processed files to the `save_folder`.

## Output
Processed models, segmentation, and landmarks will be saved in subdirectories within the save_folder, named after their identifiers.
A CSV file `mesh_info_summary.csv` will be generated in the `save_folder`, summarizing the properties of each processed mesh.


## Additional Information
Ensure that the Slicer application is running, and the Slicer Interactor is active before executing the code.

For any issues or questions, refer to the Slicer documentation or seek help from the Slicer community.
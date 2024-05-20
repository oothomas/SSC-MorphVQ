import os
import slicer
from pathlib import Path
from vtk import vtkMatrix4x4
import numpy as np
import logging
import pandas as pd

import vtk
import vtkmodules.all as vtk

from vtkmodules.util import numpy_support as vtk_np
import pymeshfix as mf

import SurfaceToolbox
from slicer.ScriptedLoadableModule import *


class PLYprep:
    def __init__(self, volume_label_folder, landmarks_folder, save_folder, num_triangles=24000):
        """
        Initialize the SlicerImporter with paths to the volume/label folder and the landmarks folder.

        Args:
        - volume_label_folder: Path to the folder containing volume and label files.
        - landmarks_folder: Path to the folder containing landmarks files.
        """
        self.nodes = None
        self.cleaner = None
        self.num_triangles = num_triangles
        self.volume_label_folder = volume_label_folder
        self.landmarks_folder = landmarks_folder
        self.save_folder = save_folder
        # Initialize an empty DataFrame with the desired columns
        self.mesh_info_df = pd.DataFrame(
            columns=['Identifier', 'NumVertices', 'NumTriangles', 'IsManifold', 'IsWatertight'])

        self.valid_identifiers, self.valid_file_sets = self.find_valid_identifiers()

    def process_file_sets(self):
        for key, value in self.valid_file_sets.items():
            self.nodes = self.import_single_file_set(key, value)
            self.correct_scale(self.nodes)
            self.applyIslandFilter(self.nodes['segmentation'])
            self.applySmoothingClosingFillHoles(self.nodes['segmentation'], kernelSize=0.15)
            self.nodes['initial_model'] = self.applyWrapSolidifyToModel(self.nodes['segmentation'])
            self.flipModelFaces(self.nodes['initial_model'])
            self.nodes['model_12K'] = self.cleanAndRemeshModel(self.nodes['initial_model'], self.num_triangles)
            self.repairModelNodeWithPyMeshFix(self.nodes['model_12K'])

            self.exportNodes(self.nodes['model_12K'], self.nodes['landmarks'],
                             self.nodes['segmentation'], self.nodes['volume'],
                             str(key), self.save_folder + str(key) + '/')

            polydata = self.nodes['model_12K'].GetPolyData()
            num_vertices, num_triangles, is_manifold, is_watertight = self.analyze_mesh_properties(polydata)
            print('Identifier:', key, 'num_vertices', num_vertices, 'num_triangles:', num_triangles,
                  'is_manifold:', is_manifold, 'is_watertight:', is_watertight)

            # Add the results to the DataFrame
            # Prepare a new row as a DataFrame
            new_row = pd.DataFrame({
                'Identifier': [key],
                'NumVertices': [num_vertices],
                'NumTriangles': [num_triangles],
                'IsManifold': [is_manifold],
                'IsWatertight': [is_watertight]
            })

            # Use pandas.concat to add the new row
            self.mesh_info_df = pd.concat([self.mesh_info_df, new_row], ignore_index=True)

            # Continue processing as before...
            slicer.mrmlScene.Clear(0)

        # Optionally, save the DataFrame to a CSV file after processing all file sets
        self.mesh_info_df.to_csv(self.save_folder + 'mesh_info_summary.csv', index=False)

        return None

    @staticmethod
    def find_ply_file_stems(directory):
        """
        Searches for all .ply files within a given directory that match the pattern *_*_Model.ply,
        and returns the stems of these filenames excluding the '_Model.ply' part.

        :param directory: The directory to search in.
        :return: A list of stems for all matching .ply files.
        """
        directory_path = Path(directory)
        matching_stems = []
        for file_path in directory_path.glob("*_*_Model.ply"):
            stem_parts = file_path.stem.split('_')[:-1]  # This removes the '_Model' part
            stem_without_model = '_'.join(stem_parts)
            matching_stems.append(stem_without_model)
        return matching_stems

    def remove_keys_by_stems(self, dictionary, directory):
        """
        Removes keys from the dictionary that match any of the stems found in .ply files within the given directory.

        :param dictionary: The dictionary from which to remove keys.
        :param directory: The directory to search for .ply files to determine the keys to remove.
        :return: The modified dictionary with keys removed.
        """
        stems = self.find_ply_file_stems(directory)
        # Iterate over a copy of the dictionary's keys list to avoid RuntimeError due to changing dict size during iteration
        for key in list(dictionary.keys()):
            if key in stems:
                del dictionary[key]
        return dictionary

    def correct_scale(self, node_dict):

        RAS_transform = np.array([[-1.00, 0, 0, 0],
                                  [0, -1.00, 0, 0],
                                  [0, 0, 1.000, 0],
                                  [0, 0, 0, 1.000]])

        scale_transform = np.array([[0.036, 0, 0, 0],
                                    [0, 0.036, 0, 0],
                                    [0, 0, 0.036, 0],
                                    [0, 0, 0, 1.000]])

        self.apply_and_harden_transform_with_array(node_dict['volume'], scale_transform)
        self.apply_and_harden_transform_with_array(node_dict['segmentation'], scale_transform)

        self.apply_and_harden_transform_with_array(node_dict['landmarks'], RAS_transform)
        self.apply_and_harden_transform_with_array(node_dict['landmarks'], scale_transform)

    @staticmethod
    def load_volume_and_segmentation(volume_path, label_path):
        """
        Load a volume and its corresponding label file as a segmentation into 3D Slicer and return the nodes.

        Args:
        - volume_path: Path to the volume file.
        - label_path: Path to the label file.

        Returns:
        - A tuple of (volume_node, segmentation_node).
        """
        # Load the volume
        volume_node = slicer.util.loadVolume(volume_path, returnNode=True)[1]

        # Load the label map as a segmentation
        labelmapVolumeNode = slicer.util.loadLabelVolume(label_path, returnNode=True)[1]

        # Create a new segmentation node
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.CreateDefaultDisplayNodes()  # for visualization
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)

        # Import the label map into the segmentation node
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, segmentationNode)

        # Optionally, remove the label map volume from the scene if it's no longer needed
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

        return volume_node, segmentationNode

    @staticmethod
    def load_landmarks(landmark_path):
        """
        Load a landmarks file into 3D Slicer and return the node.

        Args:
        - landmark_path: Path to the landmarks file.

        Returns:
        - landmarks_node: The loaded landmarks node.
        """
        landmarks_node = slicer.util.loadMarkupsFiducialList(landmark_path, returnNode=True)
        return landmarks_node

    def import_single_file_set(self, identifier, file_set):
        """
        Load a single file set (volume, label, and landmarks) based on the given identifier.

        Args:
        - identifier: A unique identifier for the file set (e.g., '152-1' for files like '152-1_smooth.nii.gz').

        Returns:
        - A dictionary with 'volume', 'label', and 'landmarks' nodes if available; None otherwise.
        """
        # Construct file paths based on the identifier
        volume_path = file_set['volume']
        label_path = file_set['segmentation']
        landmark_path = file_set['landmarks']

        # Initialize an empty dict to store the loaded nodes
        nodes = {}

        # Check if files exist and load them
        if os.path.exists(volume_path) and os.path.exists(label_path):
            nodes['volume'], nodes['segmentation'] = self.load_volume_and_segmentation(volume_path, label_path)
            nodes['landmarks'] = self.load_landmarks(landmark_path)

        if nodes:
            return nodes
        else:
            print(f"No files found for identifier: {identifier}")
            return None

    def find_valid_identifiers(self):
        """
        Check the volume/label and landmarks folders for valid identifiers, considering both underscores and hyphens,
        and accounting for the provided folder structure where each case has its own subfolder.

        Returns:
        - A tuple containing:
            - A list of unique identifiers for which corresponding volume, label, and landmark files exist.
            - A dictionary mapping each valid identifier to a dictionary of its file paths for volume, label, and landmark.
        """

        file_sets = {}
        print("Checking volume and label files across case folders...")

        # Loop through each case folder within the volume_label_folder to find matching pairs
        for case_folder_name in os.listdir(self.volume_label_folder):
            case_folder_path = os.path.join(self.volume_label_folder, case_folder_name)
            if os.path.isdir(case_folder_path):  # Ensure it's a folder
                for file_name in os.listdir(case_folder_path):
                    if file_name.endswith("__rec-subvolume-scale_2.nii.gz"):
                        identifier = file_name.split('__rec-subvolume-scale_2.nii.gz')[0]
                        volume_path = os.path.join(case_folder_path, file_name)
                        label_file_name = f"{identifier}_smooth.nii.gz"
                        label_path = os.path.join(case_folder_path, label_file_name)

                        # Check for the existence of the label file
                        if os.path.exists(label_path):
                            # Initialize the file set for this identifier
                            file_sets[identifier] = {'volume': volume_path, 'segmentation': label_path}
                    if file_name.endswith("_rec-subvolume-scale_2.nii.gz"):
                        identifier = file_name.split('_rec-subvolume-scale_2.nii.gz')[0]
                        volume_path = os.path.join(case_folder_path, file_name)
                        label_file_name = f"{identifier}_smooth.nii.gz"
                        label_path = os.path.join(case_folder_path, label_file_name)

                        # Check for the existence of the label file
                        if os.path.exists(label_path):
                            # Initialize the file set for this identifier
                            file_sets[identifier] = {'volume': volume_path, 'segmentation': label_path}

        # Loop through landmarks folder to find existing landmarks
        print("Checking landmark files...")
        for file_name in os.listdir(self.landmarks_folder):
            if file_name.endswith(".fcsv"):
                identifier = file_name.replace("-", "_").replace(".fcsv", "")
                landmark_path = os.path.join(self.landmarks_folder, file_name)
                # Add the landmark path to the corresponding file set
                if identifier in file_sets:
                    file_sets[identifier]['landmarks'] = landmark_path
                    # print(f"Found landmark for identifier: {identifier}")

        # Filter to keep only complete sets with volume, label, and landmarks
        valid_identifiers = [identifier for identifier, paths in file_sets.items() if 'landmarks' in paths]
        valid_file_sets = {identifier: paths for identifier, paths in file_sets.items() if
                           identifier in valid_identifiers}

        print("num_valid_identifiers:", len(valid_identifiers))

        return valid_identifiers, valid_file_sets

    @staticmethod
    def apply_and_harden_transform_with_array(target_node, transform_array):
        """
        Applies a transformation, defined by a NumPy array, to a target node (volume or landmark) in 3D Slicer, hardens it,
        and then deletes the transform node.

        Args:
        - target_node: The node to which the transform will be applied. This can be a volume or a markup (landmark) node.
        - transform_array: A NumPy array defining the transformation matrix. Can be 3x3 (rotation/scaling) or 4x4 (full transform).

        Returns:
        - None. The transformation is applied directly to the target node, and the transform node is deleted.
        """
        if not target_node or transform_array is None:
            print("Error: Target node or transform array is missing.")
            return

        # Embed 3x3 matrix into a 4x4 matrix if necessary
        if transform_array.shape == (3, 3):
            # Create a 4x4 matrix with the 3x3 rotation/scaling and a [0, 0, 0, 1] row for homogeneous coordinates
            full_transform_array = np.eye(4)
            full_transform_array[:3, :3] = transform_array
        elif transform_array.shape == (4, 4):
            full_transform_array = transform_array
        else:
            print("Error: Transform array must be a 3x3 or 4x4 matrix.")
            return

        # Create a new vtkMatrix4x4 and set its elements from the full_transform_array
        transform_matrix = vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                transform_matrix.SetElement(i, j, full_transform_array[i, j])

        # Create a new transform node
        transform_node = slicer.vtkMRMLLinearTransformNode()
        slicer.mrmlScene.AddNode(transform_node)
        transform_node.SetMatrixTransformToParent(transform_matrix)

        # Apply the transform to the target node
        target_node.SetAndObserveTransformNodeID(transform_node.GetID())

        # Harden (permanently apply) the transform
        slicer.vtkSlicerTransformLogic().hardenTransform(target_node)

        # Delete the transform node
        slicer.mrmlScene.RemoveNode(transform_node)

        print(f"Transform applied and hardened for node: {target_node.GetName()}. Transform node deleted.")

    @staticmethod
    def save_nodes_to_folder(directory, node_dict, identifier):
        """
        Saves nodes specified in a dictionary to a new subdirectory within the given directory. The subdirectory is named
        after the provided identifier, and files are saved with filenames based on the identifier.

        Args:
        - directory: Directory where the new subdirectory will be created and files saved.
        - node_dict: Dictionary containing keys 'volume', 'label', and 'landmarks' with corresponding Slicer nodes.
        - identifier: Identifier to be used as the name for the new subdirectory and as the base name for saved files.

        Returns:
        - A dictionary with the paths of the saved files.
        """

        # Create a new subdirectory named after the identifier
        new_directory = os.path.join(directory, identifier)
        if not os.path.isdir(new_directory):
            os.makedirs(new_directory)

        saved_paths = {}

        # Define the file extensions based on node type
        file_extensions = {
            "volume": "_Volume.nrrd",
            "segmentation": "_Segmentation.nrrd",
            "landmarks": "_Landmarks.fcsv"
        }

        # Iterate over the node dictionary and save each node
        for node_type, node in node_dict.items():
            if node_type in file_extensions:
                file_name = f"{identifier}{file_extensions[node_type]}"
                file_path = os.path.join(new_directory, file_name)
                slicer.util.saveNode(node, file_path)
                saved_paths[node_type] = file_path
                print(f"Saved {node_type} to {file_path}")

        return saved_paths

    @staticmethod
    def applyIslandFilter(segmentationNode, operation="REMOVE_SMALL_ISLANDS", minSize=1000):
        """
        Applies the Island operation to a segmentation node in 3D Slicer.

        Args:
        - segmentationNode (vtkMRMLSegmentationNode): The segmentation node to apply the filter on.
        - operation (str): The operation to perform. Options include "REMOVE_SMALL_ISLANDS" and "KEEP_LARGEST_ISLAND".
        - minSize (int): The minimum size threshold for islands when removing small islands.
        """
        # Ensure a valid segmentation node is provided
        if segmentationNode is None:
            raise ValueError("Invalid segmentation node provided")

        # Create a new segment editor to perform operations
        segmentEditor = slicer.qMRMLSegmentEditorWidget()
        segmentEditor.setMRMLScene(slicer.mrmlScene)

        # Temporary segment editor node to store parameters
        segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
        slicer.mrmlScene.AddNode(segmentEditorNode)
        segmentEditor.setMRMLSegmentEditorNode(segmentEditorNode)

        # Set the segmentation node in the segment editor
        segmentEditor.setSegmentationNode(segmentationNode)

        # Select the islands effect
        segmentEditor.setActiveEffectByName("Islands")
        effect = segmentEditor.activeEffect()

        # Configure the effect parameters
        if effect:
            effect.setParameter("Operation", operation)
            if operation == "REMOVE_SMALL_ISLANDS":
                effect.setParameter("MinimumSize", str(minSize))

            # Apply the effect
            effect.self().onApply()

        # Cleanup
        segmentEditor.setActiveEffectByName("")  # Deselect the active effect
        slicer.mrmlScene.RemoveNode(segmentEditorNode)

    @staticmethod
    def applySmoothingClosingFillHoles(segmentationNode, kernelSize=0.5):
        """
        Apply 'Closing (Fill Holes)' smoothing method to a segmentation node with a specified kernel size.

        Args:
        - segmentationNode (slicer.vtkMRMLSegmentationNode): The segmentation node to apply the smoothing on.
        - kernelSize (int): The kernel size for the 'Closing (Fill Holes)' method.
        """
        if not segmentationNode:
            print("Invalid segmentation node provided.")
            return

        # Create an instance of the Segment Editor
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
        segmentEditorWidget.setSegmentationNode(segmentationNode)

        # Select the 'Smoothing' effect
        segmentEditorWidget.setActiveEffectByName("Smoothing")
        effect = segmentEditorWidget.activeEffect()

        # Set parameters for 'Closing (Fill Holes)' smoothing
        effect.setParameter("SmoothingMethod", "CLOSING")
        effect.setParameter("KernelSizeMm", kernelSize)

        # Apply the effect to all segments
        segmentation = segmentationNode.GetSegmentation()
        segmentIDs = vtk.vtkStringArray()
        segmentation.GetSegmentIDs(segmentIDs)
        for i in range(segmentIDs.GetNumberOfValues()):
            segmentID = segmentIDs.GetValue(i)
            segmentEditorWidget.setCurrentSegmentID(segmentID)
            effect.self().onApply()

        # Cleanup
        slicer.mrmlScene.RemoveNode(segmentEditorNode)
        segmentEditorWidget = None

        print("Smoothing 'Closing (Fill Holes)' applied successfully.")

    @staticmethod
    def applyWrapSolidifyToModel(segmentationNode):
        # Ensure the input is valid
        if not segmentationNode:
            print("Invalid input segmentation node.")
            return None

        # Access the Wrap Solidify effect from the Segment Editor
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
        segmentEditorWidget.setSegmentationNode(segmentationNode)

        # Select the Wrap Solidify effect
        segmentEditorWidget.setActiveEffectByName("Wrap Solidify")
        effect = segmentEditorWidget.activeEffect()

        # Configure the effect parameters for Outer Surface
        effect.setParameter("region", "outerSurface")
        # Configure the effect to output a model
        effect.setParameter("outputType", "model")

        # Apply the effect to create a model
        effect.self().onApply()

        # Attempt to find the newly created model node in the scene
        outputModelNode = None
        for nodeIndex in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLModelNode")):
            node = slicer.mrmlScene.GetNthNodeByClass(nodeIndex, "vtkMRMLModelNode")
            if node.GetName().startswith("1"):  # Assuming the output model has a specific naming convention
                outputModelNode = node
                break

        # Clean up
        slicer.mrmlScene.RemoveNode(segmentEditorNode)
        segmentEditorWidget = None

        if outputModelNode:
            print("Wrap Solidify 'Outer Surface' applied. Model outputted.")
            return outputModelNode
        else:
            print("Failed to find the generated model from Wrap Solidify.")
            return None

    @staticmethod
    def flipModelFaces(modelNode):
        """
        Flips the faces of a model node by reversing the cell ordering.

        Args:
            modelNode (vtkMRMLModelNode): The model node to process.
        """
        if not modelNode:
            raise ValueError("Model node is None")

        polyData = modelNode.GetPolyData()

        if not polyData:
            raise ValueError("Model node does not contain polydata")

        if polyData.GetNumberOfCells() == 0:
            raise ValueError("The model node does not contain any cells")

        # Debug: Print information about the polydata
        print(f"Number of points: {polyData.GetNumberOfPoints()}")
        print(f"Number of cells: {polyData.GetNumberOfCells()}")

        # Reverse the order of points in each cell (triangle) to flip normals
        for i in range(polyData.GetNumberOfCells()):
            cellPoints = polyData.GetCell(i).GetPointIds()
            id0 = cellPoints.GetId(0)
            id2 = cellPoints.GetId(2)
            # Swap the first and last points of the cell to reverse its orientation
            cellPoints.SetId(0, id2)
            cellPoints.SetId(2, id0)

        polyData.Modified()

        # Update the model node to refresh the display
        modelNode.SetAndObservePolyData(polyData)
        modelNode.Modified()

        # Optionally, you can update normals if required
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polyData)
        normals.SetFlipNormals(True)
        normals.Update()

        modelNode.GetPolyData().SetPoints(normals.GetOutput().GetPoints())
        modelNode.GetPolyData().SetPolys(normals.GetOutput().GetPolys())
        modelNode.GetPolyData().GetPointData().SetNormals(normals.GetOutput().GetPointData().GetNormals())
        modelNode.GetPolyData().Modified()

    @staticmethod
    def cleanAndRemeshModel(inputModelNode, desiredNumberOfTriangles):
        """
        Cleans the input model node and applies remeshing to achieve a specific number of triangles.
        Automatically creates and returns an output model node based on the input model node.

        Args:
            inputModelNode (vtkMRMLModelNode): The input model node to process.
            desiredNumberOfTriangles (int): The desired number of triangles in the remeshed model.

        Returns:
            vtkMRMLModelNode: The output model node containing the cleaned and remeshed model.
        """
        if not inputModelNode:
            raise ValueError("An input model node must be provided")

        # Ensure the pyacvd package is installed for remeshing
        surfaceToolboxLogic = SurfaceToolbox.SurfaceToolboxLogic()
        if not surfaceToolboxLogic.installRemeshPrerequisites(force=True):
            raise ImportError("Required 'pyacvd' package could not be installed.")

        # Create a new model node for the output
        outputModelNode = slicer.vtkMRMLModelNode()
        outputModelNode.SetName(inputModelNode.GetName() + "_cleanedRemeshed")
        slicer.mrmlScene.AddNode(outputModelNode)

        # Copy the display properties from the input model to the output model
        outputModelDisplay = slicer.vtkMRMLModelDisplayNode()
        slicer.mrmlScene.AddNode(outputModelDisplay)
        outputModelNode.SetAndObserveDisplayNodeID(outputModelDisplay.GetID())
        if inputModelNode.GetDisplayNode():
            outputModelDisplay.Copy(inputModelNode.GetDisplayNode())

        # Clean the input model
        logging.info("Cleaning the model...")
        surfaceToolboxLogic.clean(inputModelNode, outputModelNode)

        # Calculate the clusters value for the desired number of triangles.
        # The number of clusters (points) in pyacvd is roughly half the number of triangles desired.
        clusters = desiredNumberOfTriangles // 2

        # Apply remeshing
        logging.info("Remeshing the model to approximately {} triangles...".format(desiredNumberOfTriangles))
        surfaceToolboxLogic.remesh(outputModelNode, outputModelNode, clusters=clusters)

        print("Model cleaning and remeshing completed.")

        return outputModelNode

    @staticmethod
    def flip_triangle_faces(polydata):
        """
        Flips the faces of the triangles in a vtkPolyData object.

        :param polydata: vtkPolyData object representing the surface mesh.
        :return: vtkPolyData object with flipped faces.
        """
        # Ensure input is a vtkPolyData object
        if not isinstance(polydata, vtk.vtkPolyData):
            raise TypeError("Input must be a vtkPolyData object.")

        # Create an array to store the flipped polygons
        flippedPolygons = vtk.vtkCellArray()

        # Iterate through each cell in the polydata
        for i in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(i)
            # Process only if the cell is a triangle
            if cell.GetCellType() == vtk.VTK_TRIANGLE:
                pts = cell.GetPointIds()
                # Flip the order of the vertices
                flippedPolygons.InsertNextCell(3)
                flippedPolygons.InsertCellPoint(pts.GetId(2))
                flippedPolygons.InsertCellPoint(pts.GetId(1))
                flippedPolygons.InsertCellPoint(pts.GetId(0))

        # Create a new vtkPolyData object to hold the result
        flippedPolyData = vtk.vtkPolyData()
        flippedPolyData.SetPoints(polydata.GetPoints())
        flippedPolyData.SetPolys(flippedPolygons)

        # Copy the vertex normals if they exist
        if polydata.GetPointData().GetNormals():
            flippedPolyData.GetPointData().SetNormals(polydata.GetPointData().GetNormals())

        # Optionally, recompute the normals to reflect the flipped faces
        normalGenerator = vtk.vtkPolyDataNormals()
        normalGenerator.SetInputData(flippedPolyData)
        normalGenerator.ComputePointNormalsOn()
        normalGenerator.ComputeCellNormalsOn()
        normalGenerator.Update()

        # Return the polydata with flipped faces and recomputed normals
        return normalGenerator.GetOutput()

    def repairModelNodeWithPyMeshFix(self, modelNode):
        """
        Repairs a mesh model node using PyMeshFix to make it manifold.

        Args:
            modelNode (vtkMRMLModelNode): The model node to be repaired.
        """
        if not modelNode or not modelNode.GetPolyData():
            raise ValueError("Valid model node with polydata is required")

        polyData = modelNode.GetPolyData()

        # Extract vertices using vtk_to_numpy
        vertices = vtk_np.vtk_to_numpy(polyData.GetPoints().GetData())

        faces = self.extract_faces_from_polydata(polyData)

        # Repair the mesh using PyMeshFix
        meshfix = mf.MeshFix(vertices, faces)

        meshfix.repair(verbose=True, remove_smallest_components=False, joincomp=True)

        # Get the repaired mesh
        vertices_fixed, faces_fixed = meshfix.points(), meshfix.faces()

        # Create a new vtkPolyData for the repaired mesh
        repairedPolyData = vtk.vtkPolyData()

        # Set vertices
        points = vtk.vtkPoints()
        for v in vertices_fixed:
            points.InsertNextPoint(v.tolist())
        repairedPolyData.SetPoints(points)

        # Set faces
        polys = vtk.vtkCellArray()
        for f in faces_fixed:
            polys.InsertNextCell(3, f.astype(np.int64))
        repairedPolyData.SetPolys(polys)

        # Update the model node
        modelNode.SetAndObservePolyData(repairedPolyData)

        print("Model has been repaired and updated.")

    @staticmethod
    def analyze_mesh_properties(polydata):
        num_vertices = polydata.GetNumberOfPoints()
        num_triangles = polydata.GetNumberOfCells()
        edge_extractor = vtk.vtkExtractEdges()
        edge_extractor.SetInputData(polydata)
        edge_extractor.Update()
        num_edges = edge_extractor.GetOutput().GetNumberOfLines()

        feature_edges = vtk.vtkFeatureEdges()
        feature_edges.SetInputData(polydata)
        feature_edges.BoundaryEdgesOn()
        feature_edges.FeatureEdgesOff()
        feature_edges.ManifoldEdgesOff()
        feature_edges.NonManifoldEdgesOn()
        feature_edges.Update()
        non_manifold_edges = feature_edges.GetOutput().GetNumberOfCells()

        feature_edges.BoundaryEdgesOn()
        feature_edges.ManifoldEdgesOff()
        feature_edges.NonManifoldEdgesOff()
        feature_edges.Update()
        boundary_edges = feature_edges.GetOutput().GetNumberOfCells()

        is_manifold = non_manifold_edges == 0
        is_watertight = boundary_edges == 0

        return num_vertices, num_triangles, is_manifold, is_watertight

    @staticmethod
    def extract_faces_from_polydata(polyData):
        """
        Extracts an n x 3 face array from vtkPolyData. Assumes the input polyData
        consists entirely of triangles.

        Args:
            polyData (vtk.vtkPolyData): The input polyData from which to extract faces.

        Returns:
            numpy.ndarray: An array of shape (n, 3) containing face indices.
        """
        # Ensure the mesh is triangulated
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(polyData)
        triangleFilter.PassVertsOff()
        triangleFilter.PassLinesOff()
        triangleFilter.Update()

        # Extract the triangulated mesh
        triangulatedPolyData = triangleFilter.GetOutput()

        # Extract face information
        # vtkPolyData stores faces as a series of integers:
        # [n, i1, i2, i3, n, i4, i5, i6, ...] where n is the number of points per face (always 3 here)
        # and i1, i2, i3 are the indices of the vertices in this face.
        faces_vtk = triangulatedPolyData.GetPolys().GetData()
        faces_flat = vtk_np.vtk_to_numpy(faces_vtk)

        # The array includes a leading count of vertices for each face (always 3 for triangles),
        # so we reshape the array to skip these counts.
        faces = faces_flat.reshape(-1, 4)[:, 1:4]

        return faces

    @staticmethod
    def exportNodes(modelNode, landmarkNode, segmentationNode, volumeNode, identifier, directoryPath):
        """
        Exports a model node, markups landmark node, segmentation node, and volume node to a specified directory.

        Args:
            modelNode (vtkMRMLModelNode): The model node to export.
            landmarkNode (vtkMRMLMarkupsFiducialNode): The markups landmark node to export.
            segmentationNode (vtkMRMLSegmentationNode): The segmentation node to export.
            volumeNode (vtkMRMLScalarVolumeNode): The volume node to export.
            identifier (str): An identifier to include in the filenames.
            directoryPath (str): The path to the directory where the files will be saved.
        """
        # Ensure the directory exists
        if not os.path.exists(directoryPath):
            os.makedirs(directoryPath)

        # Define file paths
        modelFilePath = os.path.join(directoryPath, f"{identifier}_Model.ply")  # Model saved as .ply
        landmarkFilePath = os.path.join(directoryPath, f"{identifier}_Landmarks.fcsv")
        segmentationFilePath = os.path.join(directoryPath, f"{identifier}_Segmentation.seg.nrrd")
        volumeFilePath = os.path.join(directoryPath, f"{identifier}_Volume.nrrd")

        # Export the model node
        slicer.util.saveNode(modelNode, modelFilePath)

        # Export the landmark node
        slicer.util.saveNode(landmarkNode, landmarkFilePath)

        # Export the segmentation node
        slicer.util.saveNode(segmentationNode, segmentationFilePath)

        # Export the volume node
        slicer.util.saveNode(volumeNode, volumeFilePath)

        print(f"All nodes have been exported to {directoryPath}")

import pandas as pd
import vtk
import json

from tqdm import tqdm
from Tools.utils import *
import numpy as np


def read_ply_folder(folder_path):
    """
    Reads a folder containing PLY files and returns a dictionary of VTK PolyData meshes.
    The keys of the dictionary are the filenames without the .ply extension.

    Parameters:
    folder_path (str): The path to the folder containing PLY files.

    Returns:
    dict: A dictionary with filenames as keys and VTK PolyData meshes as values.
    """
    ply_meshes = {}

    # Iterate over each file in the folder
    for file in os.listdir(folder_path):
        if file.endswith('.ply'):
            file_path = os.path.join(folder_path, file)
            reader = vtk.vtkPLYReader()
            reader.SetFileName(file_path)
            reader.Update()

            # Extract the file name without the .ply extension
            file_key = os.path.splitext(file)[0]

            # Store the VTK PolyData in the dictionary
            ply_meshes[file_key] = reader.GetOutput()

    return ply_meshes


def find_closest_vertices_and_coordinates(mesh, landmarks):
    """
    For each landmark, find the closest vertex on the mesh and return both the index and coordinates.

    Parameters:
    mesh (vtkPolyData): The VTK PolyData mesh.
    landmarks (numpy array): Nx3 numpy array of landmark positions.

    Returns:
    list: A list of tuples, each containing the index and coordinates of the closest vertex on the mesh for each landmark.
    """
    # Create a point locator
    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(mesh)
    point_locator.BuildLocator()

    closest_vertices = []
    closest_vertices_ix = []
    for landmark in landmarks:
        # Find the closest point on the mesh
        closest_point_id = point_locator.FindClosestPoint(landmark)
        closest_point_coords = mesh.GetPoint(closest_point_id)
        closest_vertices.append(closest_point_coords)
        closest_vertices_ix.append(closest_point_id)

    return closest_vertices_ix, np.array(closest_vertices)


def get_vertex_coordinates(mesh, vertex_indices):
    """
    Given a VTK mesh and a list of vertex indices, return the coordinates of these vertices.

    Parameters:
    mesh (vtkPolyData): The VTK PolyData mesh.
    vertex_indices (list of int): List of indices of vertices.

    Returns:
    numpy array: Array of coordinates of the specified vertices.
    """
    coordinates = []
    for idx in vertex_indices:
        point = mesh.GetPoint(idx)
        coordinates.append(point)

    return np.array(coordinates)


def read_jsonl_to_dict_with_progress(jsonl_file_path):
    """
    Reads a JSON lines file with a progress bar, processing each batch as a dictionary.

    Parameters:
    jsonl_file_path (str): Path to the JSON lines file.

    Returns:
    dict: Dictionary with keys formatted as 'shape1_name' + '_' + 'shape2_name' and
          values as dictionaries of batch data.
    """
    master_dict = {}

    # First, count the total number of lines to set the progress bar's total
    total_lines = sum(1 for _ in open(jsonl_file_path, 'r'))

    with open(jsonl_file_path, 'r') as file:
        # Wrap the file object with tqdm for a progress bar
        for line in tqdm(file, total=total_lines, desc="Reading batches"):
            batch_dict = json.loads(line)
            key = f"{batch_dict['shape1_name']}_{batch_dict['shape2_name']}"
            master_dict[key] = batch_dict

    return master_dict


def read_fcsv(file_path):
    """
    Reads an fcsv file and extracts the ID and coordinates (x, y, z).

    Parameters:
    file_path (str): The path to the fcsv file.

    Returns:
    tuple: A tuple containing a list of IDs and a numpy array of coordinates.
    """
    # Read the fcsv file, skipping the initial lines that don't contain data points
    data = pd.read_csv(file_path, comment='#', header=None)

    # Extract the IDs and the coordinates
    ids = data.iloc[:, 0].tolist()
    ids = [id.split('l')[1] for id in ids]
    coordinates = data.iloc[:, 1:4].to_numpy()

    return ids, coordinates


def process_fcsv_folder(folder_path):
    """
    Processes a folder containing fcsv files and returns a dictionary with filenames as keys.
    Each key maps to an internal dictionary containing the list of ids and numpy arrays of coordinates.

    Parameters:
    folder_path (str): The path to the folder containing fcsv files.

    Returns:
    dict: A dictionary with file names as keys and values as internal dictionaries of ids and coordinates.
    """
    result = {}

    # Iterate over each file in the folder
    for file in os.listdir(folder_path):
        if file.endswith('.fcsv'):
            specimen_name = file.split('.')[0].replace('-', '_')
            file_path = os.path.join(folder_path, file)
            ids, coordinates = read_fcsv(file_path)
            result[specimen_name] = {"ids": ids, "coordinates": coordinates, 'est.coordinates': [], 'est.ids': []}

    return result


def update_landmarks_and_correspondences(vtk_shapes, correspondences, landmarks):
    """
    Updates landmarks with estimated coordinates and filters correspondences for missing keys.

    Parameters:
    vtk_shapes (dict): Dictionary of VTK shapes.
    correspondences (dict): Dictionary of correspondences between shapes.
    landmarks (dict): Dictionary of landmarks.

    Returns:
    tuple: Updated landmarks dictionary and filtered correspondences dictionary.
    """
    updated_landmarks = landmarks.copy()
    filtered_correspondences = {}
    for key, value in correspondences.items():
        source_name, target_name = str(value['shape1_name']), str(value['shape2_name'])
        try:
            # Find the closest vertices and update landmarks
            # T12
            target_closest_vertices_ix, _ = find_closest_vertices_and_coordinates(
                vtk_shapes[target_name], updated_landmarks[target_name[:-5] + 'Landmarks']['coordinates']
            )
            source_corr_vertices_ix = np.array(value['T12_new'], dtype=int)[target_closest_vertices_ix]
            source_corr_vertices = get_vertex_coordinates(vtk_shapes[source_name], source_corr_vertices_ix)

            updated_landmarks[source_name[:-5] + 'Landmarks']['est.coordinates'].append(source_corr_vertices)
            value['T12_source_corr_vertices_ix'] = source_corr_vertices_ix
            updated_landmarks[source_name[:-5] + 'Landmarks']['est.ids'].append(target_name[:-5] + 'Landmarks')

            # Add to filtered correspondences
            filtered_correspondences[key] = value
        except Exception as e:
            print(f"Exception encountered: {e}. Skipping key: {key}.")

    return updated_landmarks, filtered_correspondences

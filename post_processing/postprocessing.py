import argparse
from .utils import (read_ply_folder, read_jsonl_to_dict_with_progress, process_fcsv_folder,
                    update_landmarks_and_correspondences)


def main(landmark_dir, vtk_shapes_dir, jsonl_file_path):
    # Import and Process Data from Full Analysis
    vtk_shapes = read_ply_folder(vtk_shapes_dir)
    correspondences = read_jsonl_to_dict_with_progress(jsonl_file_path)
    landmarks = process_fcsv_folder(landmark_dir)

    updated_landmarks, filtered_correspondences = update_landmarks_and_correspondences(vtk_shapes,
                                                                                       correspondences,
                                                                                       landmarks)
    # Further processing or saving results

    print("Landmarks and correspondences updated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Mouse Mandible Data.")
    parser.add_argument('--landmark_dir', type=str, required=True, help="Directory containing landmark files")
    parser.add_argument('--vtk_shapes_dir', type=str, required=True, help="Directory containing VTK shapes")
    parser.add_argument('--jsonl_file_path', type=str, required=True, help="Path to JSONL file with correspondences")

    args = parser.parse_args()
    main(args.landmark_dir, args.vtk_shapes_dir, args.jsonl_file_path)
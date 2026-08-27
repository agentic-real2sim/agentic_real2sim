# Process the mask data to filter out the outliers and generate the processed masks

import numpy as np
import open3d as o3d
import json
from tqdm import tqdm
import os
import glob
import cv2
import pickle
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--controller_name", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
CONTROLLER_NAME = args.controller_name

processed_masks = {}


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def read_mask(mask_path):
    # Convert the white mask into binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask


def process_pcd_mask(frame_idx, pcd_path, mask_path, mask_info, num_cam):
    global processed_masks
    processed_masks[frame_idx] = {}

    # Load the pcd data
    data = np.load(f"{pcd_path}/{frame_idx}.npz")
    points = data["points"]
    colors = data["colors"]
    masks = data["masks"]

    object_pcd = o3d.geometry.PointCloud()
    controller_pcd = o3d.geometry.PointCloud()

    for i in range(num_cam):
        # Load the object mask
        object_idx = mask_info[i]["object"]
        mask = read_mask(f"{mask_path}/{i}/{object_idx}/{frame_idx}.png")
        object_mask = np.logical_and(masks[i], mask)
        object_points = points[i][object_mask]
        object_colors = colors[i][object_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points)
        pcd.colors = o3d.utility.Vector3dVector(object_colors)
        object_pcd += pcd

        # Load the controller mask
        controller_mask = np.zeros_like(masks[i])
        for controller_idx in mask_info[i]["controller"]:
            mask = read_mask(f"{mask_path}/{i}/{controller_idx}/{frame_idx}.png")
            controller_mask = np.logical_or(controller_mask, mask)
        controller_mask = np.logical_and(masks[i], controller_mask)
        controller_points = points[i][controller_mask]
        controller_colors = colors[i][controller_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(controller_points)
        pcd.colors = o3d.utility.Vector3dVector(controller_colors)
        controller_pcd += pcd

    # Apply the outlier removal
    cl, ind = object_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
    filtered_object_points = np.asarray(
        object_pcd.select_by_index(ind, invert=True).points
    )
    object_pcd = object_pcd.select_by_index(ind)

    cl, ind = controller_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
    filtered_controller_points = np.asarray(
        controller_pcd.select_by_index(ind, invert=True).points
    )
    controller_pcd = controller_pcd.select_by_index(ind)

    # controller_pcd.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([object_pcd, controller_pcd])
    object_pcd = o3d.geometry.PointCloud()
    controller_pcd = o3d.geometry.PointCloud()
    for i in range(num_cam):
        processed_masks[frame_idx][i] = {}
        # Load the object mask
        object_idx = mask_info[i]["object"]
        mask = read_mask(f"{mask_path}/{i}/{object_idx}/{frame_idx}.png")
        object_mask = np.logical_and(masks[i], mask)
        object_points = points[i][object_mask]
        indices = np.nonzero(object_mask)
        indices_list = list(zip(indices[0], indices[1]))
        # Locate all the object_points in the filtered points
        object_indices = []
        for j, point in enumerate(object_points):
            if tuple(point) in filtered_object_points:
                object_indices.append(j)
        original_indices = [indices_list[j] for j in object_indices]
        # Update the object mask
        for idx in original_indices:
            object_mask[idx[0], idx[1]] = 0
        processed_masks[frame_idx][i]["object"] = object_mask

        # Load the controller mask
        controller_mask = np.zeros_like(masks[i])
        for controller_idx in mask_info[i]["controller"]:
            mask = read_mask(f"{mask_path}/{i}/{controller_idx}/{frame_idx}.png")
            controller_mask = np.logical_or(controller_mask, mask)
        controller_mask = np.logical_and(masks[i], controller_mask)
        controller_points = points[i][controller_mask]
        indices = np.nonzero(controller_mask)
        indices_list = list(zip(indices[0], indices[1]))
        # Locate all the controller_points in the filtered points
        controller_indices = []
        for j, point in enumerate(controller_points):
            if tuple(point) in filtered_controller_points:
                controller_indices.append(j)
        original_indices = [indices_list[j] for j in controller_indices]
        # Update the controller mask
        for idx in original_indices:
            controller_mask[idx[0], idx[1]] = 0
        processed_masks[frame_idx][i]["controller"] = controller_mask

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[i][object_mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[i][object_mask])

        object_pcd += pcd

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[i][controller_mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[i][controller_mask])

        controller_pcd += pcd

    # o3d.visualization.draw_geometries([object_pcd, controller_pcd])

    return object_pcd, controller_pcd


if __name__ == "__main__":
    pcd_path = f"{base_path}/{case_name}/pcd"
    mask_path = f"{base_path}/{case_name}/mask"

    num_cam = len(glob.glob(f"{mask_path}/mask_info_*.json"))
    frame_num = len(glob.glob(f"{pcd_path}/*.npz"))
    # Load the mask metadata
    mask_info = {}
    for i in range(num_cam):
        with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
            data = json.load(f)
        mask_info[i] = {}
        for key, value in data.items():
            if value != CONTROLLER_NAME:
                if "object" in mask_info[i]:
                    # TODO: Handle the case when there are multiple objects
                    raise NotImplementedError(
                        f"mask_info_{i}.json has multiple non-controller "
                        f"entries ({data}); multi-object cases are not "
                        f"supported yet — re-run segmentation with a "
                        f"single-object TEXT_PROMPT."
                    )
                mask_info[i]["object"] = int(key)
            if value == CONTROLLER_NAME:
                if "controller" in mask_info[i]:
                    mask_info[i]["controller"].append(int(key))
                else:
                    mask_info[i]["controller"] = [int(key)]

    vis = o3d.visualization.Visualizer()
    if not vis.create_window():
        vis = None

    object_pcd = None
    controller_pcd = None
    for i in tqdm(range(frame_num)):
        temp_object_pcd, temp_controller_pcd = process_pcd_mask(
            i, pcd_path, mask_path, mask_info, num_cam
        )
        if i == 0:
            object_pcd = temp_object_pcd
            controller_pcd = temp_controller_pcd
            if vis is not None:
                vis.add_geometry(object_pcd)
                vis.add_geometry(controller_pcd)
                # Adjust the viewpoint
                view_control = vis.get_view_control()
                view_control.set_front([1, 0, -2])
                view_control.set_up([0, 0, -1])
                view_control.set_zoom(1)
        else:
            object_pcd.points = o3d.utility.Vector3dVector(temp_object_pcd.points)
            object_pcd.colors = o3d.utility.Vector3dVector(temp_object_pcd.colors)
            controller_pcd.points = o3d.utility.Vector3dVector(
                temp_controller_pcd.points
            )
            controller_pcd.colors = o3d.utility.Vector3dVector(
                temp_controller_pcd.colors
            )
            if vis is not None:
                vis.update_geometry(object_pcd)
                vis.update_geometry(controller_pcd)
                vis.poll_events()
                vis.update_renderer()

    # Save the processed masks considering both depth filter, semantic filter and outlier filter
    with open(f"{base_path}/{case_name}/mask/processed_masks.pkl", "wb") as f:
        pickle.dump(processed_masks, f)

    # Deprecated for now
    # # Generate the videos with for masked objects and controllers
    # exist_dir(f"{base_path}/{case_name}/temp_mask")
    # for i in range(num_cam):
    #     exist_dir(f"{base_path}/{case_name}/temp_mask/{i}")
    #     exist_dir(f"{base_path}/{case_name}/temp_mask/{i}/object")
    #     exist_dir(f"{base_path}/{case_name}/temp_mask/{i}/controller")
    #     object_idx = mask_info[i]["object"]
    #     for frame_idx in range(frame_num):
    #         object_mask = read_mask(f"{mask_path}/{i}/{object_idx}/{frame_idx}.png")
    #         img = cv2.imread(f"{base_path}/{case_name}/color/{i}/{frame_idx}.png")
    #         masked_object_img = cv2.bitwise_and(
    #             img, img, mask=object_mask.astype(np.uint8) * 255
    #         )
    #         cv2.imwrite(
    #             f"{base_path}/{case_name}/temp_mask/{i}/object/{frame_idx}.png",
    #             masked_object_img,
    #         )

    #         controller_mask = np.zeros_like(object_mask)
    #         for controller_idx in mask_info[i]["controller"]:
    #             mask = read_mask(f"{mask_path}/{i}/{controller_idx}/{frame_idx}.png")
    #             controller_mask = np.logical_or(controller_mask, mask)
    #         masked_controller_img = cv2.bitwise_and(
    #             img, img, mask=controller_mask.astype(np.uint8) * 255
    #         )
    #         cv2.imwrite(
    #             f"{base_path}/{case_name}/temp_mask/{i}/controller/{frame_idx}.png",
    #             masked_controller_img,
    #         )

    #     os.system(
    #         f"ffmpeg -r 30 -start_number 0 -f image2 -i {base_path}/{case_name}/temp_mask/{i}/object/%d.png -vcodec libx264 -crf 0  -pix_fmt yuv420p {base_path}/{case_name}/temp_mask/object_{i}.mp4"
    #     )
    #     os.system(
    #         f"ffmpeg -r 30 -start_number 0 -f image2 -i {base_path}/{case_name}/temp_mask/{i}/controller/%d.png -vcodec libx264 -crf 0  -pix_fmt yuv420p {base_path}/{case_name}/temp_mask/controller_{i}.mp4"
    #     )

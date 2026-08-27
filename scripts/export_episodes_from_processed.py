# Author: Guanxiong Chen
# Adapted from DROID's collab notebook

# Original file is located at
#     https://colab.research.google.com/drive/1b4PPH4XGht4Jve2xPKMCh-AXXAQziNQa

# # DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset

# ![](https://droid-dataset.github.io/droid/assets/index/droid_teaser.jpg)

# This Colab demonstrates how to load and visualize samples from the DROID dataset. Please also check out our [dataset visualizer](https://droid-dataset.github.io/dataset.html) to explore the dataset.

# You can download the full dataset (1.7TB) using:
# ```
# gsutil -m cp -r gs://gresearch/robotics/droid <your_local_path>
# ```

# If you'd like to download an example version of the dataset with 100 episodes first (2GB), run:
# ```
# gsutil -m cp -r gs://gresearch/robotics/droid_100 <your_local_path>
# ```

# Raw DROID Data: the RLDS version of the DROID dataset is optimized for loading speed during policy training. Thus, it contains lower-resolution images (180x320 px) and no stereo or depth data. If you need access to the full-HD, stereo videos we recorded, you can download the raw DROID data with the following command:
# ```
# gsutil -m cp -r gs://gresearch/robotics/droid_raw <your_local_path>
# ```
# Here we store data in MP4 format for more efficient storage. For more information about the raw data format, see our [developer documentation](https://droid-dataset.github.io/droid/the-droid-dataset.html).

# If you want to use DROID for policy training, please check out our [policy training repo](https://github.com/droid-dataset/droid_policy_learning).


import argparse
import os
import json
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from PIL import Image
import yaml
import h5py
from droid_data_utils.droid_utils import generate_video


def save_keyframes(images, episode_idx, output_dir):
  # Save first, middle, and last frames
  num_frames = len(images)
  keyframe_indices = [0, num_frames // 2, num_frames - 1]

  for frame_idx in keyframe_indices:
    filename = f"episode_{episode_idx:06d}_frame_{frame_idx:06d}.jpg"
    filepath = os.path.join(output_dir, filename)
    images[frame_idx].save(filepath, quality=70, optimize=True)


def create_h5_dataset(h5f, name, data):
  # Stronger compression: higher gzip level + shuffle filter.
  h5f.create_dataset(
    name,
    data=data,
    compression="gzip",
    compression_opts=9,
    shuffle=True,
    chunks=True,
    track_times=False)


def main():
  parser = argparse.ArgumentParser(
    description="Inspect DROID dataset and export episode-wise data.")
  parser.add_argument(
    "--generate_video", action="store_true")
  parser.add_argument(
    "--generate_keyframes", action="store_true")
  parser.add_argument(
    "--save_obs_to_h5", action="store_true")
  parser.add_argument(
    '--save_acts_to_h5', action='store_true')
  parser.add_argument(
    '--save_lang_inst', action='store_true')
  args = parser.parse_args()

  # builder = tfds.builder_from_directory(builder_dir="droid_data/droid_100/1.0.0/")
  # print(builder.info.features)

  # NOTE: val and test split are not available in the example dataset
  dataset_name = 'droid_100'
  for split in ["train"]:
    # load the split
    print(f"\nLoading split: {split}")
    ds = tfds.load(
      dataset_name,
      data_dir="droid_data/",
      split=split,
      shuffle_files=False)
    # print num of episodes in the split
    cardinality = tf.data.experimental.cardinality(ds)
    if cardinality == tf.data.INFINITE_CARDINALITY:
      print("Number of episodes: infinite")
    elif cardinality == tf.data.UNKNOWN_CARDINALITY:
      num_episodes = sum(1 for _ in ds)
      print(f"Number of episodes: {num_episodes} (computed)")
    else:
      print(f"Number of episodes: {cardinality.numpy()}")

    # Inspect episode-level metadata for a few episodes.
    for idx, episode in enumerate(ds.take(3)):
      print(f"\nEpisode {idx} keys:", list(episode.keys()))
      if "episode_metadata" in episode:
        metadata = episode["episode_metadata"]
        if isinstance(metadata, dict):
          print("  episode_metadata keys:")
          for key, value in metadata.items():
            try:
              value_np = value.numpy()
              if isinstance(value_np, (bytes, bytearray)):
                value_np = value_np.decode("utf-8", "replace")
              if hasattr(value_np, "shape") and value_np.shape != ():
                print(f"    {key}: shape={value_np.shape} dtype={value_np.dtype}")
              else:
                print(f"    {key}: {value_np}")
            except Exception as exc:
              print(f"    {key}: <unprintable: {exc}>")
        else:
          print("  episode_metadata:", metadata)
      else:
        print("  No episode_metadata field found.")

    # Inspect step-level data for the first episode.
    print("\nInspecting step-level data for the first episode...")
    first_episode = next(iter(ds))
    print("First episode keys:", list(first_episode.keys()))
    if "steps" in first_episode:
      steps = first_episode["steps"]
      print(f"Number of steps in the first episode: {len(steps)}")
      for step_idx, step in enumerate(steps.take(3)):
        print(f"\n  Step {step_idx} keys:", list(step.keys()))
        if "observation" in step:
          obs = step["observation"]
          print("    observation keys:", list(obs.keys()))
        if "action_dict" in step:
          act_dict = step["action_dict"]
          print("    action_dict keys:", list(act_dict.keys()))
    else:
      print("  No steps field found in the episode.")

    # process each episode
    output_dir = os.path.join("outputs", "export_episodes_from_processed", dataset_name, split)
    for episode_idx, episode in enumerate(ds):
      print(f"Processing episode {episode_idx}...")
      # define directory
      episode_dir = os.path.join(output_dir, f"episode_{episode_idx:06d}")
      # define dict to store metadata info
      metadata_dict = {}

      # make images from observations
      if args.generate_video or args.generate_keyframes:
        images = []
        for i, step in enumerate(episode["steps"]):
          images.append(
            Image.fromarray(
              np.concatenate((
                    step["observation"]["exterior_image_1_left"].numpy(),
                    step["observation"]["exterior_image_2_left"].numpy(),
                    step["observation"]["wrist_image_left"].numpy(),
              ), axis=1)
            )
          )
      # generate video or keyframes
      if args.generate_video:
        os.makedirs(episode_dir, exist_ok=True)
        video_path = os.path.join(
          episode_dir, f"{episode_idx:06d}.mp4")
        generate_video(images, path=video_path)
      if args.generate_keyframes:
        keyframes_dir = os.path.join(episode_dir, "keyframes")
        os.makedirs(keyframes_dir, exist_ok=True)
        save_keyframes(images, episode_idx, keyframes_dir)

      # save observations to h5 per episode
      if args.save_obs_to_h5:
        expected_obs_ks = [
          "cartesian_position",
          "exterior_image_1_left",
          "exterior_image_2_left",
          "gripper_position",
          "joint_position",
          "wrist_image_left"]
        cart_pos = []
        ext_img1_left = []
        ext_img2_left = []
        grip_pos = []
        joint_pos = []
        wrist_img_left = []
        for data_step in episode["steps"]:
          obs = data_step["observation"]
          for k in obs.keys():
            if k not in expected_obs_ks:
              # raise error if found other keys in the step observation
              raise ValueError(f"Unexpected observation key: {k}")
            else:
              if k in [
                "cartesian_position",
                "gripper_position",
                "joint_position"]:
                # 64->32 bit float
                value_np = obs[k].numpy().astype(np.float32)
              else:
                # save as uint8 numpy array
                value_np = obs[k].numpy().astype(np.uint8)
              # append numpy array to corresponding list
              if k == "cartesian_position":
                cart_pos.append(value_np)
              elif k == "exterior_image_1_left":
                ext_img1_left.append(value_np)
              elif k == "exterior_image_2_left":
                ext_img2_left.append(value_np)
              elif k == "gripper_position":
                grip_pos.append(value_np)
              elif k == "joint_position":
                joint_pos.append(value_np)
              elif k == "wrist_image_left":
                wrist_img_left.append(value_np)
              else:
                pass  # should not reach here
        # stack each array along frame dimension
        cart_pos_np = np.stack(cart_pos, axis=0)
        ext_img1_left_np = np.stack(ext_img1_left, axis=0)
        ext_img2_left_np = np.stack(ext_img2_left, axis=0)
        grip_pos_np = np.stack(grip_pos, axis=0)
        joint_pos_np = np.stack(joint_pos, axis=0)
        wrist_img_left_np = np.stack(wrist_img_left, axis=0)
        metadata_dict.update({
          'obs_cartesian_position': list(cart_pos_np.shape),
          'obs_exterior_image_1_left': list(ext_img1_left_np.shape),
          'obs_exterior_image_2_left': list(ext_img2_left_np.shape),
          'obs_gripper_position': list(grip_pos_np.shape),
          'obs_joint_position': list(joint_pos_np.shape),
          'obs_wrist_image_left': list(wrist_img_left_np.shape),
        })
        os.makedirs(episode_dir, exist_ok=True)
        # save to h5 file
        h5_filename = f"episode_{episode_idx:06d}_obs.h5"
        h5_filepath = os.path.join(episode_dir, h5_filename)
        with h5py.File(h5_filepath, 'w') as h5f:
          create_h5_dataset(h5f, 'cartesian_position', cart_pos_np)
          create_h5_dataset(h5f, 'exterior_image_1_left', ext_img1_left_np)
          create_h5_dataset(h5f, 'exterior_image_2_left', ext_img2_left_np)
          create_h5_dataset(h5f, 'gripper_position', grip_pos_np)
          create_h5_dataset(h5f, 'joint_position', joint_pos_np)
          create_h5_dataset(h5f, 'wrist_image_left', wrist_img_left_np)
          print(f"Saved episode {episode_idx} observations to {h5_filepath}")

      # save actions to h5 per episode
      if args.save_acts_to_h5:
        expected_act_ks = [
            "cartesian_position",
            "cartesian_velocity",
            "gripper_position",
            "gripper_velocity",
            "joint_position",
            "joint_velocity"]
        act_7d = []
        cart_pos = []
        cart_vel = []
        grip_pos = []
        grip_vel = []
        joint_pos = []
        joint_vel = []
        for data_step in episode["steps"]:
          act_dict = data_step["action_dict"]
          for k in act_dict.keys():
            if k not in expected_act_ks:
              # raise error if found other keys in the step action
              raise ValueError(f"Unexpected action key: {k}")
            else:
              # 64->32 bit float
              value_np = act_dict[k].numpy().astype(np.float32)
              # append numpy array to corresponding list
              if k == "cartesian_position":
                cart_pos.append(value_np)
              elif k == "cartesian_velocity":
                cart_vel.append(value_np)
              elif k == "gripper_position":
                grip_pos.append(value_np)
              elif k == "gripper_velocity":
                grip_vel.append(value_np)
              elif k == "joint_position":
                joint_pos.append(value_np)
              elif k == "joint_velocity":
                joint_vel.append(value_np)
              else:
                pass  # should not reach here
          # also, we add the 7D action tensor
          act_7d.append(data_step["action"].numpy().astype(np.float32))
        # stack each array along frame dimension
        cart_pos_np = np.stack(cart_pos, axis=0)
        cart_vel_np = np.stack(cart_vel, axis=0)
        grip_pos_np = np.stack(grip_pos, axis=0)
        grip_vel_np = np.stack(grip_vel, axis=0)
        joint_pos_np = np.stack(joint_pos, axis=0)
        joint_vel_np = np.stack(joint_vel, axis=0)
        act_7d_np =  np.stack(act_7d, axis=0)
        metadata_dict.update({
          'action_cartesian_position': list(cart_pos_np.shape),
          'action_cartesian_velocity': list(cart_vel_np.shape),
          'action_gripper_position': list(grip_pos_np.shape),
          'action_gripper_velocity': list(grip_vel_np.shape),
          'action_joint_position': list(joint_pos_np.shape),
          'action_joint_velocity': list(joint_vel_np.shape),
          'action_7d': list(act_7d_np.shape),})
        os.makedirs(episode_dir, exist_ok=True)
        # save to h5 file
        h5_filename = f"episode_{episode_idx:06d}_acts.h5"
        h5_filepath = os.path.join(episode_dir, h5_filename)
        with h5py.File(h5_filepath, 'w') as h5f:
          create_h5_dataset(h5f, 'cartesian_position', cart_pos_np)
          create_h5_dataset(h5f, 'cartesian_velocity', cart_vel_np)
          create_h5_dataset(h5f, 'gripper_position', grip_pos_np)
          create_h5_dataset(h5f, 'gripper_velocity', grip_vel_np)
          create_h5_dataset(h5f, 'joint_position', joint_pos_np)
          create_h5_dataset(h5f, 'joint_velocity', joint_vel_np)
          create_h5_dataset(h5f, 'action_7d', act_7d_np)
          print(f"Saved episode {episode_idx} actions to {h5_filepath}")

      if args.save_lang_inst:
        os.makedirs(episode_dir, exist_ok=True)
        lang_steps = []
        for step_idx, data_step in enumerate(episode["steps"]):
          step_entry = {}
          step_entry["step_idx"] = int(step_idx)
          for key in [
            "language_instruction",
            "language_instruction_2",
            "language_instruction_3"]:
            value = data_step[key]
            try:
              value_np = value.numpy()
            except AttributeError:
              value_np = value
            if isinstance(value_np, (bytes, bytearray)):
              value_str = value_np.decode("utf-8", "replace")
            else:
              value_str = str(value_np)
            step_entry[key] = value_str
          lang_steps.append(step_entry)
        lang_path = os.path.join(
          episode_dir, f"episode_{episode_idx:06d}_lang_ins.json")
        with open(lang_path, "w", encoding="utf-8") as f:
          json.dump(lang_steps, f, ensure_ascii=False, indent=2)
        print(f"Saved episode {episode_idx} language instructions {lang_path}")

      # add metadata to metadata_dict
      metadata_dict["file_path"] = episode[
        "episode_metadata"]["file_path"].numpy().decode("utf-8", "replace")

      # save metadata
      if len(metadata_dict.keys()) > 0:
        metadata = os.path.join(
          episode_dir, f"episode_{episode_idx:06d}_metadata.yaml")
        with open(metadata, 'w') as yaml_file:
          yaml.dump(metadata_dict, yaml_file)
        print(f"Saved episode {episode_idx} metadata {metadata}")


if __name__ == "__main__":
    main()

from PIL import Image
from prismatic.vla.datasets.rlds.utils.data_utils import (
    binarize_gripper_actions,
    invert_gripper_actions,
    rel2abs_gripper_actions,
    relabel_bridge_actions,
)

import torch
import flash_attn_2_cuda as flash_attn_cuda
import numpy as np
import accelerate
import tensorflow_datasets as tfds
import cv2
import os
import tensorflow as tf
import tqdm
import matplotlib.pyplot as plt
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


import wandb
import random
import time

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="oxe_action_vis",
    name="viola",
    # track hyperparameters and run metadata
    config={
    }
)
os.environ['CURL_CA_BUNDLE'] = "/etc/ssl/certs/ca-bundle.crt"

builder_dir = 'gs://gresearch/robotics/viola/0.1.0'
# Franka: 'gs://gresearch/robotics/cmu_play_fusion/0.1.0' 
# Franka: 'gs://gresearch/robotics/viola/0.1.0' unnorm_key viola
# Franka: 'gs://gresearch/robotics/kaist_nonprehensile_converted_externally_to_rlds/0.1.0' unnorm_key 'kaist_nonprehensile_converted_externally_to_rlds'
# Franka: 'gs://gresearch/robotics/stanford_hydra_dataset_converted_externally_to_rlds/0.1.0'  unnorm_key 'stanford_hydra_dataset_converted_externally_to_rlds'
# Franka: 'gs://gresearch/robotics/utaustin_mutex/0.1.0' unnorm_key "utaustin_mutex"
# Franka: 'gs://gresearch/robotics/taco_play/0.1.0' unnorm_key "taco_play" 
# Franka: 'gs://gresearch/robotics/nyu_franka_play_dataset_converted_externally_to_rlds/0.1.0' unnorm_key "nyu_franka_play_dataset_converted_externally_to_rlds"
# WidowX: 'gs://gresearch/robotics/bridge/0.1.0/'  "bridge_orig" 


# create RLDS dataset builder
builder = tfds.builder_from_directory(builder_dir=builder_dir)
ds = builder.as_dataset(split='train')

ep = 0
for episode in iter(ds):
    # sample episode + resize to 256x256 (default third-person cam resolution)
    steps = list(episode['steps'])

    # extract goal image & language instruction
    if "bridge" in builder_dir:
        language_instruction = steps[0]['observation']['natural_language_instruction'].numpy().decode()
        images = [cv2.resize(np.array(step['observation']['image']), (512, 512)) for step in steps]
    elif "cmu_franka" in builder_dir or "nyu_franka" in builder_dir:
        language_instruction = steps[0]['language_instruction'].numpy().decode()
        images = [cv2.resize(np.array(step['observation']['image']), (512, 512)) for step in steps]
    elif "taco_play" in builder_dir:
        language_instruction = steps[0]['observation']['natural_language_instruction'].numpy().decode()
        images = [cv2.resize(np.array(step['observation']['rgb_static']), (512, 512)) for step in steps]
        gripper_images = [cv2.resize(np.array(step['observation']['rgb_gripper']), (512, 512)) for step in steps]
    elif "utaustin_mutex" in builder_dir:
        language_instruction = steps[0]['language_instruction'].numpy().decode()
        images = [cv2.resize(np.array(step['observation']['image']), (512, 512)) for step in steps]
    elif "stanford_hydra" in builder_dir:
        language_instruction = steps[0]['language_instruction'].numpy().decode()
        images = [cv2.resize(np.array(step['observation']['image']), (512, 512)) for step in steps]
        gripper_images = [cv2.resize(np.array(step['observation']['wrist_image']), (512, 512)) for step in steps]
    elif "kaist_nonprehensile" in builder_dir:
        language_instruction = steps[0]['language_instruction'].numpy().decode()
        images = [cv2.resize(np.array(step['observation']['image']), (512, 512)) for step in steps]
    elif "viola" in builder_dir:
        language_instruction = steps[0]['observation']['natural_language_instruction'].numpy().decode()
        images = [cv2.resize(np.array(step['observation']['agentview_rgb']), (512, 512)) for step in steps]
        gripper_images = [cv2.resize(np.array(step['observation']['eye_in_hand_rgb']), (512, 512)) for step in steps]
    elif "cmu_play" in builder_dir:
        language_instruction = steps[0]['language_instruction'].numpy().decode()
        images = [cv2.resize(np.array(step['observation']['image']), (512, 512)) for step in steps]
    else:
        raise TypeError(f"{builder_dir} is not supported by this script, please add a condition for true and pred values.")

    vids = np.array(images) # (T, H, W, C)
    vids = vids.transpose(0, 3, 1, 2) # (T, 3, H, W)
    wandb.log({f"video ep {ep}": wandb.Video(vids, fps=8)})

    try:
        gripper_vids = np.array(gripper_images) # (T, H, W, C)
        gripper_vids = gripper_vids.transpose(0, 3, 1, 2) # (T, 3, H, W)
        wandb.log({f"gripper video ep {ep}": wandb.Video(gripper_vids, fps=8)})
    except:
        print("This dataset does not have a gripper camera")

    # visualize episode
    print(f'Instruction: {language_instruction}')




    # Grab image input & format prompt
    # image: Image.Image = get_from_camera(...)
    prompt = language_instruction

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    print(f"Task: {prompt}")

    true_actions = []
    for step in tqdm.tqdm(range(0, len(images)-1)):

        if "bridge" in builder_dir:
            true_actions.append(np.concatenate(
                (
                    steps[step]['action']['world_vector'],
                    steps[step]['action']['rotation_delta'],
                    np.array(steps[step]['action']['open_gripper']).astype(np.float32)[None]
                ), axis=-1
            ))
        elif "cmu_franka" in builder_dir:
            true_actions.append(
                    steps[step]['action'][:7] # cmu_franka action is a vector of 8 [dX, dTheta, gripper_state, is_terminal]
            )
        elif "nyu_franka" in builder_dir:
            true_actions.append(
                    steps[step]['action'][7:14]
            )
        elif "taco_play" in builder_dir:
            true_actions.append(
                steps[step]['action']['rel_actions_world']
            )
        elif "utaustin_mutex" in builder_dir:
            true_actions.append(
                steps[step]['action']
            )
        elif "stanford_hydra" in builder_dir:
            true_actions.append(
                steps[step]['action']
            )
        elif "kaist_nonprehensile" in builder_dir:
            true_actions.append(
                steps[step]['action'][0:6] # no gripper state
            )
        elif "viola" in builder_dir:
            true_actions.append(np.concatenate(
                (
                    steps[step]["action"]["world_vector"],
                    steps[step]["action"]["rotation_delta"],
                    steps[step]["observation"]["gripper_states"],
                ),
                axis=-1,
            ))
        else:
            raise TypeError(f"{builder_dir} is not supported by this script, please add a condition for true and pred values.")
        print(f"step {step} done")

    ACTION_DIM_LABELS = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'grasp']

    # build image strip to show above actions
    img_strip = np.concatenate(np.array(images[::3]), axis=1)

    # set up plt figure
    figure_layout = [
        ['image'] * len(ACTION_DIM_LABELS),
        ACTION_DIM_LABELS
    ]
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([45, 10])

    # plot actions
    true_actions = np.array(true_actions).squeeze()
    integrate = np.sum(true_actions, axis=0)
    print(f'[dx dy dz  dyaw dpitch droll] = {integrate[0:6]}')
    wandb.log({f"Integrate x for ep {ep}": integrate[0], 
    f"Integrate y for ep {ep}": integrate[1],
    f"Integrate z for ep {ep}": integrate[2],
    f"Integrate yaw for ep {ep}": integrate[3],
    f"Integrate pitch for ep {ep}": integrate[4],
    f"Integrate roll for ep {ep}": integrate[5],
    })
    for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
        try:
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(true_actions[:, action_dim], label='ground truth')
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel('Time in one episode')
        except:
            pass

    axs['image'].imshow(img_strip)
    axs['image'].set_xlabel('Time in one episode (subsampled)')
    plt.legend()

    wandb.log({f"chart ep{ep}": plt})
    ep+=1
    if ep == 10:
        break
wandb.finish()
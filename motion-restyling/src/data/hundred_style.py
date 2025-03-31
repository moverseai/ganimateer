import logging
import os
import typing
from collections import defaultdict

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import OneHotEncoder

from src.monads.body.models import ModelAttribute, ModelType

log = logging.getLogger(__name__)


import re


class HundredStyleDataset(torch.utils.data.Dataset):
    _smpl_default_landmarks = [
        "pelvis",  # 0
        "left_hip",  # 1
        "right_hip",  # 2
        "spine1",  # 3
        "left_knee",  # 4
        "right_knee",  # 5
        "spine2",  # 6
        "left_ankle",  # 7
        "right_ankle",  # 8
        "spine3",  # 9
        "left_foot",  # 10
        "right_foot",  # 11
        "neck",  # 12
        "left_collar",  # 13
        "right_collar",  # 14
        "head",  # 15
        "left_shoulder",  # 16
        "right_shoulder",  # 17
        "left_elbow",  # 18
        "right_elbow",  # 19
        "left_wrist",  # 20
        "right_wrist",  # 21
        "left_hand",  # 22
        "right_hand",  # 23
    ]
    # TODO: we should add a mapping as some bvh has different joint names

    def __init__(
        self,
        root: str,
        styles: typing.Sequence[str],
        movements_type: typing.Sequence[str],
        sequence_ids: typing.Optional[typing.Sequence] = None,
        gender: str = "male",
        model_type: str = "smpl",
        fix_rotation: bool = False,  # apply rotation to face forward
    ) -> None:
        """
        root: str -> root folder containing the BVH files
        styles: typing.Sequence[str] -> list of styles to load
        movements_type: typing.Sequence[str] -> list of movements to load
        movements_type:
            BR	Backwards Running
            BW	Backwards Walking
            FR	Forwards Running
            FW	Forwards Walking
            ID	Idling
            SR	Sidestep Running
            SW	Sidestep Walking
            TR1/TR2/TR3	Transitions
        """
        super().__init__()
        if styles == "all" or styles == "*":
            styles = [folder for folder in os.listdir(root) if os.path.isdir(folder)]
        if movements_type == "all" or movements_type == "*":
            movements_type = [
                "BR",
                "BW",
                "FR",
                "FW",
                "ID",
                "SR",
                "SW",
                "TR1",
                "TR2",
                "TR3",
            ]
        self.joint_names = []
        self.parent_indices = []
        self.offsets = []
        self.frames = []
        self.total_frames = []
        self.channels = []
        self.frame_time = 0.0
        self.content = []
        self.style_labels = [st for st in styles if st != "Neutral"]
        self.styles = []
        self.gender = gender
        self.model_type = model_type
        self.fix_rotation = fix_rotation
        self.indices = defaultdict(list)
        # get correct mapping of joints
        for style in styles:
            for movement_type in movements_type:
                if sequence_ids is None:
                    path = os.path.join(root, style, f"{style}_{movement_type}.bvh")
                    # read the bvh file and get all the frames
                    self.load_bvh(path)
                else:
                    for sequence_id in sequence_ids:
                        path = os.path.join(
                            root, style, f"{style}_{movement_type}_{sequence_id}.bvh"
                        )
                        # read the bvh file and get all the frames
                        self.load_bvh(path)
                        self.total_frames.append(self.frames)
                        if style not in self.indices.keys():
                            for joint in self._smpl_default_landmarks:
                                if joint in self.joint_names:
                                    self.indices[style].append(
                                        self.joint_names.index(joint)
                                    )
                                else:
                                    self.indices[style].append(-1)
                        # the content should be a list of same length as the number of frames
                        self.content.extend([movement_type] * len(self.frames))
                        # similarly for the styles
                        self.styles.extend([style] * len(self.frames))
        self.frames = np.vstack(self.total_frames)
        cat = OneHotEncoder()
        self.one_hot_styles = (
            cat.fit_transform(np.array(self.styles)[np.newaxis].T)
            .toarray()
            .astype(np.float32)
        )
        self.movements = (
            cat.fit_transform(np.array(self.content)[np.newaxis].T)
            .toarray()
            .astype(np.float32)
        )
        log.info(
            f"Loaded styles {styles} and movements {movements_type}, in a total of {len(self)} samples."
        )

    def load_bvh(self, file_path: str) -> None:
        with open(file_path, "r") as f:
            lines = f.readlines()

        if len(self.joint_names) == 0:
            self.parse_hierarchy(lines)  # should be called only once
        self.parse_motion_data(lines)

    def parse_hierarchy(self, lines):
        joint_stack = []
        current_index = -1
        joint_pattern = re.compile(r"^(JOINT|ROOT)\s+([\w\s]+)$")
        offset_pattern = re.compile(
            r"^OFFSET\s+([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)\s+([-+]?[0-9]*\.?[0-9]+)$"
        )
        channel_pattern = re.compile(r"^CHANNELS\s+(\d+)\s+(.*)$")

        ignore_end_site = False
        ignore_end_site_bracket = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip empty or comment lines

            joint_match = joint_pattern.match(line)
            offset_match = offset_pattern.match(line)
            channel_match = channel_pattern.match(line)

            if line.startswith("End Site"):
                ignore_end_site = True
                ignore_end_site_bracket = True

            if joint_match:
                keyword, joint_name = joint_match.groups()
                joint_name = joint_name.strip()
                self.joint_names.append(joint_name)
                current_index += 1
                if joint_stack:
                    self.parent_indices.append(joint_stack[-1])
                else:
                    self.parent_indices.append(-1)
                joint_stack.append(current_index)
                self.offsets.append(np.zeros(3))  # Initialize with zero vector
            elif offset_match and not ignore_end_site:
                x, y, z = map(float, offset_match.groups())
                self.offsets[-1] = np.array([x, y, z])
            elif offset_match:
                # ignore end site final offset
                # reset flag
                ignore_end_site = False
            elif channel_match:
                tmp = {}
                num_channels, orders = channel_match.groups()
                tmp["num_channels"] = int(num_channels)
                tmp["orders"] = orders.split()
                self.channels.append(tmp)

            elif line == "}" and not ignore_end_site_bracket:
                if joint_stack:
                    popped = joint_stack.pop()
            elif line == "}" and ignore_end_site_bracket:
                ignore_end_site_bracket = False
            elif "Frames:" in line or "MOTION" in line:
                break

    def y_rotation_from_positions(
        self, left_hip, right_hip, left_shoulder, right_shoulder
    ):
        """
        Calculates the rotation matrix needed to align the character's local forward direction
        with the global forward direction (along the z-axis).

        Parameters:
            left_hip (np.ndarray): Left hip joint positions over time [T, 3].
            right_hip (np.ndarray): Right hip joint positions over time [T, 3].
            left_shoulder (np.ndarray): Left shoulder joint positions over time [T, 3].
            right_shoulder (np.ndarray): Right shoulder joint positions over time [T, 3].

        Returns:
            np.ndarray: Rotation matrices to apply to root rotation to face forward [T, 3, 3].
        """
        # Compute across vector (local x-axis) from hips and shoulders
        across = left_hip - right_hip + left_shoulder - right_shoulder
        across /= np.linalg.norm(across, axis=-1, keepdims=True)

        # Compute forward vector as the cross product with the y-axis
        forward = np.cross(across, np.array([0, 1, 0]))
        # forward = gaussian_filter1d(forward, sigma=20, axis=0, mode="nearest")
        forward /= np.linalg.norm(forward, axis=-1, keepdims=True)

        # Define the target forward vector (global z-axis)
        target_forward = np.array([0, 0, 1])
        # find in between rotation
        a = np.cross(forward, target_forward)
        w = np.sqrt(
            (np.linalg.norm(forward) ** 2) * (np.linalg.norm(target_forward) ** 2)
        ) + np.dot(forward, target_forward)
        # q = np.concatenate((a, w), axis=-1)
        q = np.concatenate((w[..., np.newaxis], a), axis=-1)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        # convert to rotation matrix
        # rot_mat = R.from_quat(q).as_matrix()

        # Calculate the rotation matrix needed to align the forward vector with the target
        rot_mat = R.align_vectors(target_forward, forward)[0].as_matrix()

        return rot_mat

    def parse_motion_data(self, lines):
        frame_index = lines.index("MOTION\n") + 3
        self.frame_time = float(lines[frame_index - 1].split()[2])
        self.frames = np.array(
            [np.fromstring(frame, sep=" ") for frame in lines[frame_index:]]
        )

    def get_joint_positions(self, frame):
        # Initialize positions and rotations
        positions = np.zeros((len(self.joint_names), 3))
        rotations = [R.identity()] * len(self.joint_names)
        rotvec = np.zeros((len(self.joint_names), 3))

        # Process the frame for root joint
        idx = 0
        root_channel = self.channels[0]["num_channels"]
        root_orders = self.channels[0]["orders"]
        root_rot_order = [l[0] for l in self.channels[0]["orders"][3:]]
        root_position = frame[
            idx : idx + 3
        ]  # assume that the first 3 channels are the root position
        root_rotation = R.from_euler(
            "".join(root_rot_order), frame[idx + 3 : idx + 6], degrees=True
        )  # Ensure correct rotation order
        positions[0] = root_position
        world_positions = np.zeros((len(self.joint_names), 3))
        rotations[0] = root_rotation
        rotvec[0] = root_rotation.as_rotvec()
        world_positions[0] = root_position
        idx += root_channel  # TODO: this should come fromt the bvh file itself (e.g. from the channels )

        # Process the frame for other joints
        for i in range(1, len(self.joint_names)):
            joint_orders = self.channels[i]["orders"]
            if len(joint_orders) == 6:
                joint_orders = joint_orders[3:]
            joint_order = [l[0] for l in joint_orders]
            start = 0 if self.channels[i]["num_channels"] == 3 else 3
            end = 3 if self.channels[i]["num_channels"] == 3 else 6
            rotation = R.from_euler(
                "".join(joint_order),
                frame[idx + start : idx + end],
                degrees=True,  # TODO: this should come fromt the bvh file itself (e.g. from the channels )
            )  # Ensure correct rotation order
            parent_idx = self.parent_indices[i]
            parent_rot = rotations[parent_idx]
            local_offset = self.offsets[i]
            parent_world_pos = world_positions[parent_idx]
            rotvec[i] = rotation.as_rotvec()

            rotations[i] = parent_rot * rotation
            positions[i] = parent_rot.apply(local_offset) + positions[parent_idx]
            world_positions[i] = parent_rot.apply(local_offset) + parent_world_pos
            idx += self.channels[i][
                "num_channels"
            ]  # TODO: this should come fromt the bvh file itself (e.g. from the channels )

        return positions, rotvec

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # create an one hot encoding for the style and movement
        style_array = np.zeros(len(self.style_labels))
        # get style
        style = self.styles[idx]
        # log.info(f"Style: {style}")
        if style != "Neutral":
            style_array[self.style_labels.index(style)] = 1
        movement = self.content[idx]
        # neutral should be zero and the other an increasing number starting from 1 to the different styles and movements
        style_encoding = torch.zeros(len(self.styles))
        movement_encoding = torch.zeros(len(self.content))
        style_encoding[self.styles.index(style)] = 1
        movement_encoding[self.content.index(movement)] = 1
        frame = self.frames[idx]
        joint_positions, joint_rotations = self.get_joint_positions(frame)
        # joint_positions /= 100  # Convert to meters if necessary
        # then take a random style
        transfered_style = np.zeros(len(self.style_labels))
        # if self.one_hot_styles[idx][0:1].sum() == 0.0:
        #     transfered_style[np.random.randint(0, len(self.style_labels))] = 1
        input_style = np.zeros(len(self.style_labels))
        if style != "Neutral":
            transfered_style[np.random.randint(0, len(self.style_labels))] = 0
            input_style[self.style_labels.index(style)] = 1
        else:
            transfered_style[np.random.randint(0, len(self.style_labels))] = 1

        rot_mat_to_apply = np.eye(3)
        if self.fix_rotation:
            # calculate the rotation to face forward
            left_hip = joint_positions[self.joint_names.index("left_hip")]
            right_hip = joint_positions[self.joint_names.index("right_hip")]
            left_shoulder = joint_positions[self.joint_names.index("left_shoulder")]
            right_shoulder = joint_positions[self.joint_names.index("right_shoulder")]
            rot_mat_to_apply = self.y_rotation_from_positions(
                left_hip, right_hip, left_shoulder, right_shoulder
            )
            new_joint_positions = np.einsum(
                "ij,kj->ki", rot_mat_to_apply, joint_positions
            )
            # find new root rotation
            old_root_rot = joint_rotations[0]
            new_root_rot = rot_mat_to_apply @ R.from_rotvec(old_root_rot).as_matrix()
            joint_rotations[0] = R.from_matrix(new_root_rot).as_rotvec()
            joint_positions = new_joint_positions
            # remove root
            # joint_positions_offseted = joint_positions - joint_positions[0]
            # new_joint_positions = (rot_mat_to_apply @ joint_positions_offseted.T).T

        return {
            "joint_positions": torch.tensor(joint_positions, dtype=torch.float32)[
                self.indices[style]
            ],
            "joint_rotations": torch.tensor(joint_rotations, dtype=torch.float32)[
                self.indices[style]
            ],
            # "style": style,
            "input_style": torch.tensor(input_style, dtype=torch.float32),
            # "input_style": self.one_hot_styles[idx][0:1],
            "transferred_style": torch.tensor(transfered_style, dtype=torch.float32),
            "content": self.movements[idx],
            # int(style_array[self.style_labels.index(style)]),
            "betas": torch.zeros(10),
            "zero_translation": torch.zeros(3),
            "transformed_mat": torch.from_numpy(rot_mat_to_apply).float(),
            "metadata": {
                "gender": ModelAttribute.get_enum(self.gender),
                "model_type": ModelType.get_enum(self.model_type),
            },
        }

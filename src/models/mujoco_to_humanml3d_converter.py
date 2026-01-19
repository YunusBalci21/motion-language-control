# src/models/mujoco_to_humanml3d_converter.py
"""
Convert MuJoCo observations to HumanML3D-compatible format (259 dimensions)

Note: The checkpoint we're using expects 259 features, not the standard 263.
This is a variant of HumanML3D format.

Format (259 dims):
- Root position (3): x, y, z
- Root rotation (6): 6D rotation representation
- Joint positions (63): 21 joints * 3
- Root velocity (3): vx, vy, vz
- Root angular velocity (3): wx, wy, wz
- Joint velocities (63): 21 joints * 3
- Foot contacts (4): left_heel, left_toe, right_heel, right_toe
- Additional features to reach 263

Our MuJoCo features (30 dims):
[0]: height
[1-4]: quaternion
[5-15]: joint positions (varies by env)
[16-18]: velocities (vx, vy, vz)
[26]: speed magnitude
"""

import numpy as np
import torch
from typing import Union


class MuJoCoToHumanML3DConverter:
    """Convert MuJoCo motion features to HumanML3D format (259 dims to match checkpoint)"""

    def __init__(self):
        self.humanml3d_dim = 259  # Changed from 263 to match checkpoint
        self.mujoco_dim = 30

    def convert(self, mujoco_features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Convert MuJoCo features to HumanML3D format

        Args:
            mujoco_features: (T, 30) or (B, T, 30) - MuJoCo motion features

        Returns:
            humanml3d_features: (T, 259) or (B, T, 259) - HumanML3D format (259 dims)
        """
        # Convert to torch if numpy
        if isinstance(mujoco_features, np.ndarray):
            mujoco_features = torch.from_numpy(mujoco_features).float()

        original_shape = mujoco_features.shape

        # Handle batch dimension
        if mujoco_features.dim() == 2:
            # (T, 30)
            batch_mode = False
            T, D = mujoco_features.shape
            B = 1
        elif mujoco_features.dim() == 3:
            # (B, T, 30)
            batch_mode = True
            B, T, D = mujoco_features.shape
            mujoco_features = mujoco_features.reshape(-1, D)  # (B*T, 30)
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {mujoco_features.shape}")

        assert D == self.mujoco_dim, f"Expected {self.mujoco_dim} features, got {D}"

        device = mujoco_features.device
        N = mujoco_features.shape[0]  # Total frames (B*T or T)

        # Initialize HumanML3D features
        humanml3d = torch.zeros(N, self.humanml3d_dim, device=device)

        # Extract from MuJoCo features
        height = mujoco_features[:, 0:1]  # (N, 1)
        quat = mujoco_features[:, 1:5]  # (N, 4) - quaternion
        joints = mujoco_features[:, 5:16]  # (N, 11) - joint positions
        vx = mujoco_features[:, 16:17]  # (N, 1)
        vy = mujoco_features[:, 17:18]  # (N, 1)
        vz = mujoco_features[:, 18:19]  # (N, 1)
        speed = mujoco_features[:, 26:27]  # (N, 1)

        # === Map to HumanML3D format ===

        # Root position (0:3) - use height and infer x,y from velocity
        humanml3d[:, 0:1] = torch.zeros_like(height)  # x (we don't track absolute x)
        humanml3d[:, 1:2] = torch.zeros_like(height)  # y (we don't track absolute y)
        humanml3d[:, 2:3] = height  # z (height)

        # Root rotation (3:9) - convert quaternion to 6D rotation
        rot_6d = self._quat_to_6d_rotation(quat)
        humanml3d[:, 3:9] = rot_6d

        # Joint positions (9:72) - 21 joints * 3
        # We have 11 joint values, distribute them across the 21 joints
        joint_positions_extended = self._extend_joints(joints, target_size=63)
        humanml3d[:, 9:72] = joint_positions_extended

        # Root velocity (72:75)
        humanml3d[:, 72:73] = vx
        humanml3d[:, 73:74] = vy
        humanml3d[:, 74:75] = vz

        # Root angular velocity (75:78) - approximate from quaternion changes
        humanml3d[:, 75:78] = 0.0

        # Joint velocities (78:141) - 21 joints * 3
        # Approximate as proportional to root velocity
        joint_velocities = self._approximate_joint_velocities(joints, vx, vy, vz)
        humanml3d[:, 78:141] = joint_velocities

        # Foot contacts (141:145) - 4 binary values
        # Estimate from height and velocity
        foot_contacts = self._estimate_foot_contacts(height, speed)
        humanml3d[:, 141:145] = foot_contacts

        # Additional features (145:259) - 114 dims
        additional_features = self._generate_additional_features(
            joints, vx, vy, vz, height, quat
        )
        humanml3d[:, 145:259] = additional_features

        # Reshape back to original batch structure
        if batch_mode:
            humanml3d = humanml3d.reshape(B, T, self.humanml3d_dim)
        else:
            humanml3d = humanml3d.reshape(T, self.humanml3d_dim)

        return humanml3d

    def _quat_to_6d_rotation(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to 6D rotation representation

        Args:
            quat: (N, 4) - [w, x, y, z]

        Returns:
            rot_6d: (N, 6) - 6D rotation
        """
        # Normalize quaternion
        quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)

        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Convert to rotation matrix
        r00 = 1 - 2 * (y ** 2 + z ** 2)
        r01 = 2 * (x * y - w * z)
        r02 = 2 * (x * z + w * y)
        r10 = 2 * (x * y + w * z)
        r11 = 1 - 2 * (x ** 2 + z ** 2)
        r12 = 2 * (y * z - w * x)

        # Take first 2 columns as 6D representation
        rot_6d = torch.stack([r00, r10, r01, r11, r02, r12], dim=-1)

        return rot_6d

    def _extend_joints(self, joints: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Extend joint positions from 11 to target_size (63 for 21 joints * 3)

        Args:
            joints: (N, 11)
            target_size: int (e.g., 63)

        Returns:
            extended: (N, target_size)
        """
        N = joints.shape[0]
        device = joints.device

        extended = torch.zeros(N, target_size, device=device)

        # Distribute available joint info across the target joints
        # Simple strategy: replicate and interpolate
        num_joints_target = target_size // 3
        num_joints_source = joints.shape[1]

        for i in range(num_joints_target):
            # Map source joint index
            src_idx = int(i * num_joints_source / num_joints_target)
            src_idx = min(src_idx, num_joints_source - 1)

            # Assign to 3D position (x, y, z)
            base_idx = i * 3
            extended[:, base_idx:base_idx + 3] = joints[:, src_idx:src_idx + 1].expand(-1, 3)

        return extended

    def _approximate_joint_velocities(
            self,
            joints: torch.Tensor,
            vx: torch.Tensor,
            vy: torch.Tensor,
            vz: torch.Tensor
    ) -> torch.Tensor:
        """
        Approximate joint velocities based on root velocity

        Args:
            joints: (N, 11)
            vx, vy, vz: (N, 1) each

        Returns:
            joint_vels: (N, 63)
        """
        N = joints.shape[0]
        device = joints.device

        joint_vels = torch.zeros(N, 63, device=device)

        # Simple approximation: scale root velocity by joint positions
        root_vel = torch.cat([vx, vy, vz], dim=-1)  # (N, 3)

        for i in range(21):  # 21 joints
            base_idx = i * 3
            # Each joint moves proportionally to root velocity
            # with some variation based on its index (proximal vs distal)
            scale = 1.0 - (i / 21) * 0.5  # Distal joints move less
            joint_vels[:, base_idx:base_idx + 3] = root_vel * scale

        return joint_vels

    def _estimate_foot_contacts(
            self,
            height: torch.Tensor,
            speed: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate foot contact from height and speed

        Args:
            height: (N, 1)
            speed: (N, 1)

        Returns:
            contacts: (N, 4) - [left_heel, left_toe, right_heel, right_toe]
        """
        N = height.shape[0]
        device = height.device

        contacts = torch.zeros(N, 4, device=device)

        # Simple heuristic: if height is low and speed is low, feet are on ground
        on_ground = ((height < 1.2) & (speed < 0.5)).float()

        # Assume alternating foot contacts based on time
        # This is a rough approximation
        contacts[:, 0] = on_ground.squeeze()  # left heel
        contacts[:, 1] = on_ground.squeeze()  # left toe
        contacts[:, 2] = on_ground.squeeze()  # right heel
        contacts[:, 3] = on_ground.squeeze()  # right toe

        return contacts

    def _generate_additional_features(
            self,
            joints: torch.Tensor,
            vx: torch.Tensor,
            vy: torch.Tensor,
            vz: torch.Tensor,
            height: torch.Tensor,
            quat: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate additional features to fill remaining dimensions

        Args:
            Various motion features

        Returns:
            additional: (N, 114)  # Changed from 118 to 114
        """
        N = joints.shape[0]
        device = joints.device

        additional = torch.zeros(N, 114, device=device)  # Changed from 118 to 114

        # Fill with normalized versions of available features
        # This helps the VQ-VAE encoder find patterns

        # Repeat joint info
        additional[:, :11] = joints

        # Velocity features
        additional[:, 11:14] = torch.cat([vx, vy, vz], dim=-1)

        # Height and orientation
        additional[:, 14:15] = height
        additional[:, 15:19] = quat

        # Speed-related features
        speed = torch.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        additional[:, 19:20] = speed

        # Normalized joint positions
        joints_norm = joints / (torch.norm(joints, dim=-1, keepdim=True) + 1e-8)
        additional[:, 20:31] = joints_norm

        # The rest can be left as zeros or filled with derived features
        # For now, leave as zeros

        return additional


if __name__ == "__main__":
    # Test the converter
    converter = MuJoCoToHumanML3DConverter()

    # Test with dummy data
    mujoco_features = torch.randn(20, 30)  # 20 frames

    humanml3d_features = converter.convert(mujoco_features)

    print(f"Input shape: {mujoco_features.shape}")
    print(f"Output shape: {humanml3d_features.shape}")
    print(f"Expected shape: (20, 259)")

    assert humanml3d_features.shape == (20, 259), "Shape mismatch!"

    # Test with batch
    batch_mujoco = torch.randn(4, 20, 30)  # 4 sequences of 20 frames
    batch_humanml3d = converter.convert(batch_mujoco)

    print(f"\nBatch input shape: {batch_mujoco.shape}")
    print(f"Batch output shape: {batch_humanml3d.shape}")
    print(f"Expected shape: (4, 20, 259)")

    assert batch_humanml3d.shape == (4, 20, 259), "Batch shape mismatch!"

    print("\nMuJoCo to HumanML3D converter test passed!")
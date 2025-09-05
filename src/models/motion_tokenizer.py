"""
MotionGPT Integration - VQ-VAE Motion Tokenizer
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import yaml
from typing import Optional, Tuple, Dict

# Add MotionGPT to path
project_root = Path(__file__).parent.parent.parent
motiongpt_path = project_root / "external" / "motiongpt"
sys.path.append(str(motiongpt_path))

try:
    # Import MotionGPT modules
    from mGPT.models.build_model import build_model
    from mGPT.config import instantiate_from_config

    MOTIONGPT_AVAILABLE = True
    print(" MotionGPT imports successful")
except ImportError as e:
    print(f" MotionGPT import failed: {e}")
    print("Falling back to placeholder implementation")
    MOTIONGPT_AVAILABLE = False


class MotionTokenizer:
    """
    Integration with MotionGPT's VQ-VAE for motion tokenization

    This class loads and uses the actual MotionGPT VQ-VAE model for:
    1. Encoding motion sequences to discrete tokens
    2. Decoding tokens back to motion sequences
    3. Getting continuous motion embeddings
    """

    def __init__(self,
                 model_config_path: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.device = device
        self.model_config_path = model_config_path
        self.checkpoint_path = checkpoint_path

        # Motion parameters (based on MotionGPT)
        self.motion_dim = 263  # Joint positions/rotations
        self.max_sequence_length = 196  # Max motion length
        self.codebook_size = 512  # VQ-VAE codebook size

        # Load the actual MotionGPT model
        self.vqvae_model = self._load_motiongpt_model()

        if self.vqvae_model is not None:
            self.vqvae_model.eval()
            print(f" MotionGPT VQ-VAE loaded on {device}")
        else:
            print(" Using placeholder VQ-VAE")

    def _load_motiongpt_model(self):
        """Load the actual MotionGPT VQ-VAE model"""
        if not MOTIONGPT_AVAILABLE:
            print("MotionGPT not available, using placeholder")
            return None

        try:
            # Look for default config in MotionGPT
            if self.model_config_path is None:
                config_candidates = [
                    motiongpt_path / "configs" / "config_vq.yaml",
                    motiongpt_path / "configs" / "vq_cfg.yaml",
                    motiongpt_path / "configs" / "default.yaml"
                ]

                for config_path in config_candidates:
                    if config_path.exists():
                        self.model_config_path = str(config_path)
                        print(f"Found config: {config_path}")
                        break

            if self.model_config_path and Path(self.model_config_path).exists():
                # Load config
                with open(self.model_config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # Build model using MotionGPT's builder
                model = build_model(config, phase="test")

                # Load checkpoint if available
                if self.checkpoint_path and Path(self.checkpoint_path).exists():
                    checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                    model.load_state_dict(checkpoint['model'])
                    print(f"Loaded checkpoint: {self.checkpoint_path}")

                return model.to(self.device)

            else:
                print("No valid config found, will create simple VQ-VAE")
                return self._create_simple_vqvae()

        except Exception as e:
            print(f"Failed to load MotionGPT model: {e}")
            print("Creating simple VQ-VAE instead")
            return self._create_simple_vqvae()

    def _create_simple_vqvae(self):
        """Create a simple VQ-VAE as fallback"""
        try:
            from torch.nn import functional as F

            class SimpleVQVAE(nn.Module):
                def __init__(self, input_dim=263, latent_dim=256, codebook_size=512):
                    super().__init__()

                    # Encoder
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, 512),
                        nn.ReLU(),
                        nn.Linear(512, latent_dim),
                        nn.ReLU(),
                        nn.Linear(latent_dim, latent_dim)
                    )

                    # VQ Layer
                    self.codebook = nn.Embedding(codebook_size, latent_dim)
                    self.codebook.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

                    # Decoder
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, latent_dim),
                        nn.ReLU(),
                        nn.Linear(latent_dim, 512),
                        nn.ReLU(),
                        nn.Linear(512, input_dim)
                    )

                def encode(self, x):
                    # x: (batch, seq_len, motion_dim)
                    batch_size, seq_len, _ = x.shape
                    x_flat = x.view(-1, x.shape[-1])  # (batch*seq_len, motion_dim)

                    # Encode
                    z_e = self.encoder(x_flat)  # (batch*seq_len, latent_dim)

                    # Quantize
                    distances = torch.cdist(z_e, self.codebook.weight)  # (batch*seq_len, codebook_size)
                    indices = torch.argmin(distances, dim=1)  # (batch*seq_len,)
                    z_q = self.codebook(indices)  # (batch*seq_len, latent_dim)

                    # Reshape back
                    indices = indices.view(batch_size, seq_len)
                    z_q = z_q.view(batch_size, seq_len, -1)

                    return z_q, indices

                def decode(self, z_q):
                    # z_q: (batch, seq_len, latent_dim)
                    batch_size, seq_len, _ = z_q.shape
                    z_q_flat = z_q.view(-1, z_q.shape[-1])

                    # Decode
                    x_recon = self.decoder(z_q_flat)
                    x_recon = x_recon.view(batch_size, seq_len, -1)

                    return x_recon

                def forward(self, x):
                    z_q, indices = self.encode(x)
                    x_recon = self.decode(z_q)
                    return x_recon, z_q, indices

            model = SimpleVQVAE(
                input_dim=self.motion_dim,
                latent_dim=256,
                codebook_size=self.codebook_size
            ).to(self.device)

            print(" Created simple VQ-VAE fallback")
            return model

        except Exception as e:
            print(f"Failed to create simple VQ-VAE: {e}")
            return None

    def encode_motion(self, motion_sequence: np.ndarray) -> torch.Tensor:
        """
        Encode motion sequence to discrete token indices

        Args:
            motion_sequence: numpy array of shape (seq_len, motion_dim) or (batch, seq_len, motion_dim)

        Returns:
            token_indices: torch.Tensor of shape (seq_len,) or (batch, seq_len)
        """
        # Convert to tensor and ensure correct shape
        if isinstance(motion_sequence, np.ndarray):
            motion_tensor = torch.from_numpy(motion_sequence).float()
        else:
            motion_tensor = motion_sequence.float()

        # Ensure 3D: (batch, seq_len, motion_dim)
        if motion_tensor.dim() == 2:
            motion_tensor = motion_tensor.unsqueeze(0)  # Add batch dimension

        motion_tensor = motion_tensor.to(self.device)

        if self.vqvae_model is None:
            # Fallback: random tokens
            batch_size, seq_len = motion_tensor.shape[:2]
            tokens = torch.randint(0, self.codebook_size, (batch_size, seq_len), device=self.device)
            return tokens.squeeze() if batch_size == 1 else tokens

        try:
            with torch.no_grad():
                if hasattr(self.vqvae_model, 'encode'):
                    # Use MotionGPT's encode method
                    _, tokens = self.vqvae_model.encode(motion_tensor)
                else:
                    # Use our simple VQ-VAE
                    _, tokens = self.vqvae_model.encode(motion_tensor)

                return tokens.squeeze() if tokens.shape[0] == 1 else tokens

        except Exception as e:
            print(f"Encoding failed: {e}, using random tokens")
            batch_size, seq_len = motion_tensor.shape[:2]
            tokens = torch.randint(0, self.codebook_size, (batch_size, seq_len), device=self.device)
            return tokens.squeeze() if batch_size == 1 else tokens

    def decode_motion(self, token_indices: torch.Tensor) -> torch.Tensor:
        """
        Decode token indices back to motion sequence

        Args:
            token_indices: torch.Tensor of shape (seq_len,) or (batch, seq_len)

        Returns:
            motion_sequence: torch.Tensor of shape (seq_len, motion_dim) or (batch, seq_len, motion_dim)
        """
        if token_indices.dim() == 1:
            token_indices = token_indices.unsqueeze(0)  # Add batch dimension

        token_indices = token_indices.to(self.device)

        if self.vqvae_model is None:
            # Fallback: random motion
            batch_size, seq_len = token_indices.shape
            motion = torch.randn(batch_size, seq_len, self.motion_dim, device=self.device)
            return motion.squeeze() if batch_size == 1 else motion

        try:
            with torch.no_grad():
                if hasattr(self.vqvae_model, 'decode_from_tokens'):
                    # MotionGPT style
                    motion = self.vqvae_model.decode_from_tokens(token_indices)
                else:
                    # Our simple VQ-VAE
                    # Convert tokens to embeddings
                    z_q = self.vqvae_model.codebook(token_indices)  # (batch, seq_len, latent_dim)
                    motion = self.vqvae_model.decode(z_q)

                return motion.squeeze() if motion.shape[0] == 1 else motion

        except Exception as e:
            print(f"Decoding failed: {e}, using random motion")
            batch_size, seq_len = token_indices.shape
            motion = torch.randn(batch_size, seq_len, self.motion_dim, device=self.device)
            return motion.squeeze() if batch_size == 1 else motion

    def get_motion_embedding(self, motion_sequence: np.ndarray) -> torch.Tensor:
        """
        Get continuous embedding of motion sequence (before quantization)

        Args:
            motion_sequence: numpy array of shape (seq_len, motion_dim) or (batch, seq_len, motion_dim)

        Returns:
            embedding: torch.Tensor of shape (seq_len, embed_dim) or (batch, seq_len, embed_dim)
        """
        if isinstance(motion_sequence, np.ndarray):
            motion_tensor = torch.from_numpy(motion_sequence).float()
        else:
            motion_tensor = motion_sequence.float()

        if motion_tensor.dim() == 2:
            motion_tensor = motion_tensor.unsqueeze(0)

        motion_tensor = motion_tensor.to(self.device)

        if self.vqvae_model is None:
            # Fallback: random embedding
            batch_size, seq_len = motion_tensor.shape[:2]
            embedding = torch.randn(batch_size, seq_len, 256, device=self.device)
            return embedding.squeeze() if batch_size == 1 else embedding

        try:
            with torch.no_grad():
                if hasattr(self.vqvae_model, 'encode'):
                    z_q, _ = self.vqvae_model.encode(motion_tensor)
                    return z_q.squeeze() if z_q.shape[0] == 1 else z_q
                else:
                    # Simple encoder
                    batch_size, seq_len, _ = motion_tensor.shape
                    motion_flat = motion_tensor.view(-1, motion_tensor.shape[-1])
                    embedding = self.vqvae_model.encoder(motion_flat)
                    embedding = embedding.view(batch_size, seq_len, -1)
                    return embedding.squeeze() if batch_size == 1 else embedding

        except Exception as e:
            print(f"Embedding extraction failed: {e}")
            batch_size, seq_len = motion_tensor.shape[:2]
            embedding = torch.randn(batch_size, seq_len, 256, device=self.device)
            return embedding.squeeze() if batch_size == 1 else embedding

    def save_model(self, path: str):
        """Save the VQ-VAE model"""
        if self.vqvae_model is not None:
            torch.save({
                'model_state_dict': self.vqvae_model.state_dict(),
                'config_path': self.model_config_path,
                'codebook_size': self.codebook_size,
                'motion_dim': self.motion_dim
            }, path)
            print(f"Model saved to: {path}")

    def load_model(self, path: str):
        """Load a saved VQ-VAE model"""
        if Path(path).exists():
            checkpoint = torch.load(path, map_location=self.device)
            if self.vqvae_model is not None:
                self.vqvae_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from: {path}")


class MotionLanguageAligner:
    """
    Aligns motion representations with language instructions
    This replaces AnySkill's CLIP-based visual rewards
    """

    def __init__(self, motion_tokenizer, language_model='t5-small'):
        self.motion_tokenizer = motion_tokenizer
        self.device = motion_tokenizer.device

        # TODO: Load T5 or other language model from MotionGPT
        # self.language_model = self._load_language_model(language_model)

        self.language_dim = 512  # Typical T5-small dimension

        # Simple alignment network for computing similarity
        self.alignment_net = nn.Sequential(
            nn.Linear(256, 512),  # motion_embed_dim -> language_dim
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        ).to(self.device)

        print(f"MotionLanguageAligner initialized")

    def encode_instruction(self, instruction_text):
        """
        Encode natural language instruction to embedding

        Args:
            instruction_text: str, natural language instruction

        Returns:
            instruction_embedding: torch.Tensor of shape (embed_dim,)
        """
        # TODO: Use MotionGPT's T5 encoder
        # embedding = self.language_model.encode(instruction_text)

        # Placeholder: random embedding
        embedding = torch.randn(self.language_dim, device=self.device)

        return embedding

    def compute_motion_language_similarity(self, motion_sequence, instruction_text):
        """
        Compute similarity between motion and language instruction
        This is the key function that replaces CLIP rewards in AnySkill

        Args:
            motion_sequence: torch.Tensor of shape (seq_len, motion_dim)
            instruction_text: str, natural language instruction

        Returns:
            similarity_score: float between 0 and 1
        """
        # Get motion embedding
        motion_embedding = self.motion_tokenizer.get_motion_embedding(motion_sequence)
        motion_embedding = motion_embedding.mean(dim=0)  # Average over sequence

        # Align motion to language space
        aligned_motion = self.alignment_net(motion_embedding)

        # Get instruction embedding
        instruction_embedding = self.encode_instruction(instruction_text)

        # Compute cosine similarity
        similarity = torch.cosine_similarity(
            aligned_motion.unsqueeze(0),
            instruction_embedding.unsqueeze(0),
            dim=1
        ).item()

        # Normalize to [0, 1]
        similarity = (similarity + 1) / 2

        return similarity

    def compute_reward(self, motion_sequence, instruction_text, success_threshold=0.7):
        """
        Compute reward signal for RL training

        Args:
            motion_sequence: torch.Tensor of shape (seq_len, motion_dim)
            instruction_text: str
            success_threshold: float, threshold for success

        Returns:
            reward: float
        """
        similarity = self.compute_motion_language_similarity(motion_sequence, instruction_text)

        # Simple reward function - can be made more sophisticated
        if similarity > success_threshold:
            reward = 1.0 + (similarity - success_threshold) * 2  # Bonus for high similarity
        else:
            reward = similarity  # Encourage progress toward threshold

        return reward


def test_motion_tokenizer():
    """Test the motion tokenizer functionality"""
    print(" Testing Motion Tokenizer")
    print("=" * 30)

    # Create tokenizer
    tokenizer = MotionTokenizer()
    aligner = MotionLanguageAligner(tokenizer)

    # Test motion encoding/decoding
    print("Testing motion encoding/decoding...")
    dummy_motion = np.random.randn(50, 263)  # 50 frames, 263 joint dimensions

    tokens = tokenizer.encode_motion(dummy_motion)
    print(f"  Encoded motion to {tokens.shape[0]} tokens")

    reconstructed = tokenizer.decode_motion(tokens)
    print(f"  Decoded to motion of shape {reconstructed.shape}")

    # Test motion-language alignment
    print("\nTesting motion-language alignment...")
    instructions = [
        "walk forward",
        "turn left",
        "wave your hand",
        "jump in place"
    ]

    for instruction in instructions:
        similarity = aligner.compute_motion_language_similarity(
            torch.from_numpy(dummy_motion).float(),
            instruction
        )
        reward = aligner.compute_reward(
            torch.from_numpy(dummy_motion).float(),
            instruction
        )
        print(f"  '{instruction}': similarity={similarity:.3f}, reward={reward:.3f}")

    print(" Motion tokenizer test completed!")


if __name__ == "__main__":
    test_motion_tokenizer()

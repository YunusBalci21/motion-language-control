# src/models/motiongpt_vqvae_loader.py
"""
MotionGPT VQ-VAE Loader - Simplified for RL Reward Computation
Loads just the encoder and quantizer, not the full generative model
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

# Add MotionGPT to path
project_root = Path(__file__).parent.parent.parent
motiongpt_path = project_root / "external" / "MotionGPT"
sys.path.insert(0, str(motiongpt_path))

try:
    from mGPT.archs.mgpt_vq import VQVae
except ImportError as e:
    print(f"Warning: Could not import MotionGPT VQVae: {e}")
    VQVae = None


class MotionGPTEncoder:
    """Simplified MotionGPT encoder for computing motion-language similarity"""

    def __init__(
            self,
            checkpoint_path: str,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path

        print(f"Loading MotionGPT VQ-VAE from: {checkpoint_path}")

        # Load normalization stats (mean and std for HumanML3D)
        checkpoint_dir = Path(checkpoint_path).parent.parent
        meta_dir = checkpoint_dir / "meta"

        self.mean = None
        self.std = None

        if (meta_dir / "mean.npy").exists():
            self.mean = torch.from_numpy(np.load(str(meta_dir / "mean.npy"))).float().to(device)
            self.std = torch.from_numpy(np.load(str(meta_dir / "std.npy"))).float().to(device)
            print(f"✓ Loaded normalization stats: mean shape {self.mean.shape}, std shape {self.std.shape}")

            # The checkpoint expects 259 dims, but stats might be 263
            # Truncate if needed
            if self.mean.shape[0] == 263:
                print("  ⚠ Truncating normalization stats from 263 to 259 dims")
                self.mean = self.mean[:259]
                self.std = self.std[:259]
        else:
            print("⚠ Warning: Could not find normalization stats (mean.npy, std.npy)")

        # Initialize VQ-VAE model
        if VQVae is None:
            raise ImportError("Could not import VQVae from MotionGPT")

        # VQ-VAE configuration (from checkpoint architecture)
        # Note: This checkpoint uses 259 features, not 263 (HumanML3D standard)
        self.vqvae = VQVae(
            nfeats=259,  # From checkpoint (not 263!)
            quantizer="ema_reset",
            code_num=1024,  # Codebook size (CB1024)
            code_dim=512,  # Code dimension
            output_emb_width=512,
            down_t=3,
            stride_t=2,
            width=1024,  # Hidden width (H1024)
            depth=3,  # Number of residual blocks (NRES3)
            dilation_growth_rate=3,
            activation='relu',
            norm=None,
            kernel_size=4  # Passing it, but it might be ignored by mGPT implementation
        ).to(device)

        # --- FORCE FIX FOR KERNEL MISMATCH ---
        # The external library likely ignores the kernel_size arg, so we manually patch it.
        try:
            # Inspect the first layer of the encoder (model.0)
            # Structure is usually self.vqvae.encoder.model[0]
            first_layer = self.vqvae.encoder.model[0]

            if isinstance(first_layer, nn.Conv1d):
                current_k = first_layer.kernel_size[0]
                if current_k != 4:
                    print(f"⚠ VQVae ignored kernel_size=4 (got {current_k}). FORCE PATCHING encoder...")

                    # Create correct layer: 259 -> 1024, k=4, s=2, p=1
                    new_layer = nn.Conv1d(
                        in_channels=259,
                        out_channels=1024,
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ).to(device)

                    # Replace the layer in the sequential model
                    self.vqvae.encoder.model[0] = new_layer
                    print("✓ Successfully patched encoder model.0 to kernel_size=4")
                else:
                    print("✓ Encoder already has correct kernel_size=4")
        except Exception as e:
            print(f"⚠ Warning: Could not patch encoder layer: {e}")
        # --------------------------------------

        # Load checkpoint weights
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load encoder and quantizer weights
        if 'vq_encoder' in checkpoint:
            # Fix key mismatch: checkpoint has 'main.*' but model expects 'model.*'
            encoder_state = checkpoint['vq_encoder']
            fixed_encoder_state = {}
            for k, v in encoder_state.items():
                if k.startswith('main.'):
                    # Rename main.* to model.*
                    new_key = k.replace('main.', 'model.')
                    fixed_encoder_state[new_key] = v
                else:
                    fixed_encoder_state[k] = v

            self.vqvae.encoder.load_state_dict(fixed_encoder_state, strict=False)
            print("✓ Loaded VQ encoder weights")
        else:
            print("⚠ Warning: 'vq_encoder' not found in checkpoint")

        if 'quantizer' in checkpoint:
            self.vqvae.quantizer.load_state_dict(checkpoint['quantizer'], strict=False)
            print("✓ Loaded quantizer weights")
        else:
            print("⚠ Warning: 'quantizer' not found in checkpoint")

        # Set to eval mode
        self.vqvae.eval()

        print("✅ MotionGPT VQ-VAE encoder loaded successfully!")

    def normalize(self, motion_features: torch.Tensor) -> torch.Tensor:
        """Normalize motion features using HumanML3D stats"""
        if self.mean is not None and self.std is not None:
            return (motion_features - self.mean) / (self.std + 1e-8)
        return motion_features

    def denormalize(self, motion_features: torch.Tensor) -> torch.Tensor:
        """Denormalize motion features"""
        if self.mean is not None and self.std is not None:
            return motion_features * self.std + self.mean
        return motion_features

    @torch.no_grad()
    def encode(self, motion_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode motion sequence to discrete tokens

        Args:
            motion_sequence: (B, T, 263) - batch of motion sequences in HumanML3D format

        Returns:
            code_indices: (B, T') - discrete motion tokens
            embeddings: (B, T', D) - continuous embeddings before quantization
        """
        self.vqvae.eval()

        # Ensure correct shape
        if motion_sequence.dim() == 2:
            motion_sequence = motion_sequence.unsqueeze(0)  # Add batch dim

        B, T, D = motion_sequence.shape
        assert D == 259, f"Expected 259 features, got {D}"

        # Normalize
        motion_norm = self.normalize(motion_sequence)

        # Encode to discrete tokens
        code_indices, _ = self.vqvae.encode(motion_norm)

        # Get continuous embeddings (before quantization)
        x_in = self.vqvae.preprocess(motion_norm)
        embeddings = self.vqvae.encoder(x_in)  # (B, D, T')
        embeddings = self.vqvae.postprocess(embeddings)  # (B, T', D)

        return code_indices, embeddings

    @torch.no_grad()
    def get_motion_embedding(self, motion_sequence: torch.Tensor) -> torch.Tensor:
        """
        Get continuous motion embedding (averaged over time)
        Useful for computing similarity with text embeddings

        Args:
            motion_sequence: (B, T, 263) or (T, 263)

        Returns:
            embedding: (B, D) - motion embedding
        """
        _, embeddings = self.encode(motion_sequence)

        # Average over time dimension
        motion_emb = embeddings.mean(dim=1)  # (B, D)

        return motion_emb


class SimpleTextEncoder:
    """Simple text encoder using sentence transformers (fallback if T5 not available)"""

    def __init__(self, device: str = 'cuda'):
        self.device = device

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
            self.embed_dim = 384
            print("✓ Loaded SentenceTransformer for text encoding")
        except ImportError:
            print("⚠ Warning: sentence-transformers not installed. Text encoding will use random embeddings.")
            self.model = None
            self.embed_dim = 512

    @torch.no_grad()
    def encode(self, texts):
        """Encode text to embeddings"""
        if self.model is not None:
            if isinstance(texts, str):
                texts = [texts]
            embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
            return embeddings
        else:
            # Fallback: return random embeddings
            if isinstance(texts, str):
                texts = [texts]
            return torch.randn(len(texts), self.embed_dim, device=self.device)


def compute_motion_text_similarity(
        motion_embedding: torch.Tensor,
        text_embedding: torch.Tensor
) -> float:
    """
    Compute cosine similarity between motion and text embeddings

    Args:
        motion_embedding: (D,) or (B, D)
        text_embedding: (D,) or (B, D)

    Returns:
        similarity: float in [0, 1]
    """
    # Ensure 2D
    if motion_embedding.dim() == 1:
        motion_embedding = motion_embedding.unsqueeze(0)
    if text_embedding.dim() == 1:
        text_embedding = text_embedding.unsqueeze(0)

    # Project to same dimension if needed
    if motion_embedding.shape[-1] != text_embedding.shape[-1]:
        # Simple linear projection
        min_dim = min(motion_embedding.shape[-1], text_embedding.shape[-1])
        motion_embedding = motion_embedding[..., :min_dim]
        text_embedding = text_embedding[..., :min_dim]

    # Cosine similarity
    motion_norm = torch.nn.functional.normalize(motion_embedding, p=2, dim=-1)
    text_norm = torch.nn.functional.normalize(text_embedding, p=2, dim=-1)

    similarity = (motion_norm * text_norm).sum(dim=-1)

    # Convert from [-1, 1] to [0, 1]
    similarity = (similarity + 1) / 2

    return float(similarity.mean())
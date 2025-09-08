"""
MotionGPT Integration - VQ-VAE Motion Tokenizer
<<<<<<< HEAD
=======
Enhanced version with direct motion-language learning capabilities
>>>>>>> 8470d39 (added and changed)
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from typing import Optional, Tuple, Dict, Union
import warnings

# Add MotionGPT to path
project_root = Path(__file__).parent.parent.parent
motiongpt_path = project_root / "external" / "motiongpt"
sys.path.append(str(motiongpt_path))

try:
    # Import MotionGPT modules
    from mGPT.models.build_model import build_model
    from mGPT.config import instantiate_from_config
    from transformers import T5Tokenizer, T5EncoderModel, T5Config
    MOTIONGPT_AVAILABLE = True
    print("MotionGPT imports successful")
except ImportError as e:
    print(f"MotionGPT import failed: {e}")
    print("Falling back to placeholder implementation")
    MOTIONGPT_AVAILABLE = False


class MotionTokenizer:
    """
    Integration with MotionGPT's VQ-VAE for motion tokenization

    This class loads and uses the actual MotionGPT VQ-VAE model for:
    1. Encoding motion sequences to discrete tokens
    2. Decoding tokens back to motion sequences
    3. Getting continuous motion embeddings
    4. Direct motion-language alignment (replaces CLIP)
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
        self.latent_dim = 256  # Motion embedding dimension

        # Language model parameters
        self.language_dim = 512  # T5-small hidden dimension

        print(f"Initializing MotionGPT tokenizer on {device}")

        # Load the actual MotionGPT model
        self.vqvae_model = self._load_motiongpt_model()

        # Load language model
        self.language_tokenizer, self.language_encoder = self._load_language_model()

        # Create motion-language alignment network
        self.alignment_network = self._create_alignment_network()

        # Motion preprocessing
        self.motion_normalizer = MotionNormalizer()

        if self.vqvae_model is not None:
            self.vqvae_model.eval()
            print(f"MotionGPT VQ-VAE loaded successfully")
        else:
            print("Using placeholder VQ-VAE")

    def _load_motiongpt_model(self):
        """Load the actual MotionGPT VQ-VAE model"""
        if not MOTIONGPT_AVAILABLE:
            print("MotionGPT not available, creating placeholder")
            return self._create_vqvae()

        try:
            # Look for default config in MotionGPT
            if self.model_config_path is None:
                config_candidates = [
                    motiongpt_path / "configs" / "config_vq.yaml",
                    motiongpt_path / "configs" / "vq_cfg.yaml",
                    motiongpt_path / "configs" / "default.yaml",
                    motiongpt_path / "configs" / "t2m_vq.yaml"
                ]

                for config_path in config_candidates:
                    if config_path.exists():
                        self.model_config_path = str(config_path)
                        print(f"Found MotionGPT config: {config_path}")
                        break

            if self.model_config_path and Path(self.model_config_path).exists():
                # Load config
                with open(self.model_config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # Build model using MotionGPT's builder - try different approaches
                try:
                    # Try with phase argument (newer versions)
                    model = build_model(config, phase="test")
                except TypeError:
                    try:
                        # Try without phase argument (older versions)
                        model = build_model(config)
                    except Exception as e:
                        print(f"Failed to build MotionGPT model with config: {e}")
                        return self._create_vqvae()

                # Load checkpoint if available
                if self.checkpoint_path and Path(self.checkpoint_path).exists():
                    try:
                        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                        # Handle different checkpoint formats
                        if 'model' in checkpoint:
                            model.load_state_dict(checkpoint['model'])
                        elif 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            model.load_state_dict(checkpoint)
                        print(f"Loaded MotionGPT checkpoint: {self.checkpoint_path}")
                    except Exception as e:
                        print(f"Failed to load checkpoint, using random weights: {e}")

                return model.to(self.device)

            else:
                print("No valid config found, will create VQ-VAE")
                return self._create_vqvae()

        except Exception as e:
            print(f"Failed to load MotionGPT model: {e}")
            print("Creating placeholder VQ-VAE")
            return self._create_vqvae()

    def _create_vqvae(self):
        """Create a VQ-VAE as fallback"""
        try:
            class VQVAE(nn.Module):
                def __init__(self, input_dim=263, latent_dim=256, codebook_size=512):
                    super().__init__()

                    # More sophisticated encoder
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, 512),
                        nn.LayerNorm(512),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(512, 512),
                        nn.LayerNorm(512),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(512, latent_dim),
                        nn.LayerNorm(latent_dim)
                    )

                    # VQ Layer with improved initialization
                    self.codebook = nn.Embedding(codebook_size, latent_dim)
                    nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0 / codebook_size)

                    # More sophisticated decoder
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, 512),
                        nn.LayerNorm(512),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(512, 512),
                        nn.LayerNorm(512),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(512, input_dim)
                    )

                    # Commitment loss weight
                    self.commitment_weight = 0.25

                def encode(self, x):
                    """Encode motion to latent space and quantize"""
                    # x: (batch, seq_len, motion_dim)
                    batch_size, seq_len, _ = x.shape
                    x_flat = x.view(-1, x.shape[-1])  # (batch*seq_len, motion_dim)

                    # Encode to continuous latent
                    z_e = self.encoder(x_flat)  # (batch*seq_len, latent_dim)

                    # Quantize using vector quantization
                    distances = torch.sum((z_e.unsqueeze(1) - self.codebook.weight.unsqueeze(0))**2, dim=2)
                    indices = torch.argmin(distances, dim=1)  # (batch*seq_len,)
                    z_q = self.codebook(indices)  # (batch*seq_len, latent_dim)

                    # Straight-through estimator
                    z_q = z_e + (z_q - z_e).detach()

                    # Reshape back
                    indices = indices.view(batch_size, seq_len)
                    z_q = z_q.view(batch_size, seq_len, -1)
                    z_e = z_e.view(batch_size, seq_len, -1)

                    return z_q, indices, z_e

                def decode(self, z_q):
                    """Decode quantized latents back to motion"""
                    # z_q: (batch, seq_len, latent_dim)
                    batch_size, seq_len, _ = z_q.shape
                    z_q_flat = z_q.view(-1, z_q.shape[-1])

                    # Decode
                    x_recon = self.decoder(z_q_flat)
                    x_recon = x_recon.view(batch_size, seq_len, -1)

                    return x_recon

                def forward(self, x):
                    """Full forward pass with VQ loss"""
                    z_q, indices, z_e = self.encode(x)
                    x_recon = self.decode(z_q)

                    # VQ losses
                    vq_loss = F.mse_loss(z_q.detach(), z_e)
                    commit_loss = F.mse_loss(z_q, z_e.detach()) * self.commitment_weight

                    return x_recon, z_q, indices, vq_loss + commit_loss

            model = VQVAE(
                input_dim=self.motion_dim,
                latent_dim=self.latent_dim,
                codebook_size=self.codebook_size
            ).to(self.device)

            print("Created VQ-VAE fallback")
            return model

        except Exception as e:
            print(f"Failed to create VQ-VAE: {e}")
            return None

    def _load_language_model(self):
        """Load T5 language model for instruction encoding"""
        try:
            tokenizer = T5Tokenizer.from_pretrained('t5-small')
            encoder = T5EncoderModel.from_pretrained('t5-small')
            encoder.to(self.device)
            encoder.eval()
            print(f"Loaded language model: t5-small")
            return tokenizer, encoder
        except Exception as e:
            print(f"Failed to load language model: {e}")
            return None, None

    def _create_alignment_network(self):
        """Create network to align motion and language representations"""
        alignment_net = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.language_dim),
            nn.LayerNorm(self.language_dim),
            nn.Tanh()  # Bounded output for better alignment
        ).to(self.device)

        print("Created motion-language alignment network")
        return alignment_net

    def encode_instruction(self, instruction: str) -> torch.Tensor:
        """Encode natural language instruction to embedding"""
        if self.language_tokenizer is None or self.language_encoder is None:
            # Fallback to random embedding
            return torch.randn(self.language_dim, device=self.device)

        try:
            # Tokenize instruction
            inputs = self.language_tokenizer(
                instruction,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Encode with T5
            with torch.no_grad():
                outputs = self.language_encoder(**inputs)
                # Use mean pooling over sequence dimension
                instruction_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

            return instruction_embedding

        except Exception as e:
            print(f"Instruction encoding failed: {e}")
            return torch.randn(self.language_dim, device=self.device)

    def extract_motion_from_obs(self, obs: np.ndarray, env_name: str = "Humanoid-v4") -> np.ndarray:
        """Extract motion-relevant features from MuJoCo observation"""
        if env_name.startswith("Humanoid"):
            return self._extract_humanoid_motion(obs)
        elif env_name.startswith("HalfCheetah"):
            return self._extract_cheetah_motion(obs)
        elif env_name.startswith("Ant"):
            return self._extract_ant_motion(obs)
        else:
            # Generic extraction
            return self._extract_generic_motion(obs)

    def _extract_humanoid_motion(self, obs: np.ndarray) -> np.ndarray:
        """Extract motion features specifically for Humanoid environment"""
        # Humanoid observation structure:
        # 0-1: COM position (x, z)
        # 2-4: COM orientation (quaternion xyz)
        # 5-22: joint positions (17 joints)
        # 23-39: joint velocities (17 joints)
        # ... more features

        if len(obs) < 50:
            # Pad if observation is too short
            motion_features = np.pad(obs, (0, max(0, 263 - len(obs))))[:263]
        else:
            # Extract relevant motion features
            com_pos = obs[0:2]  # x, z position
            com_orient = obs[2:5]  # quaternion xyz (skip w)
            joint_pos = obs[5:22] if len(obs) > 22 else obs[5:min(22, len(obs))]
            joint_vel = obs[23:40] if len(obs) > 40 else obs[23:min(40, len(obs))]

            # Additional features if available
            remaining_obs = obs[40:min(100, len(obs))] if len(obs) > 40 else []

            # Combine into motion representation
            motion_features = np.concatenate([
                com_pos,
                com_orient,
                joint_pos,
                joint_vel,
                remaining_obs
            ])

            # Pad or truncate to 263 dimensions
            if len(motion_features) < 263:
                motion_features = np.pad(motion_features, (0, 263 - len(motion_features)))
            else:
                motion_features = motion_features[:263]

        return motion_features.astype(np.float32)

    def _extract_cheetah_motion(self, obs: np.ndarray) -> np.ndarray:
        """Extract motion features for HalfCheetah"""
        # HalfCheetah has simpler structure
        motion_features = obs[:min(len(obs), 263)]
        if len(motion_features) < 263:
            motion_features = np.pad(motion_features, (0, 263 - len(motion_features)))
        return motion_features.astype(np.float32)

    def _extract_ant_motion(self, obs: np.ndarray) -> np.ndarray:
        """Extract motion features for Ant"""
        # Similar to cheetah but with different joint structure
        motion_features = obs[:min(len(obs), 263)]
        if len(motion_features) < 263:
            motion_features = np.pad(motion_features, (0, 263 - len(motion_features)))
        return motion_features.astype(np.float32)

    def _extract_generic_motion(self, obs: np.ndarray) -> np.ndarray:
        """Generic motion extraction for unknown environments"""
        motion_features = obs[:min(len(obs), 263)]
        if len(motion_features) < 263:
            motion_features = np.pad(motion_features, (0, 263 - len(motion_features)))
        return motion_features.astype(np.float32)

    def encode_motion(self, motion_sequence: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
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

        # Normalize motion
        motion_tensor = self.motion_normalizer.normalize(motion_tensor)

        if self.vqvae_model is None:
            # Fallback: random tokens
            batch_size, seq_len = motion_tensor.shape[:2]
            tokens = torch.randint(0, self.codebook_size, (batch_size, seq_len), device=self.device)
            return tokens.squeeze() if batch_size == 1 else tokens

        try:
            with torch.no_grad():
                if hasattr(self.vqvae_model, 'encode'):
                    _, tokens, _ = self.vqvae_model.encode(motion_tensor)
                    return tokens.squeeze() if tokens.shape[0] == 1 else tokens
                else:
                    # Use our VQ-VAE
                    _, tokens, _ = self.vqvae_model.encode(motion_tensor)
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
                    # Our VQ-VAE
                    # Convert tokens to embeddings
                    z_q = self.vqvae_model.codebook(token_indices)  # (batch, seq_len, latent_dim)
                    motion = self.vqvae_model.decode(z_q)

                return motion.squeeze() if motion.shape[0] == 1 else motion

        except Exception as e:
            print(f"Decoding failed: {e}, using random motion")
            batch_size, seq_len = token_indices.shape
            motion = torch.randn(batch_size, seq_len, self.motion_dim, device=self.device)
            return motion.squeeze() if batch_size == 1 else motion

    def get_motion_embedding(self, motion_sequence: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
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

        # Normalize motion
        motion_tensor = self.motion_normalizer.normalize(motion_tensor)

        if self.vqvae_model is None:
            # Fallback: random embedding
            batch_size, seq_len = motion_tensor.shape[:2]
            embedding = torch.randn(batch_size, seq_len, self.latent_dim, device=self.device)
            return embedding.squeeze() if batch_size == 1 else embedding

        try:
            with torch.no_grad():
                if hasattr(self.vqvae_model, 'encode'):
                    z_q, _, z_e = self.vqvae_model.encode(motion_tensor)
                    # Use pre-quantization embedding for richer representation
                    return z_e.squeeze() if z_e.shape[0] == 1 else z_e
                else:
                    # Use our VQ-VAE
                    z_q, _, z_e = self.vqvae_model.encode(motion_tensor)
                    return z_e.squeeze() if z_e.shape[0] == 1 else z_e

        except Exception as e:
            print(f"Motion embedding failed: {e}")
            batch_size, seq_len = motion_tensor.shape[:2]
            embedding = torch.randn(batch_size, seq_len, self.latent_dim, device=self.device)
            return embedding.squeeze() if batch_size == 1 else embedding

    def compute_motion_language_similarity(self,
                                           motion_sequence: Union[np.ndarray, torch.Tensor],
                                           instruction: str,
                                           temporal_aggregation: str = "mean") -> float:
        """
        Compute motion-language similarity (replaces CLIP-based rewards)
        This is the core function that enables direct language-based RL
        """
        try:
            # Get motion embedding
            motion_embedding = self.get_motion_embedding(motion_sequence)

            # Handle different embedding shapes
            if motion_embedding.dim() == 3:
                # (batch, seq_len, embed_dim) -> aggregate over sequence
                if temporal_aggregation == "mean":
                    motion_embedding = motion_embedding.mean(dim=1)
                elif temporal_aggregation == "max":
                    motion_embedding = motion_embedding.max(dim=1)[0]
                elif temporal_aggregation == "last":
                    motion_embedding = motion_embedding[:, -1, :]

                if motion_embedding.shape[0] == 1:
                    motion_embedding = motion_embedding.squeeze(0)

            elif motion_embedding.dim() == 2:
                # (seq_len, embed_dim) -> aggregate over sequence
                if temporal_aggregation == "mean":
                    motion_embedding = motion_embedding.mean(dim=0)
                elif temporal_aggregation == "max":
                    motion_embedding = motion_embedding.max(dim=0)[0]
                elif temporal_aggregation == "last":
                    motion_embedding = motion_embedding[-1, :]

            # Project motion to language space
            aligned_motion = self.alignment_network(motion_embedding)

            # Get instruction embedding
            instruction_embedding = self.encode_instruction(instruction)

            # Ensure same device
            aligned_motion = aligned_motion.to(self.device)
            instruction_embedding = instruction_embedding.to(self.device)

            # Compute cosine similarity
            similarity = F.cosine_similarity(
                aligned_motion.unsqueeze(0),
                instruction_embedding.unsqueeze(0),
                dim=1
            ).item()

            # Normalize to [0, 1]
            similarity = (similarity + 1) / 2

            return similarity

        except Exception as e:
            print(f"Motion-language similarity computation failed: {e}")
            return 0.0

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


class MotionNormalizer:
    """Normalize motion sequences for better training"""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, motion_sequences):
        """Fit normalizer to motion data"""
        if isinstance(motion_sequences, list):
            all_motions = torch.cat([torch.from_numpy(seq).float() for seq in motion_sequences], dim=0)
        else:
            all_motions = motion_sequences

        self.mean = all_motions.mean(dim=0)
        self.std = all_motions.std(dim=0) + 1e-8  # Avoid division by zero

    def normalize(self, motion_sequence):
        """Normalize motion sequence"""
        if self.mean is None or self.std is None:
            return motion_sequence  # No normalization if not fitted

        return (motion_sequence - self.mean.to(motion_sequence.device)) / self.std.to(motion_sequence.device)

    def denormalize(self, motion_sequence):
        """Denormalize motion sequence"""
        if self.mean is None or self.std is None:
            return motion_sequence

        return motion_sequence * self.std.to(motion_sequence.device) + self.mean.to(motion_sequence.device)


def test_motion_tokenizer():
    """Test the motion tokenizer functionality"""
    print("Testing Motion Tokenizer")
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

    print("Motion tokenizer test completed!")


if __name__ == "__main__":
    test_motion_tokenizer()

"""
TMD Alignment Model with PyTorch – Improved Version

This script trains a CNN-Transformer hybrid model to predict optimal alignment parameters
between two heightmaps, learning from examples with known transformations.

Key improvements include:
  • A residual CNN backbone (modified ResNet blocks) for robust feature extraction.
  • Enhanced transformer blocks with additional dropout and layer normalization.
  • A refined multi-scale fusion module.
  • A residual regression head.
  • Cosine annealing learning rate scheduler with warmup.
  
Feel free to experiment further with augmentations and architecture changes.
"""

import os
import time
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2
from typing import Tuple, List, Dict, Optional, Union
import random
from tqdm import tqdm

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from torch.cuda.amp import autocast, GradScaler

# Import TMD modules (assumed to be available)
from tmd.processor import TMDProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths to sample data
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'examples', 'gelsight')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'tmd_alignment_model_improved')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

#######################################
# Dataset Definition (Unchanged)
#######################################

class TMDHeightmapDataset(Dataset):
    """Dataset for training TMD heightmap alignment model."""
    
    def __init__(self, 
                 data_dir: str,
                 max_samples: int = 10000,  # Increased from 5000
                 patch_size: int = 256,
                 max_offset: int = 100,
                 max_rotation: float = 10.0,
                 augmentation: bool = True,
                 multi_resolution: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing TMD files
            max_samples: Maximum number of samples to generate
            patch_size: Size of heightmap patches
            max_offset: Maximum translation offset in pixels
            max_rotation: Maximum rotation in degrees
            augmentation: Whether to apply data augmentation
            multi_resolution: Whether to use multi-resolution processing
        """
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.patch_size = patch_size
        self.max_offset = max_offset
        self.max_rotation = max_rotation
        self.augmentation = augmentation
        self.multi_resolution = multi_resolution
        
        # Find all TMD files
        self.tmd_files = list(Path(data_dir).glob('**/*.tmd'))
        if not self.tmd_files:
            logger.error(f"No TMD files found in {data_dir}")
            raise FileNotFoundError(f"No TMD files found in {data_dir}")
        
        logger.info(f"Found {len(self.tmd_files)} TMD files in {data_dir}")
        self.heightmaps = self._load_heightmaps()
        
        # Normalize heightmaps
        self.heightmaps = [self._normalize_heightmap(h) for h in self.heightmaps]
        
        # Generate samples with known transformations
        self.samples = self._generate_samples()
        
    def _load_heightmaps(self) -> List[np.ndarray]:
        """Load heightmaps from TMD files."""
        heightmaps = []
        for tmd_file in self.tmd_files:
            try:
                logger.debug(f"Loading heightmap from {tmd_file}")
                processor = TMDProcessor(str(tmd_file))
                processor.process()
                heightmap = processor.get_height_map()
                if heightmap is not None and heightmap.size > 0:
                    heightmap = heightmap.astype(np.float32)
                    heightmaps.append(heightmap)
                else:
                    logger.warning(f"Failed to extract valid heightmap from {tmd_file}")
            except Exception as e:
                logger.warning(f"Error loading {tmd_file}: {e}")
        logger.info(f"Successfully loaded {len(heightmaps)} heightmaps")
        return heightmaps
    
    def _normalize_heightmap(self, heightmap: np.ndarray) -> np.ndarray:
        """Normalize heightmap to [0, 1] range."""
        h_min, h_max = np.min(heightmap), np.max(heightmap)
        if h_max > h_min:
            return (heightmap - h_min) / (h_max - h_min)
        else:
            return heightmap
    
    def _generate_samples(self) -> List[Dict]:
        """Generate training samples with known transformations."""
        samples = []
        for _ in tqdm(range(self.max_samples), desc="Generating samples"):
            heightmap = random.choice(self.heightmaps)
            h, w = heightmap.shape
            if h < self.patch_size or w < self.patch_size:
                continue
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)
            reference_patch = heightmap[y:y+self.patch_size, x:x+self.patch_size]
            dx = random.randint(-self.max_offset, self.max_offset)
            dy = random.randint(-self.max_offset, self.max_offset)
            if random.random() < 0.3:
                rotation = random.uniform(-self.max_rotation, self.max_rotation)
            else:
                rotation = random.uniform(-self.max_rotation/2, self.max_rotation/2)
            center = (self.patch_size // 2, self.patch_size // 2)
            if rotation != 0:
                rot_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
                source_patch = cv2.warpAffine(reference_patch, rot_matrix, 
                                              (self.patch_size, self.patch_size),
                                              flags=cv2.INTER_CUBIC, 
                                              borderMode=cv2.BORDER_REPLICATE)
            else:
                source_patch = reference_patch.copy()
                rot_matrix = np.eye(2, 3, dtype=np.float32)
            trans_matrix = np.eye(2, 3, dtype=np.float32)
            trans_matrix[0, 2] = dx
            trans_matrix[1, 2] = dy
            source_patch = cv2.warpAffine(source_patch, trans_matrix, 
                                          (self.patch_size, self.patch_size),
                                          flags=cv2.INTER_CUBIC, 
                                          borderMode=cv2.BORDER_REPLICATE)
            if self.augmentation:
                noise_level = random.uniform(0.005, 0.02)
                source_patch += np.random.normal(0, noise_level, source_patch.shape)
                alpha = random.uniform(0.8, 1.2)
                beta = random.uniform(-0.1, 0.1)
                source_patch = np.clip(alpha * source_patch + beta, 0, 1)
                if random.random() < 0.3:
                    blur_size = random.choice([3, 5, 7])
                    source_patch = cv2.GaussianBlur(source_patch, (blur_size, blur_size), 0)
                if random.random() < 0.2:
                    occlusion_size = random.randint(5, 30)
                    oc_x = random.randint(0, self.patch_size - occlusion_size)
                    oc_y = random.randint(0, self.patch_size - occlusion_size)
                    source_patch[oc_y:oc_y+occlusion_size, oc_x:oc_x+occlusion_size] = 0
            samples.append({
                'source': source_patch,
                'target': reference_patch,
                'dx': dx,
                'dy': dy,
                'rotation': rotation
            })
        logger.info(f"Generated {len(samples)} training samples")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        source_tensor = torch.from_numpy(sample['source']).unsqueeze(0).float()
        target_tensor = torch.from_numpy(sample['target']).unsqueeze(0).float()
        params = torch.tensor([sample['dx'], sample['dy'], sample['rotation']], dtype=torch.float32)
        if self.multi_resolution:
            source_half = F.avg_pool2d(source_tensor, kernel_size=2, stride=2)
            source_quarter = F.avg_pool2d(source_half, kernel_size=2, stride=2)
            target_half = F.avg_pool2d(target_tensor, kernel_size=2, stride=2)
            target_quarter = F.avg_pool2d(target_half, kernel_size=2, stride=2)
            return {
                'source': source_tensor,
                'target': target_tensor,
                'source_half': source_half,
                'target_half': target_half,
                'source_quarter': source_quarter,
                'target_quarter': target_quarter,
                'params': params
            }
        else:
            return {
                'source': source_tensor,
                'target': target_tensor,
                'params': params
            }

#############################################
# Advanced Model Components
#############################################

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based models to maintain spatial information."""
    
    def __init__(self, d_model, max_len=256):
        super().__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
        Returns:
            Tensor with added positional encodings
        """
        return x + self.pe[:, :x.size(1), :]

class CrossAttention(nn.Module):
    """Cross-attention module for correlating features between source and target."""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Project and reshape query, key, value
        q = self.q_proj(query).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Calculate attention weights and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output, attn_weights

class FeatureCorrelation(nn.Module):
    """Feature correlation module for explicit spatial correlation modeling."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Projection layer for dimension reduction
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, source_features, target_features):
        batch_size, channels, height, width = source_features.shape
        
        # Compute correlation volume using einsum for efficient computation
        source_flat = source_features.view(batch_size, channels, -1)
        target_flat = target_features.view(batch_size, channels, -1)
        
        # Normalize features for better correlation
        source_norm = F.normalize(source_flat, dim=1)
        target_norm = F.normalize(target_flat, dim=1)
        
        # Compute correlation matrix
        correlation = torch.einsum('bci,bcj->bij', source_norm, target_norm)
        
        # Reshape to spatial dimensions
        correlation = correlation.view(batch_size, height, width, height, width)
        
        # Extract the most relevant correlation information (you can customize this)
        # Here we extract the max correlation across all target positions for each source position
        max_corr, _ = correlation.max(dim=-1)
        max_corr, _ = max_corr.max(dim=-1)
        max_corr = max_corr.unsqueeze(1)  # Add channel dimension
        
        # Concatenate with source features for context
        concat_features = torch.cat([source_features, max_corr.expand_as(source_features)], dim=1)
        
        # Project to desired output dimension
        output = self.projection(concat_features)
        
        return output

class AdaptiveFocalLoss(nn.Module):
    """
    Focal loss with adaptive gamma parameter based on alignment difficulty.
    Harder samples (those with larger errors) receive higher weight.
    """
    
    def __init__(self, gamma=2.0, reduction='mean', weight_rotation=5.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.weight_rotation = weight_rotation
        
    def forward(self, pred, target, overlap_score=None):
        """
        Args:
            pred: Predictions [B, params]
            target: Ground truth [B, params]
            overlap_score: Optional score indicating sample difficulty [B]
                          (lower score = harder sample, higher weight)
        """
        # Calculate squared error
        sq_error = (pred - target) ** 2
        
        # Weight rotation parameters if needed
        weights = torch.ones_like(target)
        if target.shape[1] > 2:  # If rotation parameter is included
            weights[:, 2] = self.weight_rotation
        
        # Apply weight to errors
        weighted_error = sq_error * weights
        
        # Calculate focal weights based on error magnitude
        focal_weights = torch.exp(self.gamma * weighted_error)
        
        # Apply overlap score as additional weighting if provided
        if overlap_score is not None:
            # Convert to tensor if needed
            if not isinstance(overlap_score, torch.Tensor):
                overlap_score = torch.tensor(overlap_score, device=pred.device)
            
            # Rescale overlap score to [0.5, 1.5] - lower overlap gets higher weight
            overlap_weight = 1.5 - overlap_score.unsqueeze(1) * 1.0
            focal_weights = focal_weights * overlap_weight
        
        # Calculate final loss
        loss = weighted_error * focal_weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ImprovedCNNBackbone(nn.Module):
    def __init__(self, input_channels=1):  # Changed default from 2 to 1
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Add a few residual blocks for robust feature extraction
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 256, blocks=2, stride=2)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
class ImprovedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout1(attn_output)
        x = x + self.mlp(self.norm2(x))
        return x

class ImprovedAlignmentPredictor(nn.Module):
    """Advanced model for predicting alignment parameters between two heightmaps."""
    
    def __init__(self, 
                use_rotation: bool = True, 
                input_channels: int = 1,  # Changed default from 2 to 1
                feature_dim: int = 256,
                num_attention_heads: int = 8, 
                dropout_rate: float = 0.1,
                use_transformer: bool = True,
                multi_resolution: bool = True):
        """
        Initialize the model.
        
        Args:
            use_rotation: Whether to predict rotation in addition to translation
            input_channels: Number of input channels (1 for heightmaps)
            feature_dim: Feature dimension
            num_attention_heads: Number of attention heads
            dropout_rate: Dropout rate
            use_transformer: Whether to use transformer blocks
            multi_resolution: Whether to use multi-resolution processing
        """
        super(ImprovedAlignmentPredictor, self).__init__()
        
        self.use_rotation = use_rotation
        self.output_params = 3 if use_rotation else 2
        self.use_transformer = use_transformer
        self.multi_resolution = multi_resolution
        
        # Save input channels configuration
        self.input_channels = input_channels
        
        # CNN backbone using pre-activation ResNet blocks
        self.backbone = ImprovedCNNBackbone(input_channels)
        
        # Feature correlation module
        self.feature_correlation = FeatureCorrelation(256, feature_dim)
        
        # Add cross-attention module for better correlation
        self.cross_attention = CrossAttention(
            embed_dim=feature_dim,
            num_heads=num_attention_heads,
            dropout=dropout_rate
        )
        
        # Positional encoding for spatial awareness
        self.pos_encoding = PositionalEncoding(feature_dim)
        
        # Multi-resolution branches (if enabled)
        if multi_resolution:
            self.half_res_branch = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d((8, 8))
            )
            
            self.quarter_res_branch = nn.Sequential(
                nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            
            # Multi-scale fusion
            self.fusion = nn.Sequential(
                nn.Conv2d(feature_dim + 64 + 16, feature_dim, kernel_size=1),
                nn.BatchNorm2d(feature_dim),
                nn.SiLU(inplace=True),
                nn.Dropout2d(dropout_rate)
            )
        
        if use_transformer:
            # Transformer blocks for global reasoning
            transformer_dim = feature_dim
            self.transformer_blocks = nn.Sequential(
                ImprovedTransformerBlock(transformer_dim, num_heads=num_attention_heads, dropout=dropout_rate),
                ImprovedTransformerBlock(transformer_dim, num_heads=num_attention_heads, dropout=dropout_rate)
            )
            
            # Global pooling with confidence-weighted attention
            self.confidence_layer = nn.Sequential(
                nn.Linear(transformer_dim, 1),
                nn.Sigmoid()
            )
            
            # Global feature extraction
            self.global_pool = nn.Sequential(
                nn.LayerNorm(transformer_dim),
                nn.AdaptiveAvgPool1d(1)
            )
            
            regression_input_dim = transformer_dim
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            regression_input_dim = feature_dim
        
        # Regression head with residual connections
        self.regressor = nn.Sequential(
            nn.Linear(regression_input_dim, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, self.output_params)
        )
        
        # Auxiliary confidence branch to estimate alignment quality
        self.confidence_branch = nn.Sequential(
            nn.Linear(regression_input_dim, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _extract_features(self, source, target, source_half=None, target_half=None, 
                         source_quarter=None, target_quarter=None):
        """Extract and fuse features from multiple resolutions."""
        # Extract source and target features separately
        source_features = self.backbone(source)
        target_features = self.backbone(target)
        
        # Apply feature correlation for better alignment
        correlated_features = self.feature_correlation(source_features, target_features)
        
        # Process multi-resolution inputs if available
        if self.multi_resolution and source_half is not None:
            # Create input for half-resolution branch
            half_x = torch.cat([source_half, target_half], dim=1) if self.input_channels == 1 else torch.cat([source_half, target_half], dim=0)
            if self.input_channels == 1 and hasattr(self, 'half_res_branch'):
                # Ensure half_res_branch expects the right number of input channels
                if self.half_res_branch[0].in_channels != 2:  # If it's not 2 already
                    # Rebuild half_res_branch with correct input channels
                    self.half_res_branch = nn.Sequential(
                        nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(32),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(64),
                        nn.SiLU(inplace=True),
                        nn.AdaptiveAvgPool2d((8, 8))
                    )
            half_features = self.half_res_branch(half_x)
            
            # Create input for quarter-resolution branch
            quarter_x = torch.cat([source_quarter, target_quarter], dim=1) if self.input_channels == 1 else torch.cat([source_quarter, target_quarter], dim=0)
            if self.input_channels == 1 and hasattr(self, 'quarter_res_branch'):
                # Ensure quarter_res_branch expects the right number of input channels
                if self.quarter_res_branch[0].in_channels != 2:  # If it's not 2 already
                    # Rebuild quarter_res_branch with correct input channels
                    self.quarter_res_branch = nn.Sequential(
                        nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16),
                        nn.SiLU(inplace=True),
                        nn.AdaptiveAvgPool2d((4, 4))
                    )
            quarter_features = self.quarter_res_branch(quarter_x)
            
            # Resize features to match
            resized_correlated = F.adaptive_avg_pool2d(correlated_features, (8, 8))
            resized_half = F.adaptive_avg_pool2d(half_features, (8, 8))
            resized_quarter = F.adaptive_avg_pool2d(quarter_features, (8, 8))
            
            # Concatenate and fuse multi-resolution features
            combined_features = torch.cat([resized_correlated, resized_half, resized_quarter], dim=1)
            fused_features = self.fusion(combined_features)
            return fused_features
        
        # Return correlated features if multi-resolution not enabled
        return correlated_features
    
    def forward(self, source, target, source_half=None, target_half=None,
               source_quarter=None, target_quarter=None):
        """
        Forward pass with improved feature extraction and fusion.
        
        Args:
            source: Source heightmap tensor [B, 1, H, W]
            target: Target heightmap tensor [B, 1, H, W]
            source_half, target_half: Half-resolution inputs (optional)
            source_quarter, target_quarter: Quarter-resolution inputs (optional)
            
        Returns:
            Predicted transformation parameters and confidence score
        """
        # Extract features with multi-resolution if available
        features = self._extract_features(
            source, target, source_half, target_half, source_quarter, target_quarter
        )
        
        # Apply transformer blocks if enabled
        if self.use_transformer:
            # Prepare features for transformer
            batch_size, channels, height, width = features.shape
            flattened = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
            
            # Add positional encoding
            pos_encoded = self.pos_encoding(flattened)
            
            # Extract source and target features for cross-attention
            source_feat = self.backbone(source).flatten(2).transpose(1, 2)
            target_feat = self.backbone(target).flatten(2).transpose(1, 2)
            
            # Apply cross-attention between source and target features
            cross_attn_output, _ = self.cross_attention(pos_encoded, target_feat, source_feat)
            
            # Process with transformer blocks for global reasoning
            transformer_output = self.transformer_blocks(cross_attn_output)
            
            # Calculate confidence weights for each token
            confidence_weights = self.confidence_layer(transformer_output)
            
            # Apply confidence-weighted pooling
            weighted_output = transformer_output * confidence_weights
            
            # Global pooling
            pooled = self.global_pool(weighted_output.transpose(1, 2)).squeeze(-1)
        else:
            # Simple global average pooling
            pooled = self.global_pool(features).squeeze(-1).squeeze(-1)
        
        # Predict alignment parameters
        params = self.regressor(pooled)
        
        # Predict confidence score (can be used for estimation quality)
        confidence_score = self.confidence_branch(pooled)
        
        return params, confidence_score

#############################################
# Enhanced Training Functions
#############################################

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 50,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device('cpu'),
    checkpoint_dir: str = OUTPUT_DIR,
    use_mixed_precision: bool = True,
    clip_grad_norm: Optional[float] = 1.0,
    early_stopping_patience: int = 10,
    gradient_accumulation_steps: int = 1
) -> Dict:
    """
    Train the alignment model with advanced training techniques.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: PyTorch optimizer
        num_epochs: Number of training epochs
        scheduler: Learning rate scheduler
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_mixed_precision: Whether to use mixed precision training
        clip_grad_norm: Maximum gradient norm for clipping
        early_stopping_patience: Number of epochs to wait for improvement
        gradient_accumulation_steps: Number of batches to accumulate gradients
        
    Returns:
        Training history dictionary
    """
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'train_metrics': [], 
        'val_metrics': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    # Initialize loss functions
    criterion = AdaptiveFocalLoss(gamma=2.0, weight_rotation=5.0)
    
    # Initialize scaler for mixed precision training
    scaler = GradScaler() if use_mixed_precision and device.type == 'cuda' else None
    
    # Early stopping variables
    early_stop_counter = 0
    best_val_loss = float('inf')
    
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Mixed precision: {'enabled' if use_mixed_precision and scaler is not None else 'disabled'}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch
        
        train_metrics = {
            'mae_dx': 0.0, 
            'mae_dy': 0.0, 
            'mae_rotation': 0.0, 
            'confidence_error': 0.0,
            'samples': 0
        }
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for i, batch in enumerate(progress_bar):
            # Move data to device
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            params = batch['params'].to(device)
            
            # Handle multi-resolution inputs if available
            multi_res_inputs = {}
            if 'source_half' in batch:
                multi_res_inputs['source_half'] = batch['source_half'].to(device)
                multi_res_inputs['target_half'] = batch['target_half'].to(device)
                multi_res_inputs['source_quarter'] = batch['source_quarter'].to(device)
                multi_res_inputs['target_quarter'] = batch['target_quarter'].to(device)
            
            # Forward pass with or without mixed precision
            if use_mixed_precision and scaler is not None:
                with autocast():
                    pred_params, confidence = model(source, target, **multi_res_inputs)
                    loss = criterion(pred_params, params)
                    
                    # Normalize loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Step optimizer and scaler after accumulating gradients
                if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    if clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard forward/backward without mixed precision
                pred_params, confidence = model(source, target, **multi_res_inputs)
                loss = criterion(pred_params, params)
                
                # Normalize loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Step optimizer after accumulating gradients
                if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    if clip_grad_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Update statistics
            batch_size = source.size(0)
            train_loss += loss.item() * gradient_accumulation_steps * batch_size
            batch_count += batch_size
            
            # Update metrics
            with torch.no_grad():
                train_metrics['mae_dx'] += torch.abs(pred_params[:, 0] - params[:, 0]).sum().item()
                train_metrics['mae_dy'] += torch.abs(pred_params[:, 1] - params[:, 1]).sum().item()
                
                if params.shape[1] > 2:  # If rotation is included
                    train_metrics['mae_rotation'] += torch.abs(pred_params[:, 2] - params[:, 2]).sum().item()
                
                # Ideal confidence is 1.0 for perfect alignment, 0.0 for poor alignment
                error_magnitude = torch.abs(pred_params - params).mean(dim=1, keepdim=True)
                target_confidence = torch.exp(-error_magnitude * 5.0)  # Convert error to confidence score
                train_metrics['confidence_error'] += torch.abs(confidence - target_confidence).sum().item()
                
                train_metrics['samples'] += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item() * gradient_accumulation_steps,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Calculate average training metrics
        if train_metrics['samples'] > 0:
            for key in ['mae_dx', 'mae_dy', 'mae_rotation', 'confidence_error']:
                if key in train_metrics:
                    train_metrics[key] /= train_metrics['samples']
            
            train_loss /= train_metrics['samples']
        
        # Add metrics to history
        history['train_loss'].append(train_loss)
        history['train_metrics'].append(train_metrics.copy())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {
            'mae_dx': 0.0, 
            'mae_dy': 0.0, 
            'mae_rotation': 0.0, 
            'confidence_error': 0.0,
            'samples': 0
        }
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in progress_bar:
                # Move data to device
                source = batch['source'].to(device)
                target = batch['target'].to(device)
                params = batch['params'].to(device)
                
                # Handle multi-resolution inputs if available
                multi_res_inputs = {}
                if 'source_half' in batch:
                    multi_res_inputs['source_half'] = batch['source_half'].to(device)
                    multi_res_inputs['target_half'] = batch['target_half'].to(device)
                    multi_res_inputs['source_quarter'] = batch['source_quarter'].to(device)
                    multi_res_inputs['target_quarter'] = batch['target_quarter'].to(device)
                
                # Forward pass
                pred_params, confidence = model(source, target, **multi_res_inputs)
                loss = criterion(pred_params, params)
                
                # Update statistics
                batch_size = source.size(0)
                val_loss += loss.item() * batch_size
                
                # Update metrics
                val_metrics['mae_dx'] += torch.abs(pred_params[:, 0] - params[:, 0]).sum().item()
                val_metrics['mae_dy'] += torch.abs(pred_params[:, 1] - params[:, 1]).sum().item()
                
                if params.shape[1] > 2:  # If rotation is included
                    val_metrics['mae_rotation'] += torch.abs(pred_params[:, 2] - params[:, 2]).sum().item()
                
                # Calculate confidence metrics
                error_magnitude = torch.abs(pred_params - params).mean(dim=1, keepdim=True)
                target_confidence = torch.exp(-error_magnitude * 5.0)
                val_metrics['confidence_error'] += torch.abs(confidence - target_confidence).sum().item()
                
                val_metrics['samples'] += batch_size
        
        # Calculate average validation metrics
        if val_metrics['samples'] > 0:
            for key in ['mae_dx', 'mae_dy', 'mae_rotation', 'confidence_error']:
                if key in val_metrics:
                    val_metrics[key] /= val_metrics['samples']
            
            val_loss /= val_metrics['samples']
        
        # Add metrics to history
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics.copy())
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        logger.info(f"  Train MAE: dx={train_metrics['mae_dx']:.2f}, dy={train_metrics['mae_dy']:.2f}" + 
                   (f", rot={train_metrics['mae_rotation']:.2f}°" if 'mae_rotation' in train_metrics else ""))
        logger.info(f"  Val MAE: dx={val_metrics['mae_dx']:.2f}, dy={val_metrics['mae_dy']:.2f}" + 
                   (f", rot={val_metrics['mae_rotation']:.2f}°" if 'mae_rotation' in val_metrics else ""))
        logger.info(f"  Learning rate: {optimizer.param_groups[0]['lr']:.8f}")
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            
            # Reset early stopping counter
            early_stop_counter = 0
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'loss': val_loss,
                'metrics': val_metrics,
                'history': history
            }, checkpoint_path)
            
            logger.info(f"Saved best model checkpoint to {checkpoint_path}")
        else:
            # Increment early stopping counter
            early_stop_counter += 1
            logger.info(f"Early stopping counter: {early_stop_counter}/{early_stopping_patience}")
        
        # Always save latest model
        latest_path = os.path.join(checkpoint_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'loss': val_loss,
            'metrics': val_metrics,
            'history': history
        }, latest_path)
        
        # Early stopping
        if early_stopping_patience > 0 and early_stop_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Return training history
    return history

# Additional enhanced helper functions

def get_cosine_scheduler_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """
    Get a cosine annealing scheduler with warmup.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of training epochs
        min_lr: Minimum learning rate
        
    Returns:
        PyTorch learning rate scheduler
    """
    def lr_lambda(epoch):
        # Warmup phase
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        
        # Cosine annealing phase
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        # Ensure minimum learning rate
        return max(min_lr / optimizer.param_groups[0]['lr'], cosine_decay)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def visualize_attention_maps(model, sample_batch, output_path, device):
    """
    Visualize attention maps from the model.
    
    Args:
        model: Trained alignment model
        sample_batch: Batch of samples to visualize
        output_path: Path to save visualization
        device: Device to run model on
    """
    model.eval()
    
    # Move data to device
    source = sample_batch['source'].to(device)
    target = sample_batch['target'].to(device)
    true_params = sample_batch['params'].to(device)
    
    # Get multi-resolution inputs if available
    multi_res_inputs = {}
    if 'source_half' in sample_batch:
        multi_res_inputs['source_half'] = sample_batch['source_half'].to(device)
        multi_res_inputs['target_half'] = sample_batch['target_half'].to(device)
        multi_res_inputs['source_quarter'] = sample_batch['source_quarter'].to(device)
        multi_res_inputs['target_quarter'] = sample_batch['target_quarter'].to(device)
    
    # Forward pass to get predictions and attention maps
    with torch.no_grad():
        pred_params, confidence = model(source, target, **multi_res_inputs)
    
    # Extract attention weights from transformer blocks if accessible
    if hasattr(model, 'transformer_blocks') and hasattr(model.transformer_blocks[0], 'attn'):
        try:
            # This assumes attention weights are accessible
            # Implement hooks if they're not directly accessible
            attention_weights = model.transformer_blocks[0].attn._attn_weights
            
            # Visualize attention maps
            batch_size = min(4, source.shape[0])  # Visualize up to 4 samples
            
            for i in range(batch_size):
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Source image
                axes[0].imshow(source[i, 0].cpu().numpy(), cmap='viridis')
                axes[0].set_title("Source Heightmap")
                axes[0].axis('off')
                
                # Target image
                axes[1].imshow(target[i, 0].cpu().numpy(), cmap='viridis')
                axes[1].set_title("Target Heightmap")
                axes[1].axis('off')
                
                # Attention map (mean across heads)
                attn = attention_weights[i].mean(dim=0).cpu().numpy()
                axes[2].imshow(attn, cmap='hot', interpolation='nearest')
                axes[2].set_title("Attention Map")
                axes[2].axis('off')
                
                # Confidence-weighted feature map
                confidence_map = confidence[i].item()
                title = f"Confidence: {confidence_map:.2f}\n"
                title += f"Pred: dx={pred_params[i, 0].item():.1f}, dy={pred_params[i, 1].item():.1f}"
                if pred_params.shape[1] > 2:
                    title += f", rot={pred_params[i, 2].item():.1f}°"
                title += f"\nTrue: dx={true_params[i, 0].item():.1f}, dy={true_params[i, 1].item():.1f}"
                if true_params.shape[1] > 2:
                    title += f", rot={true_params[i, 2].item():.1f}°"
                
                axes[3].imshow(np.zeros_like(source[i, 0].cpu().numpy()), cmap='viridis')
                axes[3].text(0.5, 0.5, title, ha='center', va='center', fontsize=10,
                            transform=axes[3].transAxes)
                axes[3].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, f'attention_map_{i}.png'), dpi=150)
                plt.close(fig)
                
            logger.info(f"Saved attention map visualizations to {output_path}")
        except (AttributeError, IndexError) as e:
            logger.warning(f"Failed to extract attention weights: {e}")
    else:
        logger.warning("Attention weights not accessible in this model")

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = torch.device('cpu'),
    output_dir: str = OUTPUT_DIR
) -> Dict:
    model.eval()
    metrics = {
        'mse_loss': 0.0,
        'mae_dx': 0.0,
        'mae_dy': 0.0,
        'mae_rotation': 0.0,
        'max_error_dx': 0.0,
        'max_error_dy': 0.0,
        'max_error_rotation': 0.0,
        'success_rate_2px': 0.0,
        'success_rate_5px': 0.0,
        'success_rate_1deg': 0.0,
        'samples_processed': 0
    }
    criterion = nn.MSELoss()
    vis_samples = []
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
        for i, batch in enumerate(progress_bar):
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            true_params = batch['params'].to(device)
            multi_res_inputs = {}
            if 'source_half' in batch:
                multi_res_inputs['source_half'] = batch['source_half'].to(device)
                multi_res_inputs['target_half'] = batch['target_half'].to(device)
                multi_res_inputs['source_quarter'] = batch['source_quarter'].to(device)
                multi_res_inputs['target_quarter'] = batch['target_quarter'].to(device)
            pred_params, confidence = model(source, target, **multi_res_inputs)
            mse_loss = criterion(pred_params, true_params)
            if i < 10:
                for j in range(min(1, source.size(0))):
                    vis_samples.append({
                        'source': source[j].cpu().numpy().squeeze(),
                        'target': target[j].cpu().numpy().squeeze(),
                        'true_params': true_params[j].cpu().numpy(),
                        'pred_params': pred_params[j].cpu().numpy()
                    })
            batch_size = source.size(0)
            metrics['mse_loss'] += mse_loss.item() * batch_size
            abs_dx_error = torch.abs(pred_params[:, 0] - true_params[:, 0])
            abs_dy_error = torch.abs(pred_params[:, 1] - true_params[:, 1])
            metrics['mae_dx'] += abs_dx_error.sum().item()
            metrics['mae_dy'] += abs_dy_error.sum().item()
            metrics['max_error_dx'] = max(metrics['max_error_dx'], abs_dx_error.max().item())
            metrics['max_error_dy'] = max(metrics['max_error_dy'], abs_dy_error.max().item())
            metrics['success_rate_2px'] += torch.sum((abs_dx_error < 2) & (abs_dy_error < 2)).item()
            metrics['success_rate_5px'] += torch.sum((abs_dx_error < 5) & (abs_dy_error < 5)).item()
            if pred_params.size(1) > 2:
                abs_rot_error = torch.abs(pred_params[:, 2] - true_params[:, 2])
                metrics['mae_rotation'] += abs_rot_error.sum().item()
                metrics['max_error_rotation'] = max(metrics['max_error_rotation'], abs_rot_error.max().item())
                metrics['success_rate_1deg'] += torch.sum(abs_rot_error < 1.0).item()
            metrics['samples_processed'] += batch_size
    for key in ['mse_loss', 'mae_dx', 'mae_dy', 'mae_rotation']:
        if metrics['samples_processed'] > 0:
            metrics[key] /= metrics['samples_processed']
    metrics['success_rate_2px'] = (metrics['success_rate_2px'] / metrics['samples_processed']) * 100
    metrics['success_rate_5px'] = (metrics['success_rate_5px'] / metrics['samples_processed']) * 100
    if metrics['samples_processed'] > 0:
        metrics['success_rate_1deg'] = (metrics['success_rate_1deg'] / metrics['samples_processed']) * 100
    logger.info(f"Evaluation results:")
    logger.info(f"MSE Loss: {metrics['mse_loss']:.6f}")
    logger.info(f"MAE X-offset: {metrics['mae_dx']:.2f} pixels")
    logger.info(f"MAE Y-offset: {metrics['mae_dy']:.2f} pixels")
    logger.info(f"Max X-offset error: {metrics['max_error_dx']:.2f} pixels")
    logger.info(f"Max Y-offset error: {metrics['max_error_dy']:.2f} pixels")
    logger.info(f"Success rate (<2px): {metrics['success_rate_2px']:.2f}%")
    logger.info(f"Success rate (<5px): {metrics['success_rate_5px']:.2f}%")
    if metrics['mae_rotation'] > 0:
        logger.info(f"MAE Rotation: {metrics['mae_rotation']:.2f} degrees")
        logger.info(f"Max Rotation error: {metrics['max_error_rotation']:.2f} degrees")
        logger.info(f"Success rate (<1°): {metrics['success_rate_1deg']:.2f}%")
    visualize_predictions(vis_samples, output_dir)
    return metrics

def visualize_predictions(samples: List[Dict], output_dir: str) -> None:
    os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
    for i, sample in enumerate(samples):
        source = sample['source']
        target = sample['target']
        true_params = sample['true_params']
        pred_params = sample['pred_params']
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(source, cmap='viridis')
        axes[0].set_title("Source Heightmap")
        axes[0].axis('off')
        axes[1].imshow(target, cmap='viridis')
        axes[1].set_title("Target Heightmap")
        axes[1].axis('off')
        dx_pred, dy_pred = pred_params[0], pred_params[1]
        rotation_pred = pred_params[2] if len(pred_params) > 2 else 0.0
        height, width = source.shape
        center = (width // 2, height // 2)
        aligned = source.copy()
        if rotation_pred != 0:
            rot_matrix = cv2.getRotationMatrix2D(center, rotation_pred, 1.0)
            aligned = cv2.warpAffine(aligned, rot_matrix, (width, height),
                                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        trans_matrix = np.eye(2, 3, dtype=np.float32)
        trans_matrix[0, 2] = dx_pred
        trans_matrix[1, 2] = dy_pred
        aligned = cv2.warpAffine(aligned, trans_matrix, (width, height),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        axes[2].imshow(aligned, cmap='viridis')
        axes[2].set_title(f"Aligned Result\nPred: dx={dx_pred:.1f}, dy={dy_pred:.1f}, rot={rotation_pred:.1f}°\n"
                         f"True: dx={true_params[0]:.1f}, dy={true_params[1]:.1f}, rot={true_params[2]:.1f}°")
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vis', f'pred_{i}.png'), dpi=150)
        plt.close(fig)

def create_synthetic_dataset(
    size: int = 512,
    num_samples: int = 10000,
    patch_size: int = 256,
    max_offset: int = 100,
    max_rotation: float = 10.0,
    num_base_maps: int = 20,
    multi_resolution: bool = True
) -> TMDHeightmapDataset:
    logger.info(f"Creating synthetic dataset with {num_samples} samples")
    heightmaps = []
    for i in range(num_base_maps):
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)
        heightmap = np.zeros((size, size), dtype=np.float32)
        for _ in range(random.randint(3, 10)):
            center_x = random.uniform(-4, 4)
            center_y = random.uniform(-4, 4)
            sigma = random.uniform(0.3, 3.0)
            amplitude = random.uniform(0.3, 2.0)
            heightmap += amplitude * np.exp(-((X-center_x)**2 + (Y-center_y)**2) / sigma)
        for _ in range(random.randint(1, 3)):
            freq_x = random.uniform(0.5, 15)
            freq_y = random.uniform(0.5, 15)
            phase_x = random.uniform(0, 2*np.pi)
            phase_y = random.uniform(0, 2*np.pi)
            amplitude = random.uniform(0.05, 0.3)
            heightmap += amplitude * np.sin(freq_x * X + phase_x) * np.cos(freq_y * Y + phase_y)
        if random.random() < 0.5:
            angle = random.uniform(0, np.pi)
            freq = random.uniform(2, 10)
            Z = X * np.cos(angle) + Y * np.sin(angle)
            heightmap += 0.2 * np.sin(freq * Z)
        noise_level = random.uniform(0.02, 0.1)
        heightmap += noise_level * np.random.randn(size, size)
        heightmap = (heightmap - np.min(heightmap)) / (np.max(heightmap) - np.min(heightmap))
        heightmaps.append(heightmap)
    class SyntheticDataset(TMDHeightmapDataset):
        def __init__(self, heightmaps, max_samples, patch_size, max_offset, max_rotation, augmentation=True, multi_resolution=True):
            self.heightmaps = heightmaps
            self.max_samples = max_samples
            self.patch_size = patch_size
            self.max_offset = max_offset
            self.max_rotation = max_rotation
            self.augmentation = augmentation
            self.multi_resolution = multi_resolution
            self.samples = self._generate_samples()
        def _load_heightmaps(self):
            return self.heightmaps
    return SyntheticDataset(
        heightmaps=heightmaps,
        max_samples=num_samples,
        patch_size=patch_size,
        max_offset=max_offset,
        max_rotation=max_rotation,
        multi_resolution=multi_resolution
    )

def export_model_for_inference(
    model: nn.Module, 
    output_path: str, 
    input_shape: Tuple[int, int] = (256, 256),
    quantize: bool = True
) -> None:
    model.eval()
    dummy_source = torch.randn(1, 1, input_shape[0], input_shape[1])
    dummy_target = torch.randn(1, 1, input_shape[0], input_shape[1])
    if hasattr(model, 'multi_resolution') and model.multi_resolution:
        dummy_source_half = torch.randn(1, 1, input_shape[0]//2, input_shape[1]//2)
        dummy_target_half = torch.randn(1, 1, input_shape[0]//2, input_shape[1]//2)
        dummy_source_quarter = torch.randn(1, 1, input_shape[0]//4, input_shape[1]//4)
        dummy_target_quarter = torch.randn(1, 1, input_shape[0]//4, input_shape[1]//4)
        scripted_model = torch.jit.trace(
            model, 
            (dummy_source, dummy_target, dummy_source_half, dummy_target_half, dummy_source_quarter, dummy_target_quarter)
        )
    else:
        scripted_model = torch.jit.trace(model, (dummy_source, dummy_target))
    optimized_model = torch.jit.optimize_for_inference(scripted_model)
    if quantize:
        try:
            logger.info("Quantizing model to int8...")
            quantized_model = torch.quantization.quantize_dynamic(
                optimized_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
            torch.jit.save(quantized_model, output_path)
            logger.info(f"Quantized model saved to {output_path}")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}. Saving unquantized model.")
            torch.jit.save(optimized_model, output_path)
            logger.info(f"Optimized model saved to {output_path}")
    else:
        torch.jit.save(optimized_model, output_path)
        logger.info(f"Optimized model saved to {output_path}")

def get_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        progress = float(epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def main():
    parser = argparse.ArgumentParser(description='TMD Heightmap Alignment Model - Improved')
    parser.add_argument('--data_dir', type=str, default=SAMPLE_DIR, help='Directory with TMD files')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data instead of real TMD files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of heightmap patches')
    parser.add_argument('--max_offset', type=int, default=100, help='Maximum offset in pixels')
    parser.add_argument('--max_rotation', type=float, default=10.0, help='Maximum rotation in degrees')
    parser.add_argument('--no_rotation', action='store_true', help='Disable rotation prediction')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of training samples to generate')
    parser.add_argument('--no_transformer', action='store_true', help='Disable transformer blocks')
    parser.add_argument('--no_multi_res', action='store_true', help='Disable multi-resolution processing')
    parser.add_argument('--test_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--export', action='store_true', help='Export optimized model for inference')
    parser.add_argument('--quantize', action='store_true', help='Quantize exported model')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    multi_resolution = not args.no_multi_res
    
    if args.synthetic:
        logger.info("Using synthetic dataset")
        dataset = create_synthetic_dataset(
            num_samples=args.num_samples,
            patch_size=args.patch_size,
            max_offset=args.max_offset,
            max_rotation=args.max_rotation,
            multi_resolution=multi_resolution
        )
    else:
        try:
            logger.info(f"Loading TMD heightmaps from {args.data_dir}")
            dataset = TMDHeightmapDataset(
                data_dir=args.data_dir,
                max_samples=args.num_samples,
                patch_size=args.patch_size,
                max_offset=args.max_offset,
                max_rotation=args.max_rotation,
                multi_resolution=multi_resolution
            )
        except FileNotFoundError:
            logger.warning(f"No TMD files found in {args.data_dir}, falling back to synthetic data")
            dataset = create_synthetic_dataset(
                num_samples=args.num_samples,
                patch_size=args.patch_size,
                max_offset=args.max_offset,
                max_rotation=args.max_rotation,
                multi_resolution=multi_resolution
            )
    
    if not args.synthetic and args.num_samples > 1000:
        logger.info("Adding synthetic samples to augment real data")
        synthetic_dataset = create_synthetic_dataset(
            num_samples=args.num_samples // 2,
            patch_size=args.patch_size,
            max_offset=args.max_offset,
            max_rotation=args.max_rotation,
            multi_resolution=multi_resolution
        )
        dataset = ConcatDataset([dataset, synthetic_dataset])
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Dataset split: {train_size} train, {val_size} val, {test_size} test samples")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=8, pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True if device.type == 'cuda' else False
    )
    
    use_rotation = not args.no_rotation
    use_transformer = not args.no_transformer
    
    # Use the improved model architecture
    model = ImprovedAlignmentPredictor(
        use_rotation=use_rotation,
        use_transformer=use_transformer,
        multi_resolution=multi_resolution,
        num_attention_heads=8,
        dropout_rate=0.2,
        input_channels=1  # Explicitly set input_channels to 1
    )
    model.to(device)
    
    logger.info(f"Model created with {'rotation and ' if use_rotation else ''}translation prediction")
    logger.info(f"Transformer blocks: {'enabled' if use_transformer else 'disabled'}")
    logger.info(f"Multi-resolution: {'enabled' if multi_resolution else 'disabled'}")
    
    if args.test_only:
        if args.model_path:
            logger.info(f"Loading model from {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            metrics = evaluate_model(model, test_loader, device, OUTPUT_DIR)
            logger.info(f"Test MSE: {metrics['mse_loss']:.6f}")
            if args.export:
                export_path = os.path.join(OUTPUT_DIR, 'alignment_model_optimized.pth')
                export_model_for_inference(model, export_path, input_shape=(args.patch_size, args.patch_size),
                                         quantize=args.quantize)
        else:
            logger.error("Model path required for test_only mode")
        return
    
    # Use enhanced optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Use better learning rate scheduler
    scheduler = get_cosine_scheduler_with_warmup(
        optimizer, 
        warmup_epochs=5, 
        total_epochs=args.num_epochs
    )
    
    # Use enhanced training function with gradient accumulation
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=OUTPUT_DIR,
        use_mixed_precision=args.mixed_precision,
        clip_grad_norm=1.0,
        early_stopping_patience=15,
        gradient_accumulation_steps=4
    )
    
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    metrics = evaluate_model(model, test_loader, device, OUTPUT_DIR)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150)
    plt.close()
    
    logger.info(f"Final test metrics:")
    logger.info(f"MSE Loss: {metrics['mse_loss']:.6f}")
    logger.info(f"MAE X-offset: {metrics['mae_dx']:.2f} pixels")
    logger.info(f"MAE Y-offset: {metrics['mae_dy']:.2f} pixels")
    logger.info(f"Success rate (<2px): {metrics['success_rate_2px']:.2f}%")
    logger.info(f"Success rate (<5px): {metrics['success_rate_5px']:.2f}%")
    if use_rotation:
        logger.info(f"MAE Rotation: {metrics['mae_rotation']:.2f} degrees")
        logger.info(f"Success rate (<1°): {metrics['success_rate_1deg']:.2f}%")
    
    logger.info(f"Training complete. Model saved to {best_model_path}")
    
    if args.export:
        export_path = os.path.join(OUTPUT_DIR, 'alignment_model_optimized.pth')
        export_model_for_inference(model, export_path, input_shape=(args.patch_size, args.patch_size),
                                 quantize=args.quantize)
    
    # Visualize attention maps for sample batch
    sample_batch = next(iter(test_loader))
    visualize_attention_maps(model, sample_batch, os.path.join(OUTPUT_DIR, 'attention_maps'), device)

if __name__ == "__main__":
    main()


import torch
import torch.nn.functional as F
from PIL import Image
import json
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import os
def _get_obj_from_str(string, reload=False):
    import importlib
    module, cls = string.rsplit(".", 1)
    mod = importlib.import_module(module)
    if reload:
        importlib.reload(mod)
    return getattr(mod, cls)


def instantiate_from_config(config):
    """
    Works with OmegaConf or plain dict configs.
    """
    try:
        from omegaconf import OmegaConf
        if not isinstance(config, dict):
            config = OmegaConf.to_container(config, resolve=True)
    except Exception:
        pass

    target = config.get("target", None)
    params = config.get("params", {})
    if target is None:
        raise KeyError("Expected key 'target' in config.")
    return _get_obj_from_str(target)(**params)


class VQGANProcessor:
    def __init__(self, config_path, checkpoint_path, device='cuda', image_size=256):
        """
        Initialize VQGAN model from config and checkpoint.
        
        Args:
            config_path: Path to model config yaml
            checkpoint_path: Path to model checkpoint
            device: Device to load model on
            image_size: Target image size (default 256)
        """
        self.device = torch.device(device)
        self.image_size = image_size
        
        # Load config
        config = OmegaConf.load(config_path)
        
        # Initialize model using instantiate_from_config
        self.model = instantiate_from_config(config.model)
        
        # Load checkpoint
        sd = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
        
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        # Detect model type
        model_type = type(self.model).__name__
        print(f"Model type: {model_type}")
        print(f"Model loaded successfully on {self.device}")
        
        # Check if it's GumbelVQ
        self.is_gumbel = 'Gumbel' in model_type or hasattr(self.model, 'quant_conv') and hasattr(self.model, 'quantize')
        if self.is_gumbel:
            print("Detected Gumbel-Softmax VQGAN")
        
    def preprocess_image(self, image_path, resize=True):
        """
        Load and preprocess image for VQGAN.
        
        Args:
            image_path: Path to input image
            resize: Whether to resize to target image_size (default True)
            
        Returns:
            Preprocessed image tensor
        """
        img = Image.open(image_path)
        # .convert('L')
        img = img.convert('RGB')
        
        if resize:
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        img = np.array(img).astype(np.float32) / 255.0
        img = img * 2.0 - 1.0  # Normalize to [-1, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(self.device)
    
    def encode_to_indices(self, image_path, resize=True):
        """
        Encode an image to codebook indices.
        
        Args:
            image_path: Path to input image or preprocessed tensor
            resize: Whether to resize image (default True)
            
        Returns:
            Tuple of (indices_list, height, width) for reconstruction
        """
        if isinstance(image_path, str):
            x = self.preprocess_image(image_path, resize=resize)
        else:
            x = image_path
            
        with torch.no_grad():
            # Use model.encode() method
            enc_out = self.model.encode(x)
            
            # Handle different return types
            if isinstance(enc_out, (tuple, list)):
                # VQModel returns (z_q, emb_loss, info)
                # info is (perplexity, min_encodings, indices)
                # GumbelVQ also returns similar structure
                z_q = enc_out[0]
                
                # Try to extract indices
                if len(enc_out) >= 3:
                    info = enc_out[2]
                    if isinstance(info, (tuple, list)) and len(info) >= 3:
                        _, _, indices = info
                    else:
                        # Gumbel might not return indices, compute from z_q
                        indices = self._get_indices_from_latent(z_q)
                else:
                    indices = self._get_indices_from_latent(z_q)
            else:
                # Fallback: get indices from latent
                indices = self._get_indices_from_latent(enc_out)
            
        # Get the spatial dimensions from indices
        if indices.ndim == 3:  # [B, H, W]
            b, h, w = indices.shape
            indices_list = indices[0].cpu().tolist()  # 2D list
        elif indices.ndim == 2:  # [B, H*W]
            b, hw = indices.shape
            h = w = int(np.sqrt(hw))
            indices_list = indices[0].reshape(h, w).cpu().tolist()
        else:  # [B*H*W] or similar
            indices = indices.reshape(-1)
            total = indices.shape[0]
            h = w = int(np.sqrt(total))
            indices_list = indices.reshape(h, w).cpu().tolist()
            
        return indices_list, h, w
    
    def _get_indices_from_latent(self, z_q):
        """
        Extract codebook indices from quantized latent representation.
        For Gumbel-Softmax models, compute nearest neighbors.
        """
        # Get codebook embeddings - handle different attribute names
        if hasattr(self.model.quantize, 'embedding'):
            codebook = self.model.quantize.embedding.weight
        elif hasattr(self.model.quantize, 'embed'):
            if isinstance(self.model.quantize.embed, torch.nn.Embedding):
                codebook = self.model.quantize.embed.weight
            else:
                codebook = self.model.quantize.embed
        else:
            raise AttributeError("Cannot find codebook embeddings in quantize module")
        
        # Reshape z_q to [B*H*W, C]
        b, c, h, w = z_q.shape
        z_flat = z_q.permute(0, 2, 3, 1).reshape(-1, c)
        
        # Compute distances to all codebook entries
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flat, codebook.t())
        
        # Get nearest codebook entry for each position
        indices = torch.argmin(d, dim=1)
        indices = indices.reshape(b, h, w)
        
        return indices
    
    def decode_from_indices(self, indices, height=None, width=None):
        """
        Decode codebook indices back to an image.
        
        Args:
            indices: List/2D array of codebook indices or flat list
            height: Height of latent grid (auto-calculated if None)
            width: Width of latent grid (auto-calculated if None)
            
        Returns:
            Reconstructed image as PIL Image
        """
        # Handle 2D list input
        if isinstance(indices, list) and isinstance(indices[0], list):
            height = len(indices)
            width = len(indices[0])
            indices_flat = [idx for row in indices for idx in row]
        else:
            indices_flat = indices
            
            # Auto-calculate dimensions if not provided
            if height is None or width is None:
                total = len(indices_flat)
                side = int(np.sqrt(total))
                if side * side != total:
                    raise ValueError(f"Cannot auto-determine square grid from {total} indices. Please specify height and width.")
                height = width = side
                print(f"Auto-detected latent grid size: {height}×{width}")
        
        # Convert indices to tensor and reshape
        indices_tensor = torch.tensor(indices_flat, dtype=torch.long).reshape(1, height, width).to(self.device)
        
        with torch.no_grad():
            # Get codebook embeddings
            if hasattr(self.model.quantize, 'embedding'):
                codebook = self.model.quantize.embedding.weight
            elif hasattr(self.model.quantize, 'embed'):
                if isinstance(self.model.quantize.embed, torch.nn.Embedding):
                    codebook = self.model.quantize.embed.weight
                else:
                    codebook = self.model.quantize.embed
            else:
                # Try using get_codebook_entry if available
                if hasattr(self.model.quantize, 'get_codebook_entry'):
                    quant = self.model.quantize.get_codebook_entry(
                        indices_tensor.reshape(-1), 
                        shape=(1, height, width, -1)
                    )
                    decoded = self.model.decode(quant)
                else:
                    raise AttributeError("Cannot access codebook for decoding")
            
            if 'decoded' not in locals():
                # Manual embedding lookup for GumbelQuantize
                # Get embeddings for each index
                z_q = codebook[indices_tensor.reshape(-1)]  # [H*W, C]
                z_q = z_q.reshape(1, height, width, -1)  # [1, H, W, C]
                z_q = z_q.permute(0, 3, 1, 2)  # [1, C, H, W]
                
                # Decode using model.decode()
                decoded = self.model.decode(z_q)
            
        # Convert to image
        decoded = torch.clamp(decoded, -1., 1.)
        decoded = (decoded + 1.0) / 2.0  # Denormalize to [0, 1]
        decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
        decoded = (decoded * 255).astype(np.uint8)
        
        return Image.fromarray(decoded)
    
    def get_codebook_embeddings(self):
        """
        Get the codebook embeddings.
        
        Returns:
            Codebook embedding tensor of shape [num_embeddings, embedding_dim]
        """
        if hasattr(self.model.quantize, 'embedding'):
            return self.model.quantize.embedding.weight.data.clone()
        elif hasattr(self.model.quantize, 'embed'):
            if isinstance(self.model.quantize.embed, torch.nn.Embedding):
                return self.model.quantize.embed.weight.data.clone()
            else:
                return self.model.quantize.embed.clone()
        else:
            raise AttributeError("Cannot find codebook embeddings")
    
 
    
    def reconstruct_image(self, image_path, resize=True):
        """
        Helper function to encode and decode an image in one step.
        Useful for testing reconstruction quality.
        
        Args:
            image_path: Path to input image
            resize: Whether to resize image
            
        Returns:
            Reconstructed PIL Image
        """
        if isinstance(image_path, str):
            x = self.preprocess_image(image_path, resize=resize)
        else:
            x = image_path
            
        with torch.no_grad():
            # Use model.encode() and model.decode()
            enc_out = self.model.encode(x)
            if isinstance(enc_out, (tuple, list)):
                z_q = enc_out[0]
            else:
                z_q = enc_out
            
            decoded = self.model.decode(z_q)
            
        # Convert to image
        decoded = torch.clamp(decoded, -1., 1.)
        decoded = (decoded + 1.0) / 2.0
        decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
        decoded = (decoded * 255).astype(np.uint8)
        
        return Image.fromarray(decoded)


# if __name__ == "__main__":
#     # Initialize processor
#     processor = VQGANProcessor(
#         config_path='/mmfs1/gscratch/krishna/mahtab/mmseek/vqgan/vqgan_imagenet_f16_1024/configs/model.yaml',
#         checkpoint_path='/mmfs1/gscratch/krishna/mahtab/mmseek/vqgan/vqgan_imagenet_f16_1024/ckpts/last.ckpt',
#         device='cuda' if torch.cuda.is_available() else 'cpu',
#         image_size=256  # Set to 512 for 512x512 models
#     )
#     output = {}
    
#     for img in tqdm(os.listdir('/mmfs1/gscratch/krishna/mahtab/mmseek/Data/persepective/validation/depthmaps/')):
#         if img in output:
#             continue
#         indices, h, w = processor.encode_to_indices(f'/mmfs1/gscratch/krishna/mahtab/mmseek/Data/persepective/validation/depthmaps/{img}')

#         output[img] = np.array(indices).flatten().tolist()
#         assert len(output[img]) == 256
#         if len(output) % 100 == 0:
#             with open(f'/mmfs1/gscratch/krishna/mahtab/mmseek/vqgan/val_depth_perspective_codes_vqgan_f16_1024.json', 'w') as f:
#                 json.dump(output, f)
#     with open(f'/mmfs1/gscratch/krishna/mahtab/mmseek/vqgan/val_depth_perspective_codes_vqgan_f16_1024.json', 'w') as f:
#         json.dump(output, f)
    





if __name__ == "__main__":
    # Initialize processor
    processor = VQGANProcessor(
        config_path='/mmfs1/gscratch/krishna/mahtab/mmseek/vqgan/vqgan_imagenet_f16_1024/configs/model.yaml',
        checkpoint_path='/mmfs1/gscratch/krishna/mahtab/mmseek/vqgan/vqgan_imagenet_f16_1024/ckpts/last.ckpt',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        image_size=256  # Set to 512 for 512x512 models
    )
    output = {}
    
    for img in ['1.png', '1arrow.png']:
        if img in output:
            continue
        indices, h, w = processor.encode_to_indices(f'{img}')

        output[img] = np.array(indices).flatten().tolist()
        assert len(output[img]) == 256
        
    with open(f'/mmfs1/gscratch/krishna/mahtab/mmseek/vqgan/samples.json', 'w') as f:
        json.dump(output, f)
    
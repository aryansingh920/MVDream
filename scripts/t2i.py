import os
import sys
import random
import argparse
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import torch
import math

from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Modified for Mac compatibility
def patch_attention_modules(model, device):
    """Replace xformers attention with standard attention for non-CUDA devices"""
    if device != "cuda":
        # Find and patch the attention mechanisms
        for name, module in model.named_modules():
            # Look for attention layers that might use xformers
            if hasattr(module, 'forward') and 'attention' in name.lower():
                original_forward = module.forward

                # Create a new forward method that doesn't use xformers
                def new_forward(self, x, context=None, *args, **kwargs):
                    h = self.heads
                    q = self.to_q(x)
                    k = self.to_k(context if context is not None else x)
                    v = self.to_v(context if context is not None else x)

                    # Reshape for attention
                    q = q.reshape(q.shape[0], q.shape[1],
                                  h, -1).permute(0, 2, 1, 3)
                    k = k.reshape(k.shape[0], k.shape[1],
                                  h, -1).permute(0, 2, 1, 3)
                    v = v.reshape(v.shape[0], v.shape[1],
                                  h, -1).permute(0, 2, 1, 3)

                    # Standard attention
                    scale = 1.0 / math.sqrt(q.shape[-1])
                    attn = torch.matmul(q, k.transpose(-1, -2)) * scale
                    attn = torch.softmax(attn, dim=-1)
                    out = torch.matmul(attn, v)

                    # Reshape back
                    out = out.permute(0, 2, 1, 3).reshape(
                        x.shape[0], x.shape[1], -1)
                    return self.to_out(out)

                # Replace the forward method if it looks like it might use xformers
                if 'xformers' in str(original_forward) or 'memory_efficient_attention' in str(original_forward):
                    module.forward = new_forward.__get__(module, type(module))

    return model


def t2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, device="cuda", camera=None, num_frames=1):
    if type(prompt) != list:
        prompt = [prompt]

    # Avoid autocast on CPU as it causes issues
    autocast_enabled = device != "cpu"

    with torch.no_grad():
        if autocast_enabled:
            with torch.autocast(device_type=device, dtype=dtype):
                c = model.get_learned_conditioning(prompt).to(device)
                c_ = {"context": c.repeat(batch_size, 1, 1)}
                uc_ = {"context": uc.repeat(batch_size, 1, 1)}
                if camera is not None:
                    c_["camera"] = uc_["camera"] = camera
                    c_["num_frames"] = uc_["num_frames"] = num_frames

                shape = [4, image_size // 8, image_size // 8]
                samples_ddim, _ = sampler.sample(S=step, conditioning=c_,
                                                 batch_size=batch_size, shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc_,
                                                 eta=ddim_eta, x_T=None)
                x_sample = model.decode_first_stage(samples_ddim)
        else:
            # Run without autocast for CPU
            c = model.get_learned_conditioning(prompt).to(device)
            c_ = {"context": c.repeat(batch_size, 1, 1)}
            uc_ = {"context": uc.repeat(batch_size, 1, 1)}
            if camera is not None:
                c_["camera"] = uc_["camera"] = camera
                c_["num_frames"] = uc_["num_frames"] = num_frames

            shape = [4, image_size // 8, image_size // 8]
            samples_ddim, _ = sampler.sample(S=step, conditioning=c_,
                                             batch_size=batch_size, shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc_,
                                             eta=ddim_eta, x_T=None)
            x_sample = model.decode_first_stage(samples_ddim)

        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0, 2, 3, 1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view",
                        help="load pre-trained model from hugginface")
    parser.add_argument("--config_path", type=str, default=None,
                        help="load model from local config (override model_name)")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="path to local checkpoint")
    parser.add_argument("--text", type=str,
                        default="an astronaut riding a horse")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=4,
                        help="num of frames (views) to generate")
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=15)
    parser.add_argument("--camera_azim", type=int, default=90)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--patch_attention", action="store_true",
                        help="Patch attention modules to work on CPU/MPS")
    parser.add_argument("--disable_xformers", action="store_true",
                        help="Disable xformers completely")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # If using MPS (Apple Silicon) but we detect compatibility issues, fall back to CPU
    if device == "mps" and "2.6.0" in torch.__version__:
        print("Warning: Using MPS with PyTorch 2.6.0 might cause issues. Consider using CPU instead.")
        print("You can force CPU with --device cpu")

    dtype = torch.float16 if args.fp16 and device == 'cuda' else torch.float32
    batch_size = max(4, args.num_frames)

    # Force model to use the specified device throughout
    if torch.__version__ >= "2.0.0":
        torch.set_default_device(device)
    print(f"load t2i model using device: {device}... ")

    # Try to monkey patch torch modules for xformers compatibility
    if args.disable_xformers or device != "cuda":
        # Disable xformers
        import sys
        sys.modules["xformers"] = None
        sys.modules["xformers.ops"] = None

    if args.config_path is None:
        model = build_model(args.model_name, ckpt_path=args.ckpt_path)
    else:
        assert args.ckpt_path is not None, "ckpt_path must be specified!"
        config = OmegaConf.load(args.config_path)
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))

    # Apply attention patching if requested
    if args.patch_attention:
        print("Patching attention modules for compatibility...")
        model = patch_attention_modules(model, device)

    model.device = device
    model.to(device)
    model.eval()

    # Fix DDIM sampler to use the specified device
    sampler = DDIMSampler(model)
    # Make sure the model's cond_stage_model also uses the right device
    if hasattr(model, 'cond_stage_model'):
        model.cond_stage_model.device = device

    # Get unconditioned embedding
    with torch.no_grad():
        uc = model.get_learned_conditioning([""])
        if uc.device != torch.device(device):
            uc = uc.to(device)

    print("load t2i model done.")

    # pre-compute camera matrices
    if args.use_camera:
        camera = get_camera(args.num_frames, elevation=args.camera_elev,
                            azimuth_start=args.camera_azim, azimuth_span=args.camera_azim_span)
        camera = camera.repeat(batch_size//args.num_frames, 1).to(device)
    else:
        camera = None

    t = args.text + args.suffix
    set_seed(args.seed)
    images = []

    # Reduce the number of parallel generations on CPU to avoid memory issues
    num_batches = 3 if device != "cpu" else 1
    print(f"Generating {num_batches} batches of images...")

    for j in range(num_batches):
        print(f"Generating batch {j+1}/{num_batches}...")
        img = t2i(model, args.size, t, uc, sampler, step=50 if device != "cpu" else 20,
                  scale=10, batch_size=batch_size, ddim_eta=0.0,
                  dtype=dtype, device=device, camera=camera, num_frames=args.num_frames)
        img = np.concatenate(img, 1)
        images.append(img)

    images = np.concatenate(images, 0)
    Image.fromarray(images).save(f"sample.png")
    print(f"Saved output to sample.png")

import argparse
import os
import re
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import cv2
import einops
import gradio as gr
import numpy as np
import torch
from PIL import Image
from skimage.measure import label, regionprops
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from functools import partial

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from deepfloyd_if.modules.stage_II import IFStageII


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for the Colorize Diffusion application."""
    # Model paths
    model_load_path: str = "./models/G.pt"
    sam_checkpoint_path: str = "./models/sam_vit_h_4b8939.pth"
    empty_prompt_path: str = "./empty_prompt_1_77_4096.npz"
    
    # Model settings
    aux_channels: int = 7
    use_control_net: bool = True
    force_aux_8: bool = False
    deepfloyd_model: str = "IF-II-M-v1.0"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Inference settings
    steps: str = "super27"
    aug_level: float = 0.0
    support_noise_less_qsample_steps: int = 0
    dynamic_thresholding_p: float = 0.95
    dynamic_thresholding_c: float = 1.0
    sample_loop: str = "ddpm"
    
    # UI settings
    max_colors: int = 128
    default_image_resolution: int = 256
    
    # Server settings
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False
    debug: bool = False


class ColorizeDiffusion:
    """Main class for Colorize Diffusion functionality."""
    
    def __init__(self, config: Config):
        self.config = config
        self.sam_model = None
        self.predictor = None
        self.mask_generator = None
        self.stage_2_model = None
        
        # State management
        self.current_masks: List[Dict] = []
        self.mask_list: List[np.ndarray] = []
        self.rgb_list: List[Tuple[int, int, int]] = []
        self.grad_channels: Optional[np.ndarray] = None
        
        # Ensure required directories exist
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        model_dir = Path(self.config.model_load_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
    
    def load_models(self, model_path: Optional[str] = None, sam_path: Optional[str] = None) -> bool:
        """Load SAM and Stage 2 models.
        
        Args:
            model_path: Path to the model checkpoint
            sam_path: Path to the SAM checkpoint
            
        Returns:
            True if models loaded successfully, False otherwise
        """       
        try:
            # Use provided paths or defaults
            model_path = model_path or self.config.model_load_path
            sam_path = sam_path or self.config.sam_checkpoint_path
            
            # Validate paths
            if not os.path.exists(sam_path):
                logger.error(f"SAM checkpoint not found at: {sam_path}")
                gr.Info(f"SAM checkpoint not found at: {sam_path}, please change path and retry")
                return False
            
            if not os.path.exists(model_path):
                logger.error(f"Model checkpoint not found at: {model_path}")
                gr.Info(f"Model checkpoint not found at: {model_path}, please change path and retry")
                return False
            
            # Load SAM
            gr.Info("Loading SAM model...")
            logger.info("Loading SAM model...")
            self.sam_model = sam_model_registry["vit_h"](checkpoint=sam_path)
            self.sam_model.to(device=self.config.device)
            self.predictor = SamPredictor(self.sam_model)
            self.mask_generator = SamAutomaticMaskGenerator(model=self.sam_model)
            
            # Load Stage 2 model
            gr.Info("Loading Stage 2 model...")
            logger.info("Loading Stage 2 model...")
            model_kwargs = {
                'doCN': self.config.use_control_net,
                'aux_ch': self._calculate_aux_channels(),
                'attention_resolutions': '32,16'
            }
            
            self.stage_2_model = IFStageII(
                self.config.deepfloyd_model,
                device=self.config.device,
                filename=model_path,
                model_kwargs=model_kwargs
            )
            
            self._configure_stage2_model()
            
            gr.Info("Models loaded successfully")
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _calculate_aux_channels(self) -> int:
        """Calculate auxiliary channels based on configuration."""
        if self.config.force_aux_8:
            return self.config.aux_channels + (8 - self.config.aux_channels % 8) % 8
        return self.config.aux_channels
    
    def _configure_stage2_model(self):
        """Configure Stage 2 model settings."""
        if self.stage_2_model is None:
            return
            
        # Set data type
        self.stage_2_model.model.dtype = torch.float32
        self.stage_2_model.model.precision = '32'
        
        if self.config.use_control_net:
            self.stage_2_model.model.control_model.dtype = self.stage_2_model.model.dtype
            self.stage_2_model.model.control_model.precision = self.stage_2_model.model.precision
        
        # Convert parameters to correct dtype
        for name, p in self.stage_2_model.model.named_parameters():
            p.data = p.type(self.stage_2_model.model.dtype)
            p.requires_grad = False
        
        self.stage_2_model.model.eval()
    
    def resize_image(self, image: np.ndarray, resolution: int) -> np.ndarray:
        """Resize image to specified resolution while maintaining aspect ratio.
        
        Args:
            image: Input image as numpy array
            resolution: Target resolution
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        scale = resolution / min(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Round to nearest 16 for model compatibility
        new_h = (new_h // 16) * 16
        new_w = (new_w // 16) * 16
        
        interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    def center_crop(self, arr: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
        """Center crop an array to specified dimensions.
        
        Args:
            arr: Input array
            new_height: Target height
            new_width: Target width
            
        Returns:
            Cropped array
        """
        h, w = arr.shape[:2]
        start_y = max(0, (h - new_height) // 2)
        start_x = max(0, (w - new_width) // 2)
        return arr[start_y:start_y + new_height, start_x:start_x + new_width]
    
    def compute_gradients(self, image: np.ndarray, mode: str) -> np.ndarray:
        """Compute gradient channels based on mode.
        
        Args:
            image: Input image
            mode: One of 'Gradient', 'Luminance', or 'Threshold'
            
        Returns:
            Gradient channels
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if mode == 'Gradient':
            gradients = np.gradient(gray_image)
            return np.stack([(g + 255) / (2 * 255.) for g in gradients], axis=-1)
        elif mode == 'Luminance':
            luminance = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 0] / 255.0
            return np.stack([luminance, luminance], axis=-1)
        elif mode == 'Threshold':
            gradients = np.gradient(gray_image)
            return np.stack([np.greater(np.abs(g), 8).astype(np.float32) for g in gradients], axis=-1)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def generate_masks(self, image: Image.Image) -> Optional[Image.Image]:
        """Generate masks using SAM.
        
        Args:
            image: Input PIL image
            
        Returns:
            Image with masks overlaid or None if no masks found
        """
        if self.mask_generator is None:
            raise ValueError("Models haven't been loaded. Please load models first.")
        
        arr = np.asarray(image)
        self.current_masks = self.mask_generator.generate(arr)
        
        if len(self.current_masks) == 0:
            logger.warning("No masks generated")
            return None
        
        # Sort masks by area
        sorted_masks = sorted(self.current_masks, key=lambda x: x['area'], reverse=True)
        
        # Create visualization
        h, w = sorted_masks[0]['segmentation'].shape
        overlay = np.ones((h, w, 4))
        overlay[:, :, 3] = 0
        
        for mask in sorted_masks:
            m = mask['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            overlay[m] = color_mask
        
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
        result = image.copy()
        result.paste(overlay_img, mask=overlay_img)
        
        return result
    
    def fill_color_at_point(self, palette: Image.Image, color: str, 
                           control_image: Image.Image, x: int, y: int) -> Tuple[Image.Image, Image.Image]:
        """Fill color at specified point.
        
        Args:
            palette: Palette image
            color: Color string in rgba format
            control_image: Control image
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Updated control image and palette
        """
        # Parse color
        numbers = re.findall(r"[\d.]+", color)
        r, g, b = tuple(int(round(float(n))) for n in numbers[:3])
        
        width, height = control_image.size
        
        # Find the mask containing the clicked point
        best_mask = None
        min_area = float('inf')
        
        for mask in self.current_masks:
            mask['segmentation'] = self.center_crop(mask['segmentation'], height, width)
            if mask['segmentation'][x, y] and mask['area'] < min_area:
                best_mask = mask['segmentation']
                min_area = mask['area']
        
        # If no mask found, use background
        if best_mask is None:
            background = np.ones((height, width), dtype=bool)
            for mask in self.current_masks:
                background[mask['segmentation']] = False
            background[:height - 65, :] = False  # Exclude UI area
            best_mask = background
        
        # Apply color
        img_arr = np.array(control_image)
        img_arr[best_mask] = [r, g, b]
        
        palette_arr = np.array(palette)
        palette_arr[best_mask] = [r, g, b]
        
        # Update state
        self.rgb_list.append((r, g, b))
        self.mask_list.append(best_mask)
        
        return Image.fromarray(img_arr), Image.fromarray(palette_arr)
    
    @torch.no_grad()
    def generate_image(self, control_image: Image.Image, use_texture_dropout: bool) -> List[np.ndarray]:
        """Generate colorized image.
        
        Args:
            control_image: Control image with color information
            use_texture_dropout: Whether to use texture dropout
            
        Returns:
            List of generated images
        """
        if self.stage_2_model is None:
            raise ValueError("Models haven't been loaded. Please load models first.")
        
        # Prepare conditioning
        patchImage = np.asarray(control_image)
        height, width = patchImage.shape[:2]
        
        # Initialize detection map
        detected_map = np.ones((height, width, 7), dtype=np.float32)
        texture_channel = np.ones((height, width))
        color_indicator = np.full((height, width), 8 / (2 * self.config.max_colors))
        
        # Set base values
        detected_map[:, :, :3] = patchImage / 255.0
        detected_map[:, :, 3:5] = self.grad_channels
        
        # Apply masks
        for mask, rgb in zip(self.mask_list, self.rgb_list):
            detected_map[:, :, 0][mask] = rgb[0] / 255.0
            detected_map[:, :, 1][mask] = rgb[1] / 255.0
            detected_map[:, :, 2][mask] = rgb[2] / 255.0
            color_indicator[mask] = 1 / (2 * self.config.max_colors)
            color_indicator[~mask] = 1.0
            
            if use_texture_dropout:
                texture_channel[mask] = 0
        
        # Apply texture dropout
        if use_texture_dropout:
            detected_map[:, :, 3][texture_channel == 0] = 0
            detected_map[:, :, 4][texture_channel == 0] = 0
        
        detected_map[:, :, 5] = color_indicator
        detected_map[:, :, 6] = texture_channel
        
        # Prepare tensors
        control = torch.from_numpy(detected_map).float().to(self.config.device)
        color = torch.from_numpy(detected_map[:, :, :3]).float().to(self.config.device)
        
        control = control.unsqueeze(0).permute(0, 3, 1, 2)
        color = color.unsqueeze(0).permute(0, 3, 1, 2)
        
        # Load text prompts
        if not os.path.exists(self.config.empty_prompt_path):
            raise FileNotFoundError(f"Empty prompt file not found: {self.config.empty_prompt_path}")
        
        text_prompts = torch.from_numpy(
            np.load(self.config.empty_prompt_path, allow_pickle=True)['arr']
        ).to(self.config.device)
        
        # Generate image
        with torch.autocast(self.config.device, dtype=torch.float16):
            out, metadata = self.stage_2_model.embeddings_to_image(
                sample_timestep_respacing=self.config.steps,
                low_res=2 * color - 1,
                support_noise=2 * color - 1,
                support_noise_less_qsample_steps=self.config.support_noise_less_qsample_steps,
                seed=None,
                t5_embs=text_prompts[0:1, ...],
                hint=2 * control - 1,
                aug_level=self.config.aug_level,
                sample_loop=self.config.sample_loop,
                dynamic_thresholding_p=self.config.dynamic_thresholding_p,
                dynamic_thresholding_c=self.config.dynamic_thresholding_c
            )
        
        out = (out + 1) / 2
        result = (255 * out.squeeze().cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
        
        return [result]
    
    def cleanup(self):
        """Clean up resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.current_masks = []
        self.mask_list = []
        self.rgb_list = []
        self.grad_channels = None


class GradioInterface:
    """Gradio interface for Colorize Diffusion."""
    
    def __init__(self, colorize_diffusion: ColorizeDiffusion):
        self.cd = colorize_diffusion
        self.interface = None
    
    def preprocess_image(self, input_image: Optional[Image.Image], 
                        resolution: int, mode: str) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Preprocess input image."""
        if input_image is None:
            return None, None
        
        # Clear previous state
        self.cd.mask_list.clear()
        self.cd.rgb_list.clear()
        
        # Resize and compute gradients
        img_array = np.asarray(input_image)
        resized = self.cd.resize_image(img_array, resolution)
        self.cd.grad_channels = self.cd.compute_gradients(resized, mode)
        
        resized_image = Image.fromarray(resized)
        return resized_image, resized_image
    
    def on_palette_click(self, palette: Image.Image, color: str, 
                        control_image: Image.Image, evt: gr.SelectData) -> Tuple[Image.Image, Image.Image]:
        """Handle palette click event."""
        if evt is None:
            return control_image, palette
        
        y, x = evt.index[0], evt.index[1]
        return self.cd.fill_color_at_point(palette, color, control_image, x, y)
    

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        with gr.Blocks(
            title="Colorize Diffusion",
            theme=gr.themes.Soft(),
            elem_id="main-interface"
        ) as interface:
            gr.Markdown("# Colorize Diffusion\n### AI-powered image colorization with fine-grained control")
            
            # Add workflow instructions
            with gr.Row():
                gr.Markdown("""
                ## 🎨 How to Use:
                1. **Upload** your image below → 2. **Generate Masks** → 3. **Click segments** to color them → 4. **Generate** final image
                """)
            
            with gr.Row(equal_height=False, variant="panel"):
                # Input column
                with gr.Column(scale=1):
                    gr.Markdown("## Step 1: Upload & Settings")
                    
                    image_resolution = gr.Slider(
                        label="Image Resolution",
                        minimum=256,
                        maximum=1088,
                        value=self.cd.config.default_image_resolution,
                        step=64
                    )
                    
                    mode = gr.Radio(
                        ["Luminance", "Gradient", "Threshold"],
                        label="Gradient Mode",
                        value="Gradient"
                    )
                    
                    with gr.Group():
                        model_path = gr.Textbox(
                            value=self.cd.config.model_load_path,
                            label="Model Path"
                        )
                        sam_path = gr.Textbox(
                            value=self.cd.config.sam_checkpoint_path,
                            label="SAM Path"
                        )
                        load_models_btn = gr.Button(
                            "Load Models",
                            variant="secondary"
                        )
                    
                    input_image = gr.Image(
                        label="📤 Upload Your Image Here (Only Upload Spot)",
                        sources=['upload'],
                        type="pil"
                    )
                    
                    gr.Markdown("*⚠️ Only upload images above - other images are auto-generated*")
                
                # Control column
                with gr.Column(scale=2):
                    gr.Markdown("## Step 2 & 3: Generate Masks & Color")
                    
                    control_image = gr.Image(
                        label="🖼️ Processed Image (Auto-Generated)",
                        type="pil",
                        interactive=False
                    )
                    
                    with gr.Row():
                        color_picker = gr.ColorPicker(
                            label="🎨 Pick Your Color",
                            value="#FF0000"
                        )
                        get_mask_btn = gr.Button(
                            "🎯 Generate Masks",
                            variant="secondary",
                            size="lg"
                        )
                    
                    # More descriptive instructions for the palette
                    gr.Markdown("**👆 First click 'Generate Masks', then click on segments below to color them:**")
                    
                    palette = gr.Image(
                        label="🖱️ Segmented Image - Click Any Segment to Apply Color",
                        type="pil",
                        interactive=True,
                        placeholder="Masks will appear here after clicking 'Generate Masks' above"
                    )
                    
                    # Status indicator to show workflow progress
                    with gr.Row():
                        status_text = gr.Markdown("**Status:** Upload an image to begin")
                    
                    texture_dropout = gr.Checkbox(
                        label="Texture Dropout",
                        info="Remove texture in colored regions"
                    )
                    
                    generate_btn = gr.Button(
                        "🚀 Generate Final Image",
                        variant="primary",
                        size="lg"
                    )
                
                # Output column
                with gr.Column(scale=1):
                    gr.Markdown("## Step 4: Final Result")
                    result_gallery = gr.Gallery(
                        label="Generated Images",
                        show_label=False,
                        elem_id="gallery",
                        preview=True
                    )
            
            # Wire up events with status updates
            def update_status_after_upload(image, resolution, mode):
                if image is None:
                    return None, None, "**Status:** Upload an image to begin"
                
                # Call the original preprocessing
                control, palette_img = self.preprocess_image(image, resolution, mode)
                return control, palette_img, "**Status:** ✅ Image uploaded! Now click 'Generate Masks'"
            
            def update_status_after_masks(image):
                if image is None:
                    return None, "**Status:** Please upload an image first"
                
                result = self.cd.generate_masks(image)
                if result is None:
                    return None, "**Status:** ❌ No masks found. Try a different image."
                
                return result, "**Status:** ✅ Masks generated! Click on segments to color them, then generate final image"
            
            load_models_btn.click(
                fn=lambda mp, sp: (
                    gr.Info("Loading models..."),
                    self.cd.load_models(mp, sp),
                )[0],
                inputs=[model_path, sam_path],
                outputs=[]
            )
            
            input_image.change(
                fn=update_status_after_upload,
                inputs=[input_image, image_resolution, mode],
                outputs=[control_image, palette, status_text]
            )
            
            get_mask_btn.click(
                fn=update_status_after_masks,
                inputs=[palette],
                outputs=[palette, status_text]
            )
            
            # Use original palette click handler and update status separately
            palette.select(
                fn=self.on_palette_click,
                inputs=[palette, color_picker, control_image],
                outputs=[control_image, palette]
            )
            
            # Update status after palette click
            palette.select(
                fn=lambda: "**Status:** ✅ Segment colored! Add more colors or generate final image",
                inputs=[],
                outputs=[status_text]
            )
            
            generate_btn.click(
                fn=self.cd.generate_image,
                inputs=[control_image, texture_dropout],
                outputs=[result_gallery]
            )
            
            self.interface = interface
            return interface

    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        if self.interface is None:
            self.create_interface()
        
        self.interface.launch(**kwargs)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Colorize Diffusion")
    
    # Model paths
    parser.add_argument("--model_load_path", type=str, default="./models/G.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--sam_path", type=str, default="./models/sam_vit_h_4b8939.pth",
                       help="Path to SAM checkpoint")
    parser.add_argument("--empty_prompt_path", type=str, default="./empty_prompt_1_77_4096.npz",
                       help="Path to empty prompt file")
    
    # Server settings
    parser.add_argument("--server_name", "-addr", type=str, default="0.0.0.0",
                       help="Server address")
    parser.add_argument("--server_port", "-port", type=int, default=7860,
                       help="Server port")
    parser.add_argument("--share", action="store_true",
                       help="Share the app publicly")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    # Model settings
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create configuration
    config = Config(
        model_load_path=args.model_load_path,
        sam_checkpoint_path=args.sam_path,
        empty_prompt_path=args.empty_prompt_path,
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        debug=args.debug,
        device=args.device
    )
    
    # Initialize application
    try:
        logger.info("Initializing Colorize Diffusion...")
        colorize_diffusion = ColorizeDiffusion(config)
        
        # Create and launch interface
        interface = GradioInterface(colorize_diffusion)
        
        logger.info(f"Launching server on {config.server_name}:{config.server_port}")
        interface.launch(
            server_name=config.server_name,
            server_port=config.server_port,
            share=config.share,
            debug=config.debug
        )
        
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise
    finally:
        # Cleanup
        if 'colorize_diffusion' in locals():
            colorize_diffusion.cleanup()


if __name__ == "__main__":
    main()
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from functools import partial

import cv2
import einops
import gradio as gr
import matplotlib
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

from deepfloyd_if.modules.stage_II import IFStageII

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Configuration class for the application."""
    image_size: int = 256
    aux_channels: int = 7
    do_cn: bool = True
    force_aux_8: bool = False
    model_path: str = 'IF-II-M-v1.0'
    max_colors: int = 128
    device: str = 'cuda:0'
    
    # Gradio settings
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = True
    debug: bool = True


class ImageProcessor:
    """Handles image processing operations."""
    
    @staticmethod
    def resize_image(input_image: np.ndarray, resolution: int) -> np.ndarray:
        """Resize image while maintaining aspect ratio and ensuring divisibility by 16."""
        H, W, C = input_image.shape
        k = float(resolution) / min(H, W)
        new_H = int(np.round(H * k / 16.0)) * 16
        new_W = int(np.round(W * k / 16.0)) * 16
        
        interpolation = cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
        return cv2.resize(input_image, (new_W, new_H), interpolation=interpolation)
    
    @staticmethod
    def center_crop(arr: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
        """Center crop image to specified dimensions."""
        h, w = arr.shape[:2]
        start_y = max(0, (h - new_height) // 2)
        start_x = max(0, (w - new_width) // 2)
        return arr[start_y:start_y+new_height, start_x:start_x+new_width]
    
    @staticmethod
    def sort_colors(color_array: np.ndarray) -> np.ndarray:
        """Sort colors by HSV values."""
        color_array = np.clip(color_array, 0, 255).astype(np.uint8)
        reshaped_array = color_array.reshape((-1, 1, 3))
        hsv_colors = cv2.cvtColor(reshaped_array, cv2.COLOR_RGB2HSV)
        hsv_colors = hsv_colors.reshape(-1, 3)
        
        # Sort by hue, saturation, then value
        sorted_indices = np.lexsort((hsv_colors[:, 2], hsv_colors[:, 1], hsv_colors[:, 0]))
        return color_array[sorted_indices]
    
    def get_gradient_channels(self, input_image: Image.Image, 
                            image_resolution: int, mode: str) -> Tuple[Image.Image, np.ndarray]:
        """Extract gradient channels from input image."""
        if input_image is None:
            raise ValueError("Input image cannot be None")
            
        img = self.resize_image(np.asarray(input_image), image_resolution)
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gradients = np.gradient(gray_image)
        resized_image = Image.fromarray(img)
        
        if mode == 'Gradient':
            grad_channels = np.stack([(g + 255) / (2 * 255.) for g in gradients], axis=-1)
        elif mode == 'Luminance':
            luminance = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 0] / 255.0
            grad_channels = np.stack([luminance, luminance], axis=-1)
        elif mode == 'Threshold':
            grad_channels = np.stack([np.greater(np.abs(g), 8).astype(np.float32) for g in gradients], axis=-1)
        else:
            raise ValueError(f"Invalid mode: {mode}")
            
        return resized_image, grad_channels


class PaletteProcessor:
    """Handles palette application and color transfer operations."""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def apply_palette(self, pil_image: Image.Image, pil_target: Optional[Image.Image], 
                     num_colors: int, transfer_method: str, cmap: str, 
                     blend: float) -> List[np.ndarray]:
        """Apply color palette to source image."""
        if num_colors is None:
            raise ValueError("Invalid value provided for colors")
            
        num_colors = int(num_colors)
        palette_image = pil_image.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
        pal_rgb = np.asarray(palette_image.convert('RGB'))
        indexed_palette = np.array(palette_image)
        
        src_palette = np.array(palette_image.getpalette()[0:3*num_colors]).reshape(num_colors, 3) / 255.0
        height, width = indexed_palette.shape[:2]
        full_color = np.zeros((height, width, 3), dtype=np.float32)
        
        if transfer_method == 'colormap':
            dst_pal, dst_rgb, distance_matrix_d = self._apply_colormap(
                src_palette, num_colors, height, width, cmap
            )
        else:
            dst_pal, dst_rgb, distance_matrix_d = self._apply_target_palette(
                pil_target, src_palette, indexed_palette, num_colors, 
                height, width, transfer_method
            )
        # Apply color matching
        _, matching = linear_sum_assignment(-distance_matrix_d)

        for col_idx in range(num_colors):
            mask = indexed_palette == col_idx
            if transfer_method == 'colormap':
                color_map = matplotlib.colormaps[cmap]
                map_query = np.arange(0, 1 + (1/(2.*num_colors)), 1./(num_colors-1))
                cm = color_map(map_query)[:, 0:3]
                get_color = cm[matching[col_idx], :]
            else:
                targ_int = pil_target.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
                targ_col = np.array(targ_int.convert('RGB')) / 255.0
                targ_int = np.array(targ_int)
                get_mask = targ_int == matching[col_idx]
                get_color = targ_col[get_mask][0]
            full_color[mask] = blend * get_color + (1 - blend) * src_palette[col_idx]
        return [pal_rgb, dst_rgb, (full_color * 255).astype(np.uint8)]
    
    def _apply_colormap(self, src_palette: np.ndarray, num_colors: int, 
                       height: int, width: int, cmap: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply colormap-based palette transfer."""
        color_map = matplotlib.colormaps[cmap]
        map_query = np.arange(0, 1 + (1/(2.*num_colors)), 1./(num_colors-1))
        cm = color_map(map_query)[:, 0:3]
        distance_matrix_d = np.exp(-distance_matrix(src_palette, cm, p=2))
        
        # Create discrete palette
        discrete_indices = np.repeat(np.arange(num_colors), height // num_colors)
        if len(discrete_indices) < height:
            discrete_indices = np.pad(discrete_indices, (0, height - len(discrete_indices)), mode='edge')
        discrete_palette = cm[discrete_indices]
        dst_pal = np.repeat(discrete_palette[:, np.newaxis, :], width, axis=1)
        
        # Create smoothed colormap
        smoothed_indices = np.linspace(0, 1, height)
        smoothed_colormap = color_map(smoothed_indices)[:, :3]
        dst_rgb = np.repeat(smoothed_colormap[:, np.newaxis, :], width, axis=1)
        
        return dst_pal, (255*dst_rgb).astype(np.uint8), distance_matrix_d
    
    def _apply_target_palette(self, pil_target: Image.Image, src_palette: np.ndarray,
                            indexed_palette: np.ndarray, num_colors: int,
                            height: int, width: int, transfer_method: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply target image-based palette transfer."""
        targ_int = pil_target.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
        targ_rgb = np.array(targ_int.convert('RGB'))
        targ_palette = np.array(targ_int.getpalette()[0:3*num_colors]).reshape(num_colors, 3) / 255.0
        targ_int_array = np.array(targ_int)
        
        if transfer_method == 'color':
            distance_matrix_d = np.exp(-distance_matrix(src_palette, targ_palette, p=2))
        elif transfer_method == 'negative':
            distance_matrix_d = np.exp(distance_matrix(src_palette, targ_palette, p=2))
        elif transfer_method == 'frequency':
            distance_matrix_d = self._calculate_frequency_distance(indexed_palette, targ_int_array, num_colors)
        elif transfer_method == 'int':
            distance_matrix_d = np.exp(-distance_matrix(
                np.sum(src_palette, 1, keepdims=True), 
                np.sum(targ_palette, 1, keepdims=True), p=2
            ))
        else:
            raise ValueError(f"Invalid transfer method: {transfer_method}")
         #check if quantization ended up with correct amount of colors

        sortedSrc, countsSrc = np.unique(indexed_palette, return_counts=True)
        sortedTarg, countsTarg = np.unique(targ_int_array, return_counts=True) 

        if len(countsSrc) != len(countsTarg):
            print(len(countsSrc))
            print(len(countsTarg))
            raise ValueError("Palettes don't have same number of colors, try different number of colors")
         
        row_indices = np.arange(height)
        color_indices = row_indices * num_colors // height
        targ_palette_sorted = self.sort_colors(targ_palette * 255) / 255.0
        image_array = targ_palette_sorted[color_indices][:, np.newaxis, :]
        #Palette visualization of target image
        dst_pal = np.repeat(image_array, width, axis=1)

        dst_rgb = np.asarray(pil_target)
        
        return dst_pal, dst_rgb, distance_matrix_d
    
    def _calculate_frequency_distance(self, indexed_palette: np.ndarray, 
                                    targ_int_array: np.ndarray, num_colors: int) -> np.ndarray:
        """Calculate frequency-based distance matrix."""
        _, counts_src = np.unique(indexed_palette, return_counts=True)
        _, counts_targ = np.unique(targ_int_array, return_counts=True)
        
        freq_src = counts_src / float(np.sum(counts_src))
        freq_targ = counts_targ / float(np.sum(counts_targ))
        
        freq_src = freq_src.reshape(num_colors, 1)
        freq_targ = freq_targ.reshape(num_colors, 1)
        
        return np.exp(-distance_matrix(freq_src, freq_targ, p=2))
    
    def sort_colors(self, color_array: np.ndarray) -> np.ndarray:
        """Sort colors using ImageProcessor method."""
        return ImageProcessor.sort_colors(color_array)


class DiffusionModel:
    """Handles the diffusion model operations."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.stage_2: Optional[IFStageII] = None
        self.empty_prompt_embeddings = None
        
    def load_model(self, model_load_path: str) -> None:
        """Load the diffusion model."""
        try:
            model_kwargs = {
                'doCN': self.config.do_cn,
                'aux_ch': self.config.aux_channels + (8 - self.config.aux_channels % 8) % 8 if self.config.force_aux_8 else self.config.aux_channels,
                'attention_resolutions': '32,16'
            }
            
            self.stage_2 = IFStageII(
                self.config.model_path,
                device=self.config.device,
                filename=model_load_path,
                model_kwargs=model_kwargs
            )
            
            self._configure_model()
            self._load_empty_prompt()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _configure_model(self) -> None:
        """Configure model settings."""
        self.stage_2.model.dtype = torch.float32
        self.stage_2.model.precision = '32'
        
        if self.config.do_cn:
            self.stage_2.model.control_model.dtype = self.stage_2.model.dtype
            self.stage_2.model.control_model.precision = self.stage_2.model.precision
        
        for name, param in self.stage_2.model.named_parameters():
            param.data = param.type(self.stage_2.model.dtype)
            param.requires_grad = False
        
        self.stage_2.model.eval()
    
    def _load_empty_prompt(self) -> None:
        """Load empty prompt embeddings."""
        try:
            self.empty_prompt_embeddings = torch.from_numpy(
                np.load('empty_prompt_1_77_4096.npz', allow_pickle=True)['arr']
            ).to(self.config.device)
        except FileNotFoundError:
            logger.warning("Empty prompt embeddings file not found. Using default.")
            # Create default empty embeddings if file not found
            self.empty_prompt_embeddings = torch.zeros(1, 77, 4096, device=self.config.device)
    
    def generate(self, palette: Image.Image, num_colors: int, 
                texture_option: bool, grad_channels: np.ndarray) -> List[np.ndarray]:
        """Generate image using the diffusion model."""
        if self.stage_2 is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            with torch.no_grad():
                return self._run_inference(palette, num_colors, texture_option, 
                                         grad_channels)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _run_inference(self, palette: Image.Image, num_colors: int,
                      texture_option: bool, grad_channels: np.ndarray) -> List[np.ndarray]:
        """Run model inference."""
        pal = np.asarray(palette)
        height, width = pal.shape[:2]
        
        # Prepare conditioning
        detected_map = self._prepare_conditioning(
            pal, height, width, num_colors, texture_option, grad_channels
        )
        
        # Convert to tensors
        control = torch.from_numpy(detected_map.copy()).cuda()
        color = torch.from_numpy(detected_map[:, :, :3].copy()).cuda()
        
        # Reshape for model
        control = control.unsqueeze(0)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        
        color = color.unsqueeze(0)
        color = einops.rearrange(color, 'b h w c -> b c h w')
        
        # Get text embeddings
        text_prompts = self.empty_prompt_embeddings.repeat(1, 1, 1)
        
        # Run inference
        with torch.autocast("cuda", dtype=torch.float16):
            out, metadata = self.stage_2.embeddings_to_image(
                sample_timestep_respacing="super27",
                low_res=2*color-1,
                support_noise=2*color-1,
                support_noise_less_qsample_steps=0,
                seed=None,
                t5_embs=text_prompts[0:1, ...],
                hint=2*control-1,
                aug_level=0.0,
                sample_loop='ddpm',
                dynamic_thresholding_p=0.95,
                dynamic_thresholding_c=1.0
            )
        
        out = (out + 1) / 2
        result = (255 * out.squeeze().cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
        
        return [result]
    
    def _prepare_conditioning(self, pal: np.ndarray, height: int, width: int,
                            num_colors: int, texture_option: bool,
                            grad_channels: np.ndarray) -> np.ndarray:
        """Prepare conditioning tensor for the model."""
        detected_map = np.ones((height, width, 7), dtype=np.float32)
        
        # Color channels
        detected_map[:, :, :3] = pal / 255.0
        
        # Gradient channels
        detected_map[:, :, 3:5] = grad_channels
        
        # Color indicator
        color_indicator = np.full((height, width), num_colors / (2 * self.config.max_colors))
        detected_map[:, :, 5] = color_indicator
        
        # Texture channel
        texture_channel = np.zeros((height, width)) if texture_option else np.ones((height, width))
        detected_map[:, :, 6] = texture_channel
        
        return detected_map




class ColorizeDiffusionApp:
    """Main application class with proper state management."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.image_processor = ImageProcessor()
        self.palette_processor = PaletteProcessor(config)
        self.diffusion_model = DiffusionModel(config)
        self.current_grad_channels = None
        self.current_mode = None
        self.current_input_image = None

    def clear_results(self) -> List[Image.Image]:
        """Clear the results gallery to show progress bar."""
        return [] 

    def load_models(self, model_path: str) -> str:
        """Load the diffusion models and reset state."""
        try:
            self.diffusion_model.load_model(model_path)
            # Reset state when model changes
            self._reset_processing_state()
            return "Models loaded successfully! Please reprocess your input image."
        except Exception as e:
            return f"Failed to load models: {e}"
    
    def _reset_processing_state(self):
        """Reset processing state when model or mode changes."""
        self.current_grad_channels = None
        # Note: We keep current_input_image to allow reprocessing
    
    def process_input_image(self, input_image: Optional[Image.Image],
                          image_resolution: int, mode: str) -> Optional[Image.Image]:
        """Process input image and extract gradient channels."""
        if input_image is None:
            self._reset_processing_state()
            return None
            
        # Reset state if mode changed
        if self.current_mode != mode:
            self.current_mode = mode
            self._reset_processing_state()
            
        try:
            resized_image, grad_channels = self.image_processor.get_gradient_channels(
                input_image, image_resolution, mode
            )
            self.current_grad_channels = grad_channels
            self.current_input_image = resized_image
            return resized_image
        except Exception as e:
            logger.error(f"Failed to process input image: {e}")
            self._reset_processing_state()
            return None

    def apply_palette(self, input_image: Optional[Image.Image], 
                        color_image: Optional[Image.Image], num_colors: str,
                        transfer_method: str, cmap: str, blend: float) -> List[Optional[Image.Image]]:
            """Apply color palette to the input image."""
            if input_image is None:
                return [None, None, None]
                
            try:
                num_colors_int = int(num_colors)
                results = self.palette_processor.apply_palette(
                    input_image, color_image, num_colors_int, transfer_method, cmap, blend
                )
                return [Image.fromarray(result) for result in results]
            except Exception as e:
                logger.error(f"Failed to apply palette: {e}")
                return [None, None, None]
    
    def generate_image(self, palette: Optional[Image.Image], num_colors: str,
                      texture_option: bool) -> List[Image.Image]:
        """Generate final image using diffusion model."""
        if palette is None:
            return []
            
        if self.current_grad_channels is None:
            logger.error("No gradient channels available. Please process an input image first.")
            return []
            
        try:
            num_colors_int = int(num_colors)
            results = self.diffusion_model.generate(
                palette, num_colors_int, texture_option, 
                self.current_grad_channels
            )
            return [Image.fromarray(result) for result in results]
        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            return []
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface with proper state management."""
        with gr.Blocks(
            title="Colorize Diffusion",
            theme=gr.themes.Soft(),
            elem_id="main-interface",
            analytics_enabled=False
        ) as interface:
            
            with gr.Row(elem_id="content-row", equal_height=False, variant="panel"):
                # Controls column
                with gr.Column():
                    image_resolution = gr.Slider(
                        label="Image Resolution", minimum=256, maximum=1088, 
                        value=256, step=64
                    )
                    model_path = gr.Textbox(
                        value="models/G.pt", label="Model Path"
                    )
                    load_models_btn = gr.Button("Load Models", variant="secondary", size="sm")
                    load_status = gr.Textbox(label="Load Status", interactive=False)
                    
                    mode = gr.Radio(
                        ["Luminance", "Gradient", "Threshold"], 
                        label="Processing Mode", value="Gradient"
                    )
                    transfer_method = gr.Radio(
                        ["colormap", "color", "frequency", "negative", "int"],
                        label="Transfer Method", value="color"
                    )
                    colors = gr.Radio(
                        ["16", "32", "64"], label="Number of Colors", value="32"
                    )
                    cmap = gr.Textbox(value="viridis", label="Colormap")
                    blend = gr.Slider(
                        minimum=0.0, maximum=1.0, label="Blend Factor", value=1.0
                    )
                
                # Input/Processing column
                with gr.Column():
                    input_image = gr.Image(label="Source Image (Content)", sources=['upload'], type="pil")
                    color_image = gr.Image(label="Target Image (Colors)", sources=['upload'], type="pil")
                    quantize_btn = gr.Button("Quantize", variant="secondary")
                    
                    # Fixed: These should not allow uploads, they are outputs only
                    q_source = gr.Image(label="Quantized Source", type="pil", interactive=False)
                    q_target = gr.Image(label="Quantized Target", type="pil", interactive=False)
                    palette = gr.Image(label="Palette", type="pil", interactive=False)
                    
                    texture_option = gr.Checkbox(
                        label="Texture Dropout", 
                        info="Check to dropout texture layer"
                    )
                    generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
                
                # Output column
                with gr.Column():
                    result = gr.Gallery(
                        label='Generated Image', show_label=True, 
                        elem_id="gallery", preview=True
                    )
            
            # Event handlers with proper state management
            load_models_btn.click(
                fn=self.load_models, 
                inputs=[model_path], 
                outputs=[load_status]
            )
            
            # Reset state when input image or mode changes
            input_image.input(
                fn=self.process_input_image,
                inputs=[input_image, image_resolution, mode],
                outputs=[input_image]
            )
            
            # Also reset when mode changes without image change
            mode.change(
                fn=self.process_input_image,
                inputs=[input_image, image_resolution, mode],
                outputs=[input_image]
            )
            
            quantize_btn.click(
                fn=self.apply_palette,
                inputs=[input_image, color_image, colors, transfer_method, cmap, blend],
                outputs=[q_source, q_target, palette]
            )
            
            generate_btn.click(
                fn=self.clear_results,
                inputs=[],
                outputs=[result]
            ).then(
                fn=self.generate_image,
                inputs=[palette, colors, texture_option],
                outputs=[result]
            )
        
        return interface
    
    def launch(self) -> None:
        """Launch the application."""
        interface = self.create_interface()
        interface.launch(
            server_name=self.config.server_name,
            server_port=self.config.server_port,
            share=self.config.share,
            debug=self.config.debug
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Colorize Diffusion App")
    parser.add_argument("--server_name", "-addr", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", "-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--not_show_error", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--enable_text_manipulation", "-manipulate", action="store_true")
    parser.add_argument("--model_load_path", required=False, default=None,
                       help="Pretrained model path")
    return parser.parse_args()


def main():
    """Main function to run the application."""
    args = parse_arguments()
    
    # Create configuration
    config = AppConfig(
        server_name=args.server_name,
        server_port=args.server_port,
        share=True,
        debug=args.debug
    )
    
    # Create and launch app
    app = ColorizeDiffusionApp(config)
    app.launch()


if __name__ == "__main__":
    main()
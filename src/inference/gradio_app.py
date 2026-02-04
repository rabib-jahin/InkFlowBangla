"""
Standalone Gradio App for Bengali Handwriting Generation
This version includes all necessary imports and checks.

Usage:
    python gradio_app.py --model_path ./checkpoints/model.pth
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import argparse
import os
import tempfile
import sys

print("=" * 60)
print("🔍 Checking imports...")
print("=" * 60)

# Check if inference module exists
# Check if inference module exists
inference_module = 'src.inference.infer'
print("✓ Using src.inference.infer")

# Store module name for lazy loading
INFERENCE_MODULE_NAME = inference_module

# Lazy import - will be loaded when needed
BengaliInference = None
set_seed = None

def lazy_import_inference():
    """Import inference module only when needed to avoid early import errors"""
    global BengaliInference, set_seed
    
    if BengaliInference is not None:
        return BengaliInference, set_seed
    
    print(f"📦 Loading {INFERENCE_MODULE_NAME}.py...")
    try:
        from src.inference.infer import BengaliInference as BI, set_seed as ss
        
        BengaliInference = BI
        set_seed = ss
        print(f"✓ Successfully imported from {INFERENCE_MODULE_NAME}.py")
        return BengaliInference, set_seed
    except ImportError as e:
        error_msg = f"Failed to import from {INFERENCE_MODULE_NAME}.py: {e}"
        print(f"❌ ERROR: {error_msg}")
        raise ImportError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error during import: {e}"
        print(f"❌ ERROR: {error_msg}")
        raise

print("✓ Import check complete (will load on first use)")
print("=" * 60)


class BengaliHandwritingApp:
    def __init__(self, model_path, style_path, stable_dif_path):
        """Initialize the inference model"""
        self.model_path = model_path
        self.style_path = style_path
        self.stable_dif_path = stable_dif_path
        self.inference = None
        self.set_seed = None  # Will be set during initialization
        
        # Validate paths
        if not os.path.exists(model_path):
            print(f"⚠️ WARNING: Model path does not exist: {model_path}")
        if not os.path.exists(style_path):
            print(f"⚠️ WARNING: Style path does not exist: {style_path}")
        
    def initialize_model(self):
        """Lazy load the model when first needed"""
        if self.inference is None:
            print("🔧 Initializing model...")
            
            # Import inference module here (lazy loading)
            try:
                BengaliInference, set_seed = lazy_import_inference()
            except Exception as e:
                error_msg = f"Failed to load inference module: {e}"
                print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            try:
                args = argparse.Namespace(
                    model_path=self.model_path,
                    style_path=self.style_path,
                    stable_dif_path=self.stable_dif_path,
                    model_name='diffusionpen',
                    img_size=(64, 256),
                    channels=4,
                    emb_dim=320,
                    num_heads=4,
                    num_res_blocks=1,
                    latent=True,
                    mix_rate=None,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    interpolation=False,
                    img_feat=True,
                    color=True,
                    unet='unet_latent'
                )
                self.inference = BengaliInference(args)
                self.set_seed = set_seed  # Store set_seed function
                print("✅ Model initialized!")
            except Exception as e:
                print(f"❌ ERROR: Failed to initialize model")
                print(f"Error details: {e}")
                import traceback
                traceback.print_exc()
                raise
        return self.inference
    
    def generate_handwriting(
        self, 
        text, 
        style_images, 
        mode, 
        num_steps, 
        gap, 
        seed,
        writer_id
    ):
        """
        Generate Bengali handwriting
        
        Args:
            text: Bengali text to generate
            style_images: List of PIL Images or file paths
            mode: 'word' or 'sentence'
            num_steps: Number of inference steps
            gap: Gap between words in pixels
            seed: Random seed (None for random)
            writer_id: Writer ID (usually 0)
        """
        try:
            # Initialize model if not already done
            inference = self.initialize_model()
            
            # Validate inputs
            if not text or text.strip() == "":
                return None, "❌ Please enter some Bengali text"
            
            if style_images is None or len(style_images) == 0:
                return None, "❌ Please upload at least one style image"
            
            # Set seed if provided
            if seed is not None and seed >= 0:
                if self.set_seed:
                    self.set_seed(9)
                else:
                    # Fallback if set_seed not available
                    import random
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
            
            # Process style images
            print(f"📸 Processing {len(style_images)} style images...")
            
            # Save temporary style images and get paths
            temp_paths = []
            for idx, img in enumerate(style_images):
                if isinstance(img, str):
                    # Already a path
                    temp_paths.append(img)
                else:
                    # PIL Image - save temporarily
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False, 
                        suffix='.png', 
                        prefix=f'style_{idx}_'
                    )
                    img.save(temp_file.name)
                    temp_paths.append(temp_file.name)
            
            # Load style images
            style_tensor = inference.load_style_images(temp_paths)
            
            # Generate based on mode
            print(f"🚀 Generating in '{mode}' mode...")
            if mode == 'word':
                output_image = inference.generate(
                    text, 
                    style_tensor, 
                    writer_id=writer_id, 
                    num_inference_steps=num_steps
                )
            else:  # sentence mode
                output_image = inference.generate_sentence(
                    text, 
                    style_tensor, 
                    writer_id=writer_id, 
                    num_inference_steps=num_steps,
                    gap=gap
                )
            
            # Cleanup temp files
            for path in temp_paths:
                if os.path.exists(path) and path.startswith(tempfile.gettempdir()):
                    try:
                        os.remove(path)
                    except:
                        pass
            
            success_msg = f"✅ Successfully generated '{text}' in {mode} mode with {num_steps} steps"
            return output_image, success_msg
            
        except Exception as e:
            error_msg = f"❌ Error during generation: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None, error_msg


def create_gradio_interface(
    model_path="./checkpoints/model.pth",
    style_path="./style_models/mixed_bengali_mobilenetv2_100.pth",
    stable_dif_path="runwayml/stable-diffusion-v1-5"
):
    """Create and return the Gradio interface"""
    
    app = BengaliHandwritingApp(model_path, style_path, stable_dif_path)
    
    # Example texts
    example_texts = [
        "আমার সোনার বাংলা",
        "সে একটি কলম কিনেছে",
        "বাংলা ভাষা",
        "শুভ নববর্ষ"
    ]
    
    # Detect Gradio version
    import gradio
    gradio_version = gradio.__version__
    print(f"📦 Using Gradio version: {gradio_version}")
    
    # Use appropriate file type based on version
    major_version = int(gradio_version.split('.')[0])
    file_type = "filepath" if major_version >= 4 else "file"
    
    with gr.Blocks(title="Bengali Handwriting Generation") as demo:
        gr.Markdown(
            """
            # 🇧🇩 Bengali Handwriting Generation
            
            Generate beautiful Bengali handwriting in any style! Upload reference images of handwriting samples, 
            enter your text, and watch the AI create handwriting that matches your style.
            
            ### 📝 How to use:
            1. **Upload Style Images**: Upload 1-5 images of handwriting samples (the style you want to mimic)
            2. **Enter Text**: Type or paste Bengali text
            3. **Configure Settings**: Adjust generation parameters (optional)
            4. **Generate**: Click the button and wait for your handwriting!
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🎨 Input")
                
                # Text input
                text_input = gr.Textbox(
                    label="Bengali Text",
                    placeholder="Enter Bengali text here... (e.g., আমার সোনার বাংলা)",
                    lines=3,
                    value="আমার সোনার বাংলা"
                )
                
                # Quick examples
                gr.Examples(
                    examples=[[txt] for txt in example_texts],
                    inputs=[text_input],
                    label="Example Texts"
                )
                
                # Style images upload - version compatible
                style_images_input = gr.File(
                    label="Style Reference Images (1-5 images)",
                    file_count="multiple",
                    file_types=["image"],
                    type=file_type
                )
                
                gr.Markdown(
                    """
                    💡 **Tip**: Upload clear images of handwriting samples. 
                    More samples (up to 5) generally produce better results.
                    """
                )
                
                # Settings
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    mode_input = gr.Radio(
                        choices=["word", "sentence"],
                        value="sentence",
                        label="Generation Mode",
                        info="Word mode: generates single word | Sentence mode: generates full sentence with spacing"
                    )
                    
                    num_steps_input = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Inference Steps",
                        info="More steps = better quality but slower (50 is recommended)"
                    )
                    
                    gap_input = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=5,
                        label="Word Gap (pixels)",
                        info="Space between words in sentence mode"
                    )
                    
                    seed_input = gr.Number(
                        label="Random Seed (optional)",
                        value=9,
                        precision=0,
                        info="Use -1 for random, or set a number for reproducible results"
                    )
                    
                    writer_id_input = gr.Number(
                        label="Writer ID",
                        value=0,
                        precision=0,
                        info="Usually keep at 0"
                    )
                
                # Generate button
                generate_btn = gr.Button(
                    "✨ Generate Handwriting",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("## 🖼️ Output")
                
                # Output image
                output_image = gr.Image(
                    label="Generated Handwriting",
                    type="pil"
                )
                
                # Status message
                status_message = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )
        
        # Info section
        gr.Markdown(
            """
            ---
            ### 📖 About
            
            This application uses a diffusion-based model to generate Bengali handwriting. 
            The model learns the style from reference images and applies it to your input text.
            
            **Model Features:**
            - Supports Bengali Unicode characters
            - Style transfer from reference images
            - Word and sentence generation modes
            - Automatic word spacing and layout
            
            ### 🔧 Technical Details
            - Model: DiffusionPen with UNet architecture
            - Style Encoder: MobileNetV2
            - Text Encoder: CANINE (Character-level)
            - VAE: Stable Diffusion VAE for latent space
            """
        )
        
        # Event handlers - version compatible
        def process_and_generate(text, style_files, mode, steps, gap, seed, writer_id):
            if style_files is None or len(style_files) == 0:
                return None, "❌ Please upload at least one style image"
            
            # Handle different Gradio versions
            style_images = []
            for file_obj in style_files:
                try:
                    # Gradio 4.x: direct filepath string
                    # Gradio 3.x: TemporaryFile object with .name attribute
                    if isinstance(file_obj, str):
                        file_path = file_obj
                    elif hasattr(file_obj, 'name'):
                        file_path = file_obj.name
                    else:
                        file_path = str(file_obj)
                    
                    img = Image.open(file_path)
                    style_images.append(img)
                except Exception as e:
                    return None, f"❌ Error loading image: {str(e)}"
            
            # Generate
            result_img, status = app.generate_handwriting(
                text=text,
                style_images=style_images,
                mode=mode,
                num_steps=int(steps),
                gap=int(gap),
                seed=int(seed) if seed >= 0 else None,
                writer_id=int(writer_id)
            )
            
            return result_img, status
        
        generate_btn.click(
            fn=process_and_generate,
            inputs=[
                text_input,
                style_images_input,
                mode_input,
                num_steps_input,
                gap_input,
                seed_input,
                writer_id_input
            ],
            outputs=[output_image, status_message]
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="Bengali Handwriting Generation - Gradio Interface")
    parser.add_argument('--model_path', type=str, default='./checkpoints/model.pth', 
                        help='Path to trained model checkpoint')
    parser.add_argument('--style_path', type=str, default='./style_models/mixed_bengali_mobilenetv2_100.pth',
                        help='Path to style extractor model')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='Path to stable diffusion model')
    parser.add_argument('--share', action='store_true', help='Create public share link')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--server_port', type=int, default=7860, help='Server port')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🇧🇩 Bengali Handwriting Generation - Web Interface")
    print("=" * 60)
    print(f"Model Path: {args.model_path}")
    print(f"Style Path: {args.style_path}")
    print(f"Stable Diffusion: {args.stable_dif_path}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    # Create interface
    demo = create_gradio_interface(
        model_path=args.model_path,
        style_path=args.style_path,
        stable_dif_path=args.stable_dif_path
    )
    
    # Launch
    print("🚀 Launching Gradio interface...")
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True
    )


if __name__ == "__main__":
    main()
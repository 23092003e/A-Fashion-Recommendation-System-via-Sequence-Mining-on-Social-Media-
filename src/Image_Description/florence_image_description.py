"""
Florence-2 Fashion Image Description Generator
===============================================
This script uses Microsoft's Florence-2 model to generate detailed descriptions
for fashion images. It processes all images in a specified folder and saves
the descriptions to a CSV file.

Usage:
    python florence_image_description.py --input_folder "path/to/images" --output_file "descriptions.csv"
"""

import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')


def load_florence_model(model_name="microsoft/Florence-2-base"):
    """
    Load Florence-2 model and processor.

    Args:
        model_name: Model variant to use. Options:
            - "microsoft/Florence-2-base" (faster, less accurate)
            - "microsoft/Florence-2-large" (slower, more accurate)

    Returns:
        model, processor, device
    """
    print(f"Loading Florence-2 model: {model_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    model.eval()
    print("Model loaded successfully!")

    return model, processor, device


def generate_description(image_path, model, processor, device, task="<MORE_DETAILED_CAPTION>"):
    """
    Generate description for a single image using Florence-2.

    Args:
        image_path: Path to the image file
        model: Florence-2 model
        processor: Florence-2 processor
        device: torch device
        task: Task prompt for Florence-2. Options:
            - "<CAPTION>" : Brief caption
            - "<DETAILED_CAPTION>" : Detailed caption
            - "<MORE_DETAILED_CAPTION>" : Very detailed caption (recommended for fashion)
            - "<OD>" : Object detection

    Returns:
        Generated description string
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')

        # Prepare inputs
        inputs = processor(text=task, images=image, return_tensors="pt")

        # Convert to float16 if using CUDA (fix dtype mismatch)
        if device.type == 'cuda':
            inputs = {
                k: v.to(device).half() if v.dtype == torch.float32 else v.to(device)
                for k, v in inputs.items()
            }
        else:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
                do_sample=False
            )

        # Decode
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Parse the output (Florence-2 returns text with task tokens)
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )

        # Extract the caption text
        if task in parsed_answer:
            description = parsed_answer[task]
        else:
            description = str(parsed_answer)

        return description

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return ""


def generate_fashion_description(image_path, model, processor, device):
    """
    Generate fashion-specific description using multiple prompts.
    Combines general caption with more specific fashion details.

    Args:
        image_path: Path to the image file
        model: Florence-2 model
        processor: Florence-2 processor
        device: torch device

    Returns:
        Dictionary with different description types
    """
    descriptions = {}

    # 1. Detailed caption (main description)
    descriptions['detailed_caption'] = generate_description(
        image_path, model, processor, device,
        task="<MORE_DETAILED_CAPTION>"
    )

    # 2. Brief caption
    descriptions['brief_caption'] = generate_description(
        image_path, model, processor, device,
        task="<CAPTION>"
    )

    return descriptions


def process_image_folder(input_folder, output_file, model_name="microsoft/Florence-2-base",
                         detailed_only=True):
    """
    Process all images in a folder and generate descriptions.

    Args:
        input_folder: Path to folder containing images
        output_file: Path to output CSV file
        model_name: Florence-2 model variant
        detailed_only: If True, only generate detailed captions (faster)

    Returns:
        DataFrame with image descriptions
    """
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}

    # Get list of image files
    image_files = []
    for filename in os.listdir(input_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            image_files.append(filename)

    if not image_files:
        print(f"No images found in {input_folder}")
        return None

    print(f"Found {len(image_files)} images to process")

    # Load model
    model, processor, device = load_florence_model(model_name)

    # Process images
    results = []

    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_folder, filename)

        # Extract image ID from filename (remove extension)
        image_id = os.path.splitext(filename)[0]

        if detailed_only:
            # Faster: only detailed caption
            description = generate_description(
                image_path, model, processor, device,
                task="<MORE_DETAILED_CAPTION>"
            )
            results.append({
                'image_id': image_id,
                'filename': filename,
                'description': description
            })
        else:
            # Full: multiple caption types
            descriptions = generate_fashion_description(
                image_path, model, processor, device
            )
            results.append({
                'image_id': image_id,
                'filename': filename,
                'brief_caption': descriptions['brief_caption'],
                'detailed_caption': descriptions['detailed_caption']
            })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nDescriptions saved to: {output_file}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate fashion image descriptions using Florence-2'
    )
    parser.add_argument(
        '--input_folder',
        type=str,
        default='images/original_images',
        help='Path to folder containing images'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='input/image_descriptions.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='microsoft/Florence-2-base',
        choices=['microsoft/Florence-2-base', 'microsoft/Florence-2-large'],
        help='Florence-2 model variant'
    )
    parser.add_argument(
        '--detailed_only',
        action='store_true',
        default=True,
        help='Only generate detailed captions (faster)'
    )

    args = parser.parse_args()

    # Check if input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder not found: {args.input_folder}")
        print("Please create the folder and add images, or specify a different path.")
        return

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process images
    df = process_image_folder(
        input_folder=args.input_folder,
        output_file=args.output_file,
        model_name=args.model,
        detailed_only=args.detailed_only
    )

    if df is not None:
        print(f"\nProcessed {len(df)} images")
        print("\nSample descriptions:")
        print("-" * 50)
        for i, row in df.head(3).iterrows():
            print(f"Image: {row['filename']}")
            if 'description' in df.columns:
                print(f"Description: {row['description'][:200]}...")
            else:
                print(f"Caption: {row['detailed_caption'][:200]}...")
            print("-" * 50)


# ============ NOTEBOOK/INTERACTIVE MODE ============
def run_interactive(input_folder, output_file="input/image_descriptions.csv",
                    model_name="microsoft/Florence-2-base"):
    """
    Run in interactive/notebook mode.

    Example:
        from florence_image_description import run_interactive
        df = run_interactive("images/original_images")
    """
    return process_image_folder(
        input_folder=input_folder,
        output_file=output_file,
        model_name=model_name,
        detailed_only=True
    )


if __name__ == "__main__":
    main()

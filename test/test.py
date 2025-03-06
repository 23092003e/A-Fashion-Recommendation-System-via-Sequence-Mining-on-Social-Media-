import os
import torch
import csv
import re
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import CLIPImageProcessor
import time

# Define paths
input_folder = r"C:\Users\ADMIN\Desktop\ITDSIU21099_HoangVanManh\Fashion-Marketing-Automation-Solutions\images\original_images"
output_folder = "descriptions"
os.makedirs(output_folder, exist_ok=True)

# CSV output file
csv_output_path = os.path.join(output_folder, "clothing_descriptions.csv")

# Load Vicuna model
print("Loading Vicuna-7b model...")
model_name = "lmsys/vicuna-7b-v1.5"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
)
model.to(device)

# Load CLIP image processor for image preprocessing
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_post_id(filename):
    """Extract the post_id from the filename (assuming it's the number in the filename)."""
    match = re.search(r'(\d+)', filename)
    if match:
        return match.group(1)
    return filename  # Return the filename as fallback if no number found

def generate_description(image_path):
    """Generate clothing description from an image using Vicuna."""
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        processed_image = image_processor(images=image, return_tensors="pt").to(device)
        
        # Create prompt for the model
        prompt = {
            "high": """Please provide an extremely detailed description of this image that could be used to recreate it.
                   Include information about:
                   1. Overall composition and layout
                   2. Main subjects and their characteristics
                   3. Colors, lighting, and atmosphere
                   4. Textures and materials
                   5. Background and environment
                   6. Style and artistic elements
                   7. Any unique or distinctive features""",
            "medium": """Please describe this image in detail, including the main elements,
                     composition, colors, and notable features.""",
            "low": "Please describe the main elements of this image concisely."
        }
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate description
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=250,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode the output
        description = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "ASSISTANT:" in description:
            description = description.split("ASSISTANT:")[1].strip()
            
        return description
    
    except Exception as e:
        return f"Error processing image: {str(e)}"

def process_all_images():
    """Process all images in the input folder and save descriptions to CSV."""
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    print(f"Found {len(image_files)} images to process")
    
    # Create CSV file and write header
    with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["post_id", "description"])
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(input_folder, image_file)
            
            # Extract post_id from filename
            post_id = extract_post_id(image_file)
            
            print(f"Processing image {i+1}/{len(image_files)}: {image_file} (post_id: {post_id})")
            
            # Generate description
            description = generate_description(image_path)
            
            # Save individual description to text file (optional)
            txt_output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
            with open(txt_output_path, "w", encoding="utf-8") as f:
                f.write(description)
            
            # Write to CSV
            csv_writer.writerow([post_id, description])
            
            print(f"Description saved for post_id: {post_id}")
            
            # Add a small delay to prevent overloading
            time.sleep(1)
    
    print(f"All descriptions saved to {csv_output_path}")

if __name__ == "__main__":
    print("Starting clothing image description generation...")
    process_all_images()
    print("All images processed successfully!")
import numpy as np
from PIL import Image
import torch
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# Add the sam2 directory to Python path
sam2_path = os.path.join(os.path.dirname(__file__), "sam2")
if sam2_path not in sys.path:
    sys.path.append(sam2_path)

from sam2.sam2_image_predictor import SAM2ImagePredictor

def generate_point_prompts(image_shape, num_points=5):
    """Generate evenly spaced point prompts across the image."""
    h, w = image_shape[:2]
    points = []
    labels = []
    
    # Center point
    points.append([w//2, h//2])
    labels.append(1)
    
    # Quarter points
    points.append([w//4, h//4])
    points.append([3*w//4, h//4])
    points.append([w//4, 3*h//4])
    points.append([3*w//4, 3*h//4])
    labels.extend([1] * 4)
    
    return np.array(points), np.array(labels)

def overlay_masks(image, masks, points, alpha=0.5, point_size=10):
    """Overlay multiple masks on the image with different colors and show points."""
    overlay = np.array(image).copy()
    
    colors = [
        (1, 0, 0, alpha),  # Red
        (0, 1, 0, alpha),  # Green
        (0, 0, 1, alpha),  # Blue
        (1, 1, 0, alpha),  # Yellow
        (1, 0, 1, alpha),  # Magenta
    ]
    
    # Overlay each mask with a different color
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        mask_rgba = np.zeros((*mask.shape, 4))
        mask_rgba[mask > 0] = color
        overlay = overlay * (1 - mask_rgba[..., 3:]) + mask_rgba[..., :3] * 255 * mask_rgba[..., 3:]
    
    # Draw points
    for i, (x, y) in enumerate(points):
        color = colors[i % len(colors)]
        # Draw a filled circle
        y_indices, x_indices = np.ogrid[-point_size:point_size+1, -point_size:point_size+1]
        mask = x_indices*x_indices + y_indices*y_indices <= point_size*point_size
        y_range = slice(max(0, y-point_size), min(overlay.shape[0], y+point_size+1))
        x_range = slice(max(0, x-point_size), min(overlay.shape[1], x+point_size+1))
        mask = mask[max(0, point_size-y):min(2*point_size+1, overlay.shape[0]-y+point_size),
                   max(0, point_size-x):min(2*point_size+1, overlay.shape[1]-x+point_size)]
        overlay[y_range, x_range][mask] = np.array(color[:3]) * 255
    
    return overlay.astype(np.uint8)

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_image> <output_mask>")
        sys.exit(1)
    input_path, output_path = sys.argv[1], sys.argv[2]

    # Load image
    image = Image.open(input_path).convert("RGB")
    image_np = np.array(image)

    # Load SAM2 model
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-base-plus", device="cuda" if torch.cuda.is_available() else "cpu")

    # Set image
    predictor.set_image(image_np)

    # Generate multiple point prompts
    point_coords, point_labels = generate_point_prompts(image_np.shape)
    
    # Store all masks
    all_masks = []
    
    # Generate mask for each point
    for i in range(len(point_coords)):
        # Use single point for each prediction
        single_point_coords = point_coords[i:i+1]
        single_point_labels = point_labels[i:i+1]
        
        # Predict mask
        masks, ious, logits = predictor.predict(
            point_coords=single_point_coords,
            point_labels=single_point_labels,
            multimask_output=False  # Get single best mask
        )
        mask = masks[0] if masks.ndim == 3 else masks
        all_masks.append(mask)

    # Create overlay with points
    overlay = overlay_masks(image, all_masks, point_coords)
    
    # Save result
    result_img = Image.fromarray(overlay)
    result_img.save(output_path)
    print(f"Saved overlay to {output_path}")

if __name__ == "__main__":
    main()

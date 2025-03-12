import os
from PIL import Image
import numpy as np

def split_image_into_grid(input_image_path, output_dir=None, grid_size=(4, 4)):
    """
    Takes an input image and splits it into a grid of equal pieces.
    
    Args:
        input_image_path (str or PIL.Image): Path to the input image or a PIL Image object
        output_dir (str, optional): Directory to save the split images. If None, images won't be saved.
        grid_size (tuple, optional): Grid dimensions as (rows, cols). Default is (4, 4).
        
    Returns:
        list: List of PIL Image objects representing the split pieces
    """
    # Load the image if a path is provided
    if isinstance(input_image_path, str):
        img = Image.open(input_image_path)
    else:
        img = input_image_path
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Calculate the size of each grid piece
    rows, cols = grid_size
    piece_width = img.width // cols
    piece_height = img.height // rows
    
    # Split the image into grid pieces
    pieces = []
    for i in range(rows):
        for j in range(cols):
            left = j * piece_width
            upper = i * piece_height
            right = min(left + piece_width, img.width)  # Ensure we don't go beyond image bounds
            lower = min(upper + piece_height, img.height)  # Ensure we don't go beyond image bounds
            
            piece = img.crop((left, upper, right, lower))
            pieces.append(piece)
            
            # Save the piece if output_dir is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = f"piece_{i}_{j}.png"
                piece.save(os.path.join(output_dir, filename))
    
    return pieces

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Split an image into a grid of equal pieces')
    parser.add_argument('input_image', help='Path to the input image')
    parser.add_argument('--output_dir', help='Directory to save the split images', default='split_output')
    parser.add_argument('--rows', type=int, default=4, help='Number of rows in the grid')
    parser.add_argument('--cols', type=int, default=4, help='Number of columns in the grid')
    
    args = parser.parse_args()
    
    split_image_into_grid(
        args.input_image, 
        args.output_dir, 
        grid_size=(args.rows, args.cols)
    )
    
    print(f"Image split into {args.rows}x{args.cols} grid and saved to {args.output_dir}")

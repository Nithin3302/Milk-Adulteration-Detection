import cv2
import numpy as np
import os
from tqdm import tqdm
import random
import sys
from datetime import datetime
from packaging import version  # Add this import

def check_dependencies():
    """Verify correct versions of dependencies"""
    try:
        import numpy as np
        import cv2
        from tqdm import tqdm
        
        # Check NumPy version
        np_version = np.__version__
        if not (version.parse("1.26.0") <= version.parse(np_version) < version.parse("2.2.0")):
            print(f"Warning: NumPy version {np_version} might cause conflicts with TensorFlow")
            
        return True
    except ImportError as e:
        print(f"Error: Missing required package - {str(e)}")
        return False

def create_pathogen_pattern(width=128, height=128):
    """Generate synthetic pathogen pattern"""
    # Create a black background
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Number of bacterial cells/clusters to draw
    n_bacteria = random.randint(15, 35)  # Increased range for more variety
    
    # Different pathogen patterns with weights
    patterns = {
        'cocci': 0.4,    # Circular bacteria
        'bacilli': 0.3,  # Rod-shaped bacteria
        'cluster': 0.2,  # Bacterial clusters
        'chain': 0.1     # Streptococcus-like chains
    }
    
    for _ in range(n_bacteria):
        x = random.randint(10, width-10)
        y = random.randint(10, height-10)
        
        # Choose pattern type based on weights
        pattern_type = random.choices(
            list(patterns.keys()), 
            weights=list(patterns.values())
        )[0]
        
        # Base color with slight variations
        base_color = random.randint(150, 200)
        color = tuple(base_color + random.randint(-20, 20) for _ in range(3))
        
        if pattern_type == 'cocci':
            radius = random.randint(2, 5)
            cv2.circle(image, (x, y), radius, color, -1)
            
        elif pattern_type == 'bacilli':
            length = random.randint(8, 15)
            width_b = random.randint(2, 4)
            angle = random.randint(0, 180)
            rect = ((x, y), (length, width_b), angle)
            box = cv2.boxPoints(rect)
            # Changed from np.int0 to np.intp
            box = np.array(box, dtype=np.intp)
            cv2.drawContours(image, [box], 0, color, -1)
            
        elif pattern_type == 'chain':
            # Create chain of circles (streptococcus-like)
            n_cells = random.randint(3, 6)
            radius = random.randint(2, 4)
            angle = random.uniform(0, 2*np.pi)
            for i in range(n_cells):
                cx = int(x + i*radius*2*np.cos(angle))
                cy = int(y + i*radius*2*np.sin(angle))
                if 0 <= cx < width and 0 <= cy < height:
                    cv2.circle(image, (cx, cy), radius, color, -1)
            
        else:  # cluster
            points = np.random.randint(0, 15, (random.randint(3, 8), 2))
            points = points + [x, y]
            # Convert points to integer type
            points = np.array(points, dtype=np.intp)
            cv2.fillPoly(image, [points], color)
    
    # Add microscope-like effects
    noise = np.random.normal(0, 2, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    return image

def main():
    """Main execution function"""
    if not check_dependencies():
        sys.exit(1)
        
    try:
        # Paths
        src_path = os.path.join("data", "raw", "real", "pure")
        dst_path = os.path.join("data", "raw", "real", "pathogens")
        
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source directory not found: {src_path}")
            
        os.makedirs(dst_path, exist_ok=True)
        
        # Get pure milk images
        pure_images = [f for f in os.listdir(src_path) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not pure_images:
            raise ValueError("No images found in source directory")
        
        print(f"Found {len(pure_images)} pure milk images")
        print(f"Generating pathogen images...")
        
        # Process images with progress bar
        with tqdm(total=len(pure_images), desc="Generating", unit="img") as pbar:
            for i, img_name in enumerate(pure_images):
                pure_img = cv2.imread(os.path.join(src_path, img_name))
                if pure_img is None:
                    continue
                    
                height, width = pure_img.shape[:2]
                pathogen_img = create_pathogen_pattern(width, height)
                
                # Blend with very faint background
                faint_pure = cv2.addWeighted(pure_img, 0.1, np.zeros_like(pure_img), 0.9, 0)
                final_img = cv2.addWeighted(pathogen_img, 0.7, faint_pure, 0.3, 0)
                
                # Save with consistent naming
                new_name = f"pathogen_{i+1:04d}.jpg"
                cv2.imwrite(os.path.join(dst_path, new_name), final_img)
                pbar.update(1)
        
        print(f"\nSuccess! Generated pathogen images saved in {dst_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
import re
import math
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

def process_png_content(image_content):
    """
    Process raw PNG binary content to extract character point clouds.
    
    Args:
        image_content (bytes): Raw bytes of the PNG file.
        
    Returns:
        list[dict]: A list of dicts, each containing 'points' (list of [x, y]).
                    Each dict represents a detected character.
                    
    Methodology:
        1. Decode byte stream to image using OpenCV.
        2. Invert colors (samples are dark text on light background; we need white text on black for contours).
        3. Apply morphological opening to remove small noise (thin lines).
        4. Find contours of connected components.
        5. Filter out tiny contours (noise).
        6. Sort contours from left to right (by X coordinate).
        7. Convert contour points to standard list format.
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for PNG support")
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return []
        
    # Standardize size (Upscale if too small)
    # This is critical for reliable erosion/morphology on small images
    h, w = img.shape
    if h < 50:
        scale = 3.0
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
    # Preprocessing to clean noise
    # 1. Gaussian Blur to reduce high frequency noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 2. Adaptive Thresholding or Otsu
    # Invert typical captcha (dark text on light background) -> text becomes white
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2.5 Remove horizontal lines (Strikethrough)
    # Use a vertical kernel (1, 2) to preserve vertical structure but kill thin horizontal lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # 3. Morphological operations
    # If characters are touching, we NEED to erode to separate them
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    paths = []
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    c_list = list(zip(contours, bounding_boxes))
    # Sort by X to maintain order
    c_list.sort(key=lambda x: x[1][0])
    
    final_paths = []
    
    for cnt, bbox in c_list:
        x, y, w, h = bbox
        area = cv2.contourArea(cnt)
        
        # Filter tiny noise
        if area < 150 or h < 20:
            continue
            
        # Heuristic: Check aspect ratio (Width / Height)
        # Typical char is taller than wide (ratio < 1.0) or square.
        # If ratio > 1.2, it might be 2 chars fused.
        # If ratio > 2.5, it might be 3 chars.
        
        aspect_ratio = w / float(h)
        
        # Simple splitting logic
        if aspect_ratio > 1.1:
            # Estimate how many chars
            # Assume average char aspect ratio is ~0.75 - 0.9
            # Dynamic split:
            num_chars = int(round(aspect_ratio / 0.80))
            if num_chars < 2: num_chars = 2
            
            step = w // num_chars
            
            # Divide the contour into chunks
            for i in range(num_chars):
                sub_points = []
                start_x = x + i * step
                end_x = x + (i + 1) * step
                
                # Filter points in this x-range
                # Reshape to list of [x, y]
                pts = cnt.reshape(-1, 2)
                for px, py in pts:
                    if start_x <= px < end_x:
                        sub_points.append([px, py])
                
                if len(sub_points) > 10: # Increased from 5 because upscaled
                    final_paths.append({'points': sub_points})
        else:
            points = cnt.reshape(-1, 2).tolist()
            final_paths.append({'points': points})
            
    return final_paths

def parse_svg_paths(svg_content):
    """
    Parses SVG content string to extract character path data.
    
    Args:
        svg_content (str): The raw SVG XML string.
        
    Returns:
        list[dict]: A list of dicts [{'d': str, 'points': list, 'x': float}].
                    Sorted by the 'x' coordinate (left to right).
                    
    Methodology:
        1. Regex match all <path> tags.
        2. Filter out paths with 'fill="none"' or 'stroke' which usually represent noise lines.
        3. Extract the 'd' attribute (path data).
        4. Parse numbers from 'd' to form point lists.
        5. Calculate min-x for each path to sort them.
    """
    # Simple regex to extract <path> tags
    paths = []
    path_tags = re.findall(r'<path[^>]+>', svg_content)
    
    for tag in path_tags:
        # Heuristic: skip noise paths
        if 'fill="none"' in tag or 'stroke' in tag:
            continue
            
        d_match = re.search(r'd="([^"]+)"', tag)
        if d_match:
            d = d_match.group(1)
            # Get points to find x-coordinate for sorting
            pts = extract_points_from_d(d)
            if pts:
                min_x = min(p[0] for p in pts)
                paths.append({'d': d, 'points': pts, 'x': min_x})
    
    # Sort characters from left to right
    paths.sort(key=lambda item: item['x'])
    return paths

def extract_points_from_d(d):
    """
    Parses the SVG path 'd' attribute string into a list of points.
    
    Args:
        d (str): SVG path data string.
        
    Returns:
        list[tuple]: List of (x, y) tuples.
        
    Note:
        This uses a simplified regex approach that treats all sequences of numbers
        as coordinate pairs. It effectively flattens complex SVG commands (M, L, C, Q)
        into a point cloud, which is sufficient for shape matching.
    """
    # Extract all numbers
    nums = [float(x) for x in re.findall(r'-?\d*\.?\d+', d)]
    points = []
    # Pair them up
    for i in range(0, len(nums)-1, 2):
        points.append((nums[i], nums[i+1]))
    return points

def normalize_points(points, size=100):
    """
    Normalizes a point cloud to fit within a fixed size box (default 100x100).
    
    Args:
        points (list): List of (x, y) coordinates.
        size (int): Target dimension size.
        
    Returns:
        list[tuple]: Normalized point list.
        
    Methodology:
        1. Find bounding box (min_x, max_x, min_y, max_y).
        2. Calculate scale factor to fit the largest dimension into `size`.
        3. Center the cloud within the target box.
        
    Purpose:
        Makes the recognition scale-invariant and position-invariant.
    """
    if not points:
        return []

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if w == 0 or h == 0:
        return []
    
    scale = size / max(w, h)
    
    # Center in the box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    new_points = []
    for x, y in points:
        nx = (x - center_x) * scale + (size / 2)
        ny = (y - center_y) * scale + (size / 2)
        new_points.append((nx, ny))
        
    return new_points

def render_ascii_art(points, width=40, height=20):
    """
    Visualizes a point cloud as ASCII art. Useful for debugging specific characters.
    
    Args:
        points (list): List of (x, y) coordinates.
        width (int): Output width in characters.
        height (int): Output height in characters.
        
    Returns:
        str: Multiline string representation.
    """
    if not points:
        return ""
        
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if w == 0 or h == 0:
        return ""
    
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Mapping
    for x, y in points:
        # map x from min_x..max_x to 0..width-1
        # map y from min_y..max_y to 0..height-1
        ix = int((x - min_x) / w * (width - 1))
        iy = int((y - min_y) / h * (height - 1))
        
        if 0 <= ix < width and 0 <= iy < height:
            grid[iy][ix] = '#'
            
    return "\n".join("".join(row) for row in grid)

def downsample_points(points, max_points=100):
    """
    Downsamples a point cloud to a maximum point count.
    
    Args:
        points (list): Input point list.
        max_points (int): Maximum number of points allowed.
        
    Returns:
        list: Reduced list of points.
        
    Purpose:
        Reduces computational load for distance calculations while preserving shape.
        It uses simple uniform sampling/decimation.
    """
    if len(points) <= max_points:
        return points
    step = len(points) / max_points
    return [points[int(i * step)] for i in range(max_points)]

import json
import math
import os
from .utils import parse_svg_paths, normalize_points, downsample_points

class CaptchaSolver:
    """
    Core class responsible for solving CAPTCHAs using a knowledge base of labeled patterns.
    
    Attributes:
        db_path (str): Path to the JSON file storing the training data.
        knowledge_base (list): In-memory list of labeled patterns loaded from disk.
                               Each item is {'label': str, 'points': list[tuple]}.
    """
    def __init__(self, db_path='database.json'):
        self.db_path = db_path
        self.knowledge_base = []
        self.load_db()

    def load_db(self):
        """
        Loads the knowledge base from the JSON file.
        Initializes an empty list if file doesn't exist or is invalid.
        """
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    self.knowledge_base = json.load(f)
            except:
                self.knowledge_base = []
        else:
            self.knowledge_base = []

    def save_db(self):
        """
        Persists the current knowledge base to the JSON file.
        """
        with open(self.db_path, 'w') as f:
            json.dump(self.knowledge_base, f)

    def add_training_data(self, points, label):
        """
        Adds a new labeled sample to the knowledge base and saves it.
        
        Args:
            points (list): Normalized list of (x, y) coordinates.
            label (str): Correct character label (e.g., "A", "7").
        """
        self.knowledge_base.append({
            'label': label,
            'points': points
        })
        self.save_db()

    def calculate_distance(self, pts1, pts2):
        """
        Calculates a distance metric (Chamfer distance) between two point clouds.
        
        Args:
            pts1 (list): First point cloud.
            pts2 (list): Second point cloud.
            
        Returns:
            float: The average minimum squared distance from curve 1 to curve 2.
            
        Note:
            This function computes the 'one-way' Chamfer distance from pts1 to pts2.
            Meaning for every point in pts1, it finds the nearest point in pts2 and sums the squared distances.
            For true similarity, we typically average distance(p1, p2) and distance(p2, p1).
        """
        # Optimization: Downsample if too many points to avoid O(N^2) hitting performance
        p1 = pts1 if len(pts1) < 100 else downsample_points(pts1)
        p2 = pts2 if len(pts2) < 100 else downsample_points(pts2)
        
        error = 0
        for x1, y1 in p1:
            min_dist = float('inf')
            # Brute force search for nearest neighbor in p2
            for x2, y2 in p2:
                dist = (x1 - x2)**2 + (y1 - y2)**2
                if dist < min_dist:
                    min_dist = dist
            error += min_dist
            
        return error / len(p1)
    
    def solve(self, input_data):
        """
        Solves a CAPTCHA by matching its distinct characters against the knowledge base.
        
        Args:
            input_data (Union[str, list]): 
                - str: Raw SVG content (will be parsed into paths).
                - list: List of dicts {'points': [...]} (already extracted contours, e.g., from PNG).
        
        Returns:
            str: The recognized text string (e.g., "AB12CD").
            
        Algorithm:
            1. Parse input into individual character contours (paths).
            2. For each character path:
               a. Normalize points (scale and center).
               b. Compare against EVERY pattern in the knowledge base using bidirectional Chamfer distance.
               c. Select the label with the lowest distance score (nearest neighbor).
            3. Concatenate labels to form the result.
        """
        from .utils import parse_svg_paths, normalize_points
        
        # Handle polymorphic input (SVG string vs PNG contour list)
        if isinstance(input_data, list):
            paths = input_data
        else:
            paths = parse_svg_paths(input_data)
            
        final_text = ""
        for p in paths:
            pts = p['points']
            if not pts: 
                continue
            
            # Normalize to canonical size for matching
            norm_pts = normalize_points(pts)
            
            best_label = "?"
            best_score = float('inf')
            
            # 1-NN Classification search
            for entry in self.knowledge_base:
                db_pts = entry['points']
                # Calculate symmetric distance
                d1 = self.calculate_distance(norm_pts, db_pts)
                d2 = self.calculate_distance(db_pts, norm_pts)
                score = (d1 + d2) / 2
                
                if score < best_score:
                    best_score = score
                    best_label = entry['label']
            
            # Optional thresholding could be added here to reject weak matches
            if best_score < 1000: 
                final_text += best_label
            else:
                final_text += "?"
                
        return final_text

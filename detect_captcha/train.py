import os
import sys
from detect_captcha.utils import parse_svg_paths, normalize_points
from detect_captcha.solver import CaptchaSolver
import functools

# Auto-flush print for real-time feedback in some environments
print = functools.partial(print, flush=True)

def rebuild_database(labeled_dir='labeled_data'):
    """
    Rebuilds the training database (database.json) from scratch using labeled SVG files.
    
    Args:
        labeled_dir (str): Path to the directory containing labeled SVG files.
                           Files must be named as per their label (e.g., "AB12CD.svg").
    
    Process:
        1. Clears existing database.
        2. Iterates through all .svg files in `labeled_dir`.
        3. Parses each file to extract character paths.
        4. Validates that the number of extracted paths matches the label length.
           (Mismatch usually indicates parsing noise or connected characters).
        5. If valid, normalizes each character and saves it to the solver's knowledge base.
    
    Outcome:
        A populated `database.json` file used by the Solver for identification.
    """
    solver = CaptchaSolver()
    # RESET database
    solver.knowledge_base = []
    print("Database cleared. Starting rebuild from", labeled_dir)
    
    if not os.path.exists(labeled_dir):
        print(f"Error: Directory '{labeled_dir}' not found.")
        return
    
    files = [f for f in os.listdir(labeled_dir) if f.endswith('.svg')]
    files.sort()
    
    success_count = 0
    fail_count = 0
    total_chars = 0
    
    for filename in files:
        filepath = os.path.join(labeled_dir, filename)
        label_text = os.path.splitext(filename)[0].upper() # e.g. "A7VXAT"
        
        with open(filepath, 'r') as f:
            content = f.read()
            
        paths = parse_svg_paths(content)
        
        # Validation: Path count usually matches label length.
        # However, sometimes 'i' might be split or noise might appear.
        # We'll trust the parser's filtering logic (it ignores stroke/no-fill).
        
        if len(paths) != len(label_text):
            print(f"WARNING: {filename} - Found {len(paths)} paths vs {len(label_text)} chars. Skipping.")
            fail_count += 1
            # Optional: visualize to see why? or just skip. User provided labeled data, assume it's clean or we skip bad ones.
            continue
            
        # Add to DB
        for i, p in enumerate(paths):
            norm = normalize_points(p['points'])
            char_label = label_text[i]
            solver.add_training_data(norm, char_label)
            total_chars += 1
            
        success_count += 1
        
    print(f"\nRebuild Complete.")
    print(f"Processed {success_count} files successfully.")
    print(f"Skipped {fail_count} files due to mismatch.")
    print(f"Total patterns in database: {len(solver.knowledge_base)}")

if __name__ == "__main__":
    rebuild_database()

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import csv
import shutil
import os
from pathlib import Path
import sys
from typing import List, Optional, Any

DATASET_DIR = Path("dataset_osm")
CSV_PATH = DATASET_DIR / "labels.csv"
TRACK_DIR = DATASET_DIR / "track"
NOT_TRACK_DIR = DATASET_DIR / "not_track"
DISPLAY_SIZE = (600, 600)  # Size to display image in window

class ChipReviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Positive Chip Reviewer")
        
        # State
        self.rows: List[Optional[List[str]]] = []
        self.headers = []
        self.current_queue_index = 0
        self.review_queue = [] # List of indices in self.rows to review
        
        # Layout
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.lbl_image = tk.Label(self.frame, text="Loading...")
        self.lbl_image.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.lbl_info = tk.Label(root, text="", font=("Arial", 14, "bold"))
        self.lbl_info.pack(pady=5)

        self.lbl_help = tk.Label(root, text="⬅️ LEFT: Keep (Yes)   |   ➡️ RIGHT: Move to Negatives (No)   |   ⬇️ DOWN: Delete", font=("Arial", 12))
        self.lbl_help.pack(pady=10)

        # Bindings
        root.bind('<Left>', self.action_keep)
        root.bind('<Right>', self.action_move)
        root.bind('<Down>', self.action_delete)
        root.bind('<Escape>', self.close_app)
        
        # Load Data
        self.load_data()
        self.show_current()

    def load_data(self):
        if not CSV_PATH.exists():
            messagebox.showerror("Error", f"Could not find {CSV_PATH}")
            self.root.destroy()
            return

        with open(CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            self.headers = next(reader)
            self.rows = list(reader)

        print(f"Loaded {len(self.rows)} total rows.")
        
        # Identify rows to review: Label='1' (track) and file actually exists
        for i, row in enumerate(self.rows):
            # Row structure: [filepath, label, lat, lon, osmid, tag, sport, kind]
            # label is index 1
            if row is None:
                continue
            if row[1] == '1':
                full_path = DATASET_DIR / row[0]
                if full_path.exists():
                    self.review_queue.append(i)
        
        print(f"Found {len(self.review_queue)} positive examples to review.")
        if not self.review_queue:
            messagebox.showinfo("Done", "No positive samples found to review!")
            self.root.destroy()

    def show_current(self):
        if self.current_queue_index >= len(self.review_queue):
            messagebox.showinfo("Done", "Review complete!")
            self.close_app()
            return
            
        row_idx = self.review_queue[self.current_queue_index]
        row = self.rows[row_idx]
        filepath = DATASET_DIR / row[0]
        
        try:
            # Load and Resize Image
            pil_img = Image.open(filepath)
            
            # Aspect ratio resize
            pil_img.thumbnail(DISPLAY_SIZE, Image.Resampling.LANCZOS)
            
            self.tk_img = ImageTk.PhotoImage(pil_img) # Keep ref to prevent GC
            self.lbl_image.config(image=self.tk_img, text="")
            
            # Update Info Label
            # filename / current index
            self.lbl_info.config(text=f"Image {self.current_queue_index + 1} / {len(self.review_queue)}\n{filepath.name}")
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            self.current_queue_index += 1
            self.show_current()

    def action_keep(self, event):
        """Keep as positive (track)."""
        # Just advance
        self.current_queue_index += 1
        self.show_current()

    def action_move(self, event):
        """Move to negatives (not_track)."""
        row_idx = self.review_queue[self.current_queue_index]
        row = self.rows[row_idx]
        
        old_path = DATASET_DIR / row[0]
        filename = old_path.name
        
        # Ensure not_track directory exists
        NOT_TRACK_DIR.mkdir(exist_ok=True)
        new_path = NOT_TRACK_DIR / filename
        
        try:
            shutil.move(old_path, new_path)
            
            # Update Data Frame
            label_idx = self.headers.index("label") if "label" in self.headers else 1
            path_idx = self.headers.index("filepath") if "filepath" in self.headers else 0
            kind_idx = self.headers.index("kind") if "kind" in self.headers else 7
            
            self.rows[row_idx][label_idx] = '0'
            self.rows[row_idx][path_idx] = str(new_path.relative_to(DATASET_DIR))
            # Optional: update 'kind' to indicate it was manually moved? 
            # Keeping it simple for now, maybe mark as hard_negative
            if len(self.rows[row_idx]) > kind_idx:
                self.rows[row_idx][kind_idx] = "manual_negative"
                
            print(f"Moved {filename} to not_track")
        except Exception as e:
            print(f"Failed to move file: {e}")
            
        self.current_queue_index += 1
        self.show_current()

    def action_delete(self, event):
        """Delete file entirely."""
        row_idx = self.review_queue[self.current_queue_index]
        row = self.rows[row_idx]
        path = DATASET_DIR / row[0]
        
        try:
            if path.exists():
                os.remove(path)
            
            # Mark row as None (deleted) to skip writing later
            self.rows[row_idx] = None
            print(f"Deleted {path.name}")
        except Exception as e:
            print(f"Failed to delete file: {e}")
            
        self.current_queue_index += 1
        self.show_current()

    def close_app(self, event=None):
        print("Saving updated labels.csv...")
        try:
            with open(CSV_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                for row in self.rows:
                    if row is not None: # Skip deleted rows
                        writer.writerow(row)
            print("Successfully saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV: {e}")
        
        # Ask to run the split generation before closing
        #if messagebox.askyesno("Generate Splits", "Do you want to create validation/test splits now?"):
        #    self.create_validation_test_split()
        
        self.root.destroy()
        


if __name__ == "__main__":
    if not DATASET_DIR.exists():
        print(f"Dataset directory {DATASET_DIR} not found.")
        sys.exit(1)
        
    root = tk.Tk()
    app = ChipReviewApp(root)
    root.mainloop()



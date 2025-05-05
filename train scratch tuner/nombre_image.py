import os

import matplotlib.pyplot as plt
import numpy as np


def plot_images_per_class(train_dir):
    """
    Create a bar graph showing the number of images per class in the training dataset.
    
    Args:
        train_dir (str): Path to the training directory containing class subdirectories
    """
    # Get list of classes (subdirectories)
    classes = sorted(entry.name for entry in os.scandir(train_dir) if entry.is_dir())
    
    # Count files per class
    counts = []
    for cls in classes:
        cls_folder = os.path.join(train_dir, cls)
        # Count only audio files
        files = [f for f in os.listdir(cls_folder) if f.lower().endswith((".wav", ".ogg"))]
        counts.append(len(files))
    
    # Create the bar chart
    plt.figure(figsize=(16, 8))
    bars = plt.bar(classes, counts, color='royalblue')
    
    # Add count numbers on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{counts[i]}',
                 ha='center', va='bottom', fontsize=8)
    
    # Add labels and title
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Nombre d\'images', fontsize=14)
    plt.title('Nombre d\'images par classe dans le jeu de données d\'entraînement', fontsize=16)
    
    # Rotate class names for better readability
    plt.xticks(rotation=90, fontsize=10)
    
    # Add grid to help with readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ensure layout fits well
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('images_per_class.png', dpi=300)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Path to the training directory
    train_dir = "./bird_audio_dataset/train"
    
    # Generate and display the plot
    plot_images_per_class(train_dir)
    
    print("Graph saved as 'images_per_class.png'")
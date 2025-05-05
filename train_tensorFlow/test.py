import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from collections import Counter
 
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
 
# Define constants for audio processing (matching your training parameters)
SAMPLE_RATE = 32000
DURATION = 10  # Modifié pour correspondre au modèle (était 5 secondes)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
TARGET_SAMPLES = SAMPLE_RATE * DURATION
 
# Path to the model (use one of your trained models)
MODEL_PATH = 'IA_BIRD/models_result/8/improved_model_dataset.h5'
 
# Classes from dataset-2
BIRD_CLASSES = ["barswa", "comsan", "eaywag1", "thrnig1", "wlwwar", "woosan"]
 
def load_model_and_classes():
    """Load the model and return with predefined classes"""
    try:
        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH)
       
        # Return model and hardcoded classes
        return model, BIRD_CLASSES
           
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, []
 
def process_audio(file_path):
    """Load and process an audio file into a mel-spectrogram"""
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
       
        # Standardize duration
        if len(audio) > TARGET_SAMPLES:
            # Take a segment from the middle
            start = (len(audio) - TARGET_SAMPLES) // 2
            audio = audio[start:start + TARGET_SAMPLES]
        else:
            # If too short, pad with zeros
            padding = TARGET_SAMPLES - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
       
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
       
        # Convert to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0, top_db=80.0)
       
        # Normaliser entre -1 et 1
        mel_spec_norm = 2 * ((mel_spec_db - mel_spec_db.min()) /
                            (mel_spec_db.max() - mel_spec_db.min())) - 1
       
        # Add channel dimension for the model
        mel_spec_input = mel_spec_norm[..., np.newaxis]
       
        # Add batch dimension
        mel_spec_input = np.expand_dims(mel_spec_input, axis=0)
       
        return audio, sr, mel_spec_input, mel_spec_db
   
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None, None, None
 
def predict_bird(model, mel_spec_input, class_names):
    """Make a prediction using the model"""
    if mel_spec_input is None or model is None:
        return "Error processing audio", 0, {}
   
    try:
        # Make prediction
        predictions = model.predict(mel_spec_input)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
       
        # Get class name
        if predicted_class_idx < len(class_names):
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = f"Unknown (Class {predicted_class_idx})"
       
        # Show all classes probabilities as a dictionary
        all_predictions = {class_names[i]: float(predictions[0][i])
                          for i in range(min(len(class_names), len(predictions[0])))}
       
        return predicted_class, confidence, all_predictions
   
    except Exception as e:
        print(f"Error making prediction: {e}")
        return "Error during prediction", 0, {}

def process_folder(folder_path, model, class_names):
    """Process all audio files in a folder and return prediction results"""
    results = []
    
    # Get all audio files in the folder
    audio_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.wav', '.mp3', '.ogg'))]
    
    if not audio_files:
        return []
    
    # Process each file
    for audio_file in audio_files:
        file_path = os.path.join(folder_path, audio_file)
        
        # Process audio
        audio, sr, mel_spec_input, mel_spec_db = process_audio(file_path)
        
        if audio is not None:
            # Make prediction
            predicted_class, confidence, all_predictions = predict_bird(
                model, mel_spec_input, class_names)
            
            # Add to results
            results.append({
                'file': audio_file,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': all_predictions,
                'audio': audio,
                'sr': sr,
                'mel_spec_db': mel_spec_db
            })
    
    return results
 
class BirdSoundClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Classification de Sons d\'Oiseaux')
        self.root.geometry('1000x800')  # Slightly larger to accommodate additional controls
       
        # Load model
        self.model, self.class_names = load_model_and_classes()
        if self.model is None:
            messagebox.showerror("Error", f"Could not load model from {MODEL_PATH}")
            root.destroy()
            return
       
        # Create frames
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(pady=10)
       
        self.folder_frame = tk.Frame(root)
        self.folder_frame.pack(pady=5, fill=tk.X)
        
        self.results_frame = tk.Frame(root)
        self.results_frame.pack(pady=5, fill=tk.X)
        
        self.middle_frame = tk.Frame(root)
        self.middle_frame.pack(pady=10, fill=tk.BOTH, expand=True)
       
        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.pack(pady=10)
        
        self.stats_frame = tk.Frame(root)
        self.stats_frame.pack(pady=5, fill=tk.X)
       
        # Create widgets
        self.create_widgets()
       
        # Current audio file
        self.current_audio_path = None
        self.audio_data = None
        self.sr = None
        
        # Folder results
        self.folder_results = []
        self.current_result_index = 0
   
    def create_widgets(self):
        # Top frame - Upload buttons and info label
        self.info_label = tk.Label(self.top_frame,
                                  text='Chargez un fichier audio ou un dossier d\'oiseaux à classifier')
        self.info_label.pack(pady=10)
       
        button_frame = tk.Frame(self.top_frame)
        button_frame.pack(pady=5)
        
        self.upload_button = tk.Button(button_frame, text='Charger un fichier',
                                      command=self.load_audio_file)
        self.upload_button.pack(side=tk.LEFT, padx=10)
        
        self.folder_button = tk.Button(button_frame, text='Charger un dossier',
                                      command=self.load_folder)
        self.folder_button.pack(side=tk.LEFT, padx=10)
       
        self.file_label = tk.Label(self.top_frame, text='')
        self.file_label.pack(pady=5)
        
        # Folder results frame (initially hidden)
        self.folder_results_label = tk.Label(self.folder_frame, text='Résultats du dossier:')
        self.folder_results_label.pack(pady=5, anchor='w')
        
        navigation_frame = tk.Frame(self.folder_frame)
        navigation_frame.pack(fill=tk.X, pady=5)
        
        self.prev_button = tk.Button(navigation_frame, text='< Précédent', 
                                    command=self.show_previous_result, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=10)
        
        self.results_indicator = tk.Label(navigation_frame, text='')
        self.results_indicator.pack(side=tk.LEFT, expand=True)
        
        self.next_button = tk.Button(navigation_frame, text='Suivant >', 
                                    command=self.show_next_result, state=tk.DISABLED)
        self.next_button.pack(side=tk.RIGHT, padx=10)
        
        # Initially hide folder frame
        self.folder_frame.pack_forget()
        
        # Results frame for listing multiple results
        self.results_tree = ttk.Treeview(self.results_frame, columns=('file', 'bird', 'confidence'),
                                       show='headings', selectmode='browse')
        self.results_tree.heading('file', text='Fichier')
        self.results_tree.heading('bird', text='Oiseau')
        self.results_tree.heading('confidence', text='Confiance')
        
        self.results_tree.column('file', width=250)
        self.results_tree.column('bird', width=150)
        self.results_tree.column('confidence', width=100)
        
        self.results_tree.pack(fill=tk.X, padx=10)
        self.results_tree.bind('<<TreeviewSelect>>', self.on_result_select)
        
        # Initially hide results frame
        self.results_frame.pack_forget()
       
        # Middle frame - Visualization
        self.fig = Figure(figsize=(8, 10))  # Increased height for the third subplot
       
        # Create canvas for waveform, spectrogram and classification results
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.middle_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
       
        # Bottom frame - Results summary
        self.result_label = tk.Label(self.bottom_frame,
                                    text='', font=('Arial', 12))
        self.result_label.pack(pady=5)
        
        # Stats frame for displaying class counts
        self.stats_label = tk.Label(self.stats_frame, text="Statistiques par classe:", font=('Arial', 10, 'bold'))
        self.stats_label.pack(anchor='w', padx=10, pady=5)
        
        # Create a frame for the stats with fixed height
        self.stats_content = tk.Frame(self.stats_frame, height=100)
        self.stats_content.pack(fill=tk.X, padx=10)
        
        # Create a canvas with scrollbar for the stats (in case there are many classes)
        self.stats_canvas = tk.Canvas(self.stats_content, height=100)
        self.stats_scrollbar = ttk.Scrollbar(self.stats_content, orient="vertical", command=self.stats_canvas.yview)
        self.stats_scrollable_frame = ttk.Frame(self.stats_canvas)
        
        self.stats_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.stats_canvas.configure(
                scrollregion=self.stats_canvas.bbox("all")
            )
        )
        
        self.stats_canvas.create_window((0, 0), window=self.stats_scrollable_frame, anchor="nw")
        self.stats_canvas.configure(yscrollcommand=self.stats_scrollbar.set)
        
        self.stats_canvas.pack(side="left", fill="both", expand=True)
        self.stats_scrollbar.pack(side="right", fill="y")
        
        # Initially hide stats frame
        self.stats_frame.pack_forget()
   
    def load_audio_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[('Audio Files', '.wav .mp3 .ogg')])
       
        if not file_path:
            return
        
        # Hide folder-related frames
        self.folder_frame.pack_forget()
        self.results_frame.pack_forget()
       
        self.current_audio_path = file_path
        self.file_label.config(text=os.path.basename(file_path))
       
        # Process the audio
        self.audio_data, self.sr, mel_spec_input, mel_spec_db = process_audio(file_path)
       
        if self.audio_data is not None:
            # Make prediction
            predicted_class, confidence, all_predictions = predict_bird(
                self.model, mel_spec_input, self.class_names)
           
            # Display visualizations with classification results
            self.display_visualizations(mel_spec_db, predicted_class, all_predictions)
           
            # Update result label
            self.result_label.config(text=f'Oiseau détecté : {predicted_class} (Confiance : {confidence:.2f})')
        else:
            messagebox.showerror("Error", "Could not process the audio file.")
    
    def load_folder(self):
        folder_path = filedialog.askdirectory(title="Sélectionner un dossier contenant des fichiers audio")
        
        if not folder_path:
            return
        
        self.file_label.config(text=f"Dossier: {os.path.basename(folder_path)}")
        
        # Process all files in the folder
        self.folder_results = process_folder(folder_path, self.model, self.class_names)
        
        if not self.folder_results:
            messagebox.showinfo("Info", "Aucun fichier audio trouvé dans ce dossier.")
            return
        
        # Display folder results UI
        self.folder_frame.pack(after=self.top_frame, fill=tk.X)
        self.results_frame.pack(after=self.folder_frame, fill=tk.X)
        self.stats_frame.pack(after=self.results_frame, fill=tk.X)
        
        # Populate results tree
        self.results_tree.delete(*self.results_tree.get_children())
        for i, result in enumerate(self.folder_results):
            self.results_tree.insert('', 'end', values=(
                result['file'], 
                result['predicted_class'], 
                f"{result['confidence']:.2f}"
            ))
        
        # Update statistics
        self.update_class_statistics()
        
        # Show first result
        self.current_result_index = 0
        self.update_result_navigation()
        self.show_current_result()
    
    def on_result_select(self, event):
        selection = self.results_tree.selection()
        if selection:
            item_id = selection[0]
            item_index = self.results_tree.index(item_id)
            self.current_result_index = item_index
            self.show_current_result()
            self.update_result_navigation()
    
    def show_previous_result(self):
        if self.current_result_index > 0:
            self.current_result_index -= 1
            self.show_current_result()
            self.update_result_navigation()
            # Also update tree selection
            self.results_tree.selection_set(self.results_tree.get_children()[self.current_result_index])
    
    def show_next_result(self):
        if self.current_result_index < len(self.folder_results) - 1:
            self.current_result_index += 1
            self.show_current_result()
            self.update_result_navigation()
            # Also update tree selection
            self.results_tree.selection_set(self.results_tree.get_children()[self.current_result_index])
    
    def show_current_result(self):
        if not self.folder_results:
            return
        
        result = self.folder_results[self.current_result_index]
        
        # Update display
        self.display_visualizations(
            result['mel_spec_db'], 
            result['predicted_class'], 
            result['all_predictions']
        )
        
        # Update audio data (for playback if implemented later)
        self.audio_data = result['audio']
        self.sr = result['sr']
        
        # Update result label
        self.result_label.config(
            text=f"Fichier: {result['file']} | Oiseau détecté: {result['predicted_class']} (Confiance: {result['confidence']:.2f})"
        )
    
    def update_result_navigation(self):
        total = len(self.folder_results)
        current = self.current_result_index + 1
        
        self.results_indicator.config(text=f"{current} / {total}")
        
        # Enable/disable navigation buttons
        self.prev_button.config(state=tk.NORMAL if self.current_result_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_result_index < total - 1 else tk.DISABLED)
    
    def update_class_statistics(self):
        """Update the statistics display with class counts"""
        # Clear existing stats
        for widget in self.stats_scrollable_frame.winfo_children():
            widget.destroy()
        
        if not self.folder_results:
            return
        
        # Count occurrences of each class
        class_counts = Counter([result['predicted_class'] for result in self.folder_results])
        
        # Calculate confidence stats
        class_confidences = {}
        for cls in class_counts.keys():
            confidences = [result['confidence'] for result in self.folder_results 
                          if result['predicted_class'] == cls]
            class_confidences[cls] = {
                'avg': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences)
            }
        
        # Create header
        header_frame = ttk.Frame(self.stats_scrollable_frame)
        header_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(header_frame, text="Classe", width=15, font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_frame, text="Nombre", width=8, font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_frame, text="% du total", width=10, font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_frame, text="Conf. Moy.", width=10, font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_frame, text="Conf. Min", width=10, font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Label(header_frame, text="Conf. Max", width=10, font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Add separator
        ttk.Separator(self.stats_scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=2)
        
        # Total count
        total_files = len(self.folder_results)
        
        # Sort classes by count (most frequent first)
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            cls_frame = ttk.Frame(self.stats_scrollable_frame)
            cls_frame.pack(fill=tk.X, pady=2)
            
            percentage = (count / total_files) * 100
            conf_stats = class_confidences[cls]
            
            ttk.Label(cls_frame, text=cls, width=15).pack(side=tk.LEFT, padx=5)
            ttk.Label(cls_frame, text=str(count), width=8).pack(side=tk.LEFT, padx=5)
            ttk.Label(cls_frame, text=f"{percentage:.1f}%", width=10).pack(side=tk.LEFT, padx=5)
            ttk.Label(cls_frame, text=f"{conf_stats['avg']:.3f}", width=10).pack(side=tk.LEFT, padx=5)
            ttk.Label(cls_frame, text=f"{conf_stats['min']:.3f}", width=10).pack(side=tk.LEFT, padx=5)
            ttk.Label(cls_frame, text=f"{conf_stats['max']:.3f}", width=10).pack(side=tk.LEFT, padx=5)
        
        # Add total row
        ttk.Separator(self.stats_scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=2)
        
        total_frame = ttk.Frame(self.stats_scrollable_frame)
        total_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(total_frame, text="TOTAL", width=15, font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Label(total_frame, text=str(total_files), width=8, font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Label(total_frame, text="100%", width=10).pack(side=tk.LEFT, padx=5)
        
        # Adjust canvas height based on content
        self.stats_scrollable_frame.update_idletasks()
        frame_height = min(150, self.stats_scrollable_frame.winfo_reqheight())
        self.stats_canvas.config(height=frame_height)
        self.stats_content.config(height=frame_height)
   
    def display_visualizations(self, mel_spec_db, predicted_class, all_predictions):
        self.fig.clear()
       
        # Create subplots for waveform, spectrogram, and classification results
        # GridSpec allows more control over subplot sizes
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.2])
       
        ax1 = self.fig.add_subplot(gs[0])
        ax2 = self.fig.add_subplot(gs[1])
        ax3 = self.fig.add_subplot(gs[2])
       
        # Plot waveform
        librosa.display.waveshow(self.audio_data, sr=self.sr, ax=ax1)
        ax1.set_title('Forme d\'onde')
       
        # Plot spectrogram
        img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel',
                                     sr=self.sr, fmax=8000, ax=ax2)
        self.fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        ax2.set_title('Spectrogramme Mel')
       
        # Plot classification results as horizontal bar chart
        # Sort by probability
        sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        birds = [item[0] for item in sorted_predictions]
        probs = [item[1] for item in sorted_predictions]
       
        # Horizontal bar chart
        bars = ax3.barh(birds, probs, color='skyblue')
       
        # Highlight the predicted class
        for i, bar in enumerate(bars):
            if birds[i] == predicted_class:
                bar.set_color('orange')
       
        # Add text with exact probability values
        for i, v in enumerate(probs):
            ax3.text(v + 0.01, i, f'{v:.3f}', va='center')
       
        ax3.set_xlim(0, 1.1)  # Set x-axis limit to accommodate text
        ax3.set_title('Résultats de classification')
        ax3.set_xlabel('Probabilité')
       
        self.fig.tight_layout()
        self.canvas.draw()
 
if __name__ == '__main__':
    root = tk.Tk()
    app = BirdSoundClassifierApp(root)
    root.mainloop()
 
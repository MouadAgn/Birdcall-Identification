import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

# Importer les classes nécessaires de notre modèle d'entraînement
from train import BirdAudioDataset, ImprovedAudioClassifier

# Configuration
MODEL_PATH = "best_bird_audio_classifier.pth"
SAMPLE_RATE = 16000
N_MELS = 128
TARGET_LENGTH = 156
class AudioPlayer:
    def __init__(self):
        self.playing = False
        self.audio_thread = None
    
    def play_audio(self, filepath):
        import pygame
        
        if self.playing:
            self.stop()
        
        pygame.mixer.init()
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        self.playing = True
        
        # Créer un thread pour suivre quand l'audio se termine
        self.audio_thread = threading.Thread(target=self._check_audio_status)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def _check_audio_status(self):
        import pygame
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        self.playing = False
    
    def stop(self):
        if self.playing:
            import pygame
            pygame.mixer.music.stop()
            self.playing = False

class BirdSoundClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Classification de Sons d\'Oiseaux')
        self.root.geometry('700x600')
        
        # Charger le modèle
        self.load_model()
        
        # Player audio
        self.player = AudioPlayer()
        
        # Interface utilisateur
        self.setup_ui()
    
    def load_model(self):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            self.classes = checkpoint['classes']
            self.class_to_idx = checkpoint['class_to_idx']
            self.model = ImprovedAudioClassifier(n_classes=len(self.classes), n_mels=N_MELS)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Modèle chargé avec succès. Classes: {self.classes}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            self.classes = []
            self.model = None
    
    def setup_ui(self):
        # Section haut - instructions
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.pack(fill=tk.X)
        
        ttk.Label(header_frame, text='Classification de Sons d\'Oiseaux', font=('Arial', 16, 'bold')).pack()
        ttk.Label(header_frame, text='Sélectionnez un fichier audio (.wav ou .ogg) à classifier').pack(pady=5)
        
        # Section boutons
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill=tk.X)
        
        self.upload_button = ttk.Button(button_frame, text='Sélectionner Audio', command=self.upload_audio)
        self.upload_button.pack(side=tk.LEFT, padx=5)
        
        self.play_button = ttk.Button(button_frame, text='Écouter', command=self.play_audio, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        self.predict_button = ttk.Button(button_frame, text='Classifier', command=self.predict, state=tk.DISABLED)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        # Section du spectrogramme
        self.spec_frame = ttk.LabelFrame(self.root, text="Spectrogramme", padding="10")
        self.spec_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Créer une figure matplotlib pour le spectrogramme
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.spec_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Section résultats
        result_frame = ttk.LabelFrame(self.root, text="Résultats", padding="10")
        result_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.result_label = ttk.Label(result_frame, text='En attente de classification...', font=('Arial', 12))
        self.result_label.pack(pady=5)
        
        # Barre de statut
        self.status_bar = ttk.Label(self.root, text='Prêt', relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Variables pour stocker le chemin de fichier actuel
        self.current_file = None
        self.waveform = None
        self.sr = None
    
    def upload_audio(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.wav *.ogg"), ("All files", "*.*")]
        )
        if file_path:
            self.current_file = file_path
            self.status_bar['text'] = f"Fichier sélectionné : {os.path.basename(file_path)}"
            
            # Activer les boutons
            self.play_button['state'] = tk.NORMAL
            self.predict_button['state'] = tk.NORMAL
            
            # Charger et traiter l'audio
            self.load_audio()
            
            # Afficher le spectrogramme
            self.display_spectrogram()
    
    def load_audio(self):
        try:
            # Charger l'audio avec la logique adaptée de notre Dataset
            self.status_bar['text'] = "Chargement de l'audio..."
            
            try:
                if self.current_file.lower().endswith(".ogg"):
                    try:
                        # Essayer d'abord avec soundfile
                        self.waveform, self.sr = torchaudio.load(self.current_file, backend="soundfile")
                    except Exception:
                        # Si échec, essayer avec librosa
                        import librosa
                        audio, sr = librosa.load(self.current_file, sr=SAMPLE_RATE, mono=True)
                        self.waveform = torch.tensor(audio).unsqueeze(0)
                        self.sr = sr
                else:
                    self.waveform, self.sr = torchaudio.load(self.current_file)
            except Exception as e:
                self.status_bar['text'] = f"Erreur: {str(e)}"
                return False
            
            # Conversion mono
            if self.waveform.shape[0] > 1:
                self.waveform = torch.mean(self.waveform, dim=0, keepdim=True)
            
            # Rééchantillonnage
            if self.sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(self.sr, SAMPLE_RATE)
                self.waveform = resampler(self.waveform)
                self.sr = SAMPLE_RATE
            
            self.status_bar['text'] = "Audio chargé avec succès"
            return True
        except Exception as e:
            self.status_bar['text'] = f"Erreur lors du chargement: {str(e)}"
            return False
    
    def display_spectrogram(self):
        if self.waveform is None:
            return
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=2048,
            hop_length=512
        )
        db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
        mel_spec = mel_spectrogram(self.waveform)
        log_mel_spec = db_transform(mel_spec)
        self.ax.clear()
        self.ax.imshow(log_mel_spec[0, :, :].numpy(), aspect='auto', origin='lower', cmap='viridis')
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Mel bin')
        self.ax.set_title('Mel Spectrogram')
        self.fig.tight_layout()
        self.canvas.draw()
    
    def play_audio(self):
        if self.current_file:
            self.player.play_audio(self.current_file)
            self.status_bar['text'] = "Lecture en cours..."
    
    def predict(self):
        if not self.model or self.waveform is None:
            self.status_bar['text'] = "Modèle non chargé ou audio non disponible"
            return
        try:
            self.status_bar['text'] = "Classification en cours..."
            max_length = 5 * SAMPLE_RATE
            waveform = self.waveform.clone()
            if waveform.shape[1] > max_length:
                waveform = waveform[:, :max_length]
            if waveform.shape[1] < max_length:
                padding = max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            transform = torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=SAMPLE_RATE,
                    n_mels=N_MELS,
                    n_fft=2048,
                    hop_length=512,
                    window_fn=torch.hann_window
                ),
                torchaudio.transforms.AmplitudeToDB(top_db=80)
            )
            spec = transform(waveform)
            # Normalisation comme dans le dataset
            spec = (spec - spec.mean()) / (spec.std() + 1e-10)
            # Ajuster la taille temporelle à 156 frames
            current_length = spec.shape[2]
            if current_length > TARGET_LENGTH:
                spec = spec[:, :, :TARGET_LENGTH]
            elif current_length < TARGET_LENGTH:
                padding = TARGET_LENGTH - current_length
                spec = torch.nn.functional.pad(spec, (0, padding))
            spec = spec.unsqueeze(0)  # [1, 1, n_mels, time]
            with torch.no_grad():
                outputs = self.model(spec)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item()
                predicted_class = self.classes[predicted_idx]
            self.result_label['text'] = f"Prédiction: {predicted_class} ({confidence*100:.1f}% confiance)"
            self.status_bar['text'] = "Classification terminée"
        except Exception as e:
            self.status_bar['text'] = f"Erreur lors de la prédiction: {str(e)}"
            self.result_label['text'] = "Erreur de classification"

if __name__ == "__main__":
    # Pour utiliser pygame (lecture audio)
    try:
        import pygame
    except ImportError:
        print("Pygame n'est pas installé. Installez-le avec: pip install pygame")
        print("La lecture audio ne sera pas disponible.")
    
    root = tk.Tk()
    app = BirdSoundClassifierApp(root)
    root.mainloop()
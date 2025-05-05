import base64
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from app import ImprovedAudioClassifier


class AudioClassifier:
    def __init__(self, model_path="final_bird_audio_classifier.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 16000
        self.n_mels = 92  # Changed from 128 to 92
        
        # Load the model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint['classes']
        self.class_to_idx = checkpoint['class_to_idx']
        
        # Create model and load weights

        self.model = ImprovedAudioClassifier(n_classes=len(self.classes), n_mels=self.n_mels)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Create transform
        self.transform = torch.nn.Sequential(
            MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,  # Changed to use the updated n_mels value
                n_fft=2048,
                hop_length=512,
                window_fn=torch.hann_window
            ),
            AmplitudeToDB(top_db=80)
        )
        
    def process_audio(self, audio_path):
        """Process audio file and return spectrogram"""
        try:
            # Load audio
            if audio_path.lower().endswith(".ogg"):
                waveform, sr = torchaudio.load(audio_path, backend="soundfile")
            else:
                waveform, sr = torchaudio.load(audio_path)
                
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
                
            # Standardize length
            max_length = 5 * self.sample_rate  # 5 seconds
            if waveform.shape[1] > max_length:
                waveform = waveform[:, :max_length]
            
            if waveform.shape[1] < max_length:
                padding = max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
                
            return waveform, sr
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None, None
            
    def get_waveform_image(self, waveform, sr):
        """Generate and return base64 image of waveform"""
        plt.figure(figsize=(10, 3))
        plt.plot(waveform[0].numpy())
        plt.title('Waveform')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return base64.b64encode(buf.read()).decode('utf-8')
        
    def get_spectrogram_image(self, spec):
        """Generate and return base64 image of spectrogram"""
        plt.figure(figsize=(10, 4))
        plt.imshow(spec[0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.xlabel('Time Frames')
        plt.ylabel('Mel Bands')
        plt.tight_layout()
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return base64.b64encode(buf.read()).decode('utf-8')
        
    def classify(self, audio_path):
        """Classify an audio file and return results with visualizations"""
        waveform, sr = self.process_audio(audio_path)
        
        if waveform is None:
            return {
                'success': False,
                'error': 'Failed to process audio file'
            }
        
        # Generate spectrogram
        spec = self.transform(waveform)
        
        # Normalize spectrogram
        spec = (spec - spec.mean()) / (spec.std() + 1e-10)
        
        # Ensure correct time dimension (156 frames)
        target_length = 156
        current_length = spec.shape[2]
        
        if current_length > target_length:
            spec = spec[:, :, :target_length]
        elif current_length < target_length:
            padding = target_length - current_length
            spec = torch.nn.functional.pad(spec, (0, padding))
        
        # Get waveform and spectrogram images
        waveform_img = self.get_waveform_image(waveform, sr)
        spec_img = self.get_spectrogram_image(spec)
        
        # Make prediction
        with torch.no_grad():
            input_tensor = spec.unsqueeze(0).to(self.device)  # Add batch dimension
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            predictions = []
            for i in range(len(top_indices)):
                idx = top_indices[i].item()
                bird_class = self.classes[idx]
                probability = top_probs[i].item() * 100
                predictions.append({
                    'class': bird_class,
                    'probability': probability
                })
        
        return {
            'success': True,
            'waveform_img': waveform_img,
            'spectrogram_img': spec_img,
            'predictions': predictions
        }
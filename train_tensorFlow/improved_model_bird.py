import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import random
import warnings
import pickle
import time
import gc
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

# Ignorer les avertissements
warnings.filterwarnings('ignore')

# Définir les chemins des datasets
ORIGINAL_DATASET_PATH = "IA_BIRD/dataset-2"
SEGMENTS_DATASET_PATH = "IA_BIRD/dataset/best_segments_bird_audio"
MODELS_DIR = "IA_BIRD/models_result"

# Créer le dossier pour sauvegarder les modèles s'il n'existe pas
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def get_next_version_dir():
    """Trouve le prochain numéro de version disponible et crée le dossier correspondant"""
    version = 1
    while os.path.exists(os.path.join(MODELS_DIR, str(version))):
        version += 1
    
    version_dir = os.path.join(MODELS_DIR, str(version))
    os.makedirs(version_dir)
    print(f"Création du dossier de résultats: {version_dir}")
    return version_dir

# Paramètres globaux
SAMPLE_RATE = 32000  # Taux d'échantillonnage standard
DURATION = 10  # Durée en secondes pour chaque échantillon (standardisée)
N_MELS = 128  # Nombre de filtres Mel pour le spectrogramme
N_FFT = 1024  # Taille de la FFT
HOP_LENGTH = 512  # Pas entre les fenêtres d'analyse
MAX_DB = 80  # Valeur maximale de dB pour le spectrogramme
REF_DB = 1.0  # Valeur de référence pour le spectrogramme
BATCH_SIZE = 8  # Taille des batchs réduite pour éviter l'OOM
EPOCHS = 50  # Augmenté pour permettre un entraînement plus long avec early stopping
MIN_SAMPLES_PER_CLASS = 100  # Nombre minimal d'échantillons par classe après augmentation

# Configuration du GPU si disponible
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs disponibles avec allocation mémoire optimisée: {len(gpus)}")
    except RuntimeError as e:
        print(e)
else:
    print("Aucun GPU détecté, utilisation du CPU")


class ImprovedAudioDataProcessor:
    def __init__(self, dataset_path, sample_rate=32000, duration=5, n_mels=128, n_fft=1024, hop_length=512):
        """
        Initialise le processeur de données audio avec améliorations
        
        Args:
            dataset_path: Chemin vers le dataset
            sample_rate: Taux d'échantillonnage cible
            duration: Durée cible pour chaque échantillon
            n_mels: Nombre de filtres Mel pour le spectrogramme
            n_fft: Taille de la FFT
            hop_length: Pas entre les fenêtres d'analyse
        """
        self.dataset_path = dataset_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_samples = sample_rate * duration
        self.classes = None
        self.class_to_idx = None
        self.class_weights = None
    
    def load_audio_file(self, file_path):
        """Charge un fichier audio et standardise sa durée"""
        try:
            # Charger l'audio
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Standardiser la durée
            if len(audio) > self.target_samples:
                # Si le fichier est plus long, on prend un segment aléatoire
                start = np.random.randint(0, len(audio) - self.target_samples)
                audio = audio[start:start + self.target_samples]
            else:
                # Si le fichier est plus court, on le complète avec du silence
                padding = self.target_samples - len(audio)
                audio = np.pad(audio, (0, padding), 'constant')
            
            return audio
        except Exception as e:
            print(f"Erreur lors du chargement de {file_path}: {e}")
            return None
    
    def audio_to_melspectrogram(self, audio):
        """Convertit un signal audio en spectrogramme Mel avec normalisation améliorée"""
        if audio is None:
            return None
            
        # Calculer le spectrogramme Mel
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convertir en décibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0, top_db=80.0)
        
        # Normaliser entre -1 et 1 (plus robuste que 0-1)
        mel_spec_norm = 2 * ((mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())) - 1
        
        return mel_spec_norm
    
    def apply_data_augmentation(self, audio, augmentation_strength="normal"):
        """
        Applique la data augmentation à un signal audio avec intensité variable
        
        Args:
            audio: Signal audio à augmenter
            augmentation_strength: Intensité de l'augmentation (faible, normal, fort)
        
        Returns:
            Signal audio augmenté
        """
        augmented_audio = audio.copy()
        
        # Configuration des paramètres selon l'intensité
        if augmentation_strength == "strong":
            shift_range = 0.4
            pitch_range = 4
            noise_level = 0.01
            stretch_range = 0.3
            n_min_augmentations = 2
            n_max_augmentations = 4
        elif augmentation_strength == "weak":
            shift_range = 0.1
            pitch_range = 1
            noise_level = 0.002
            stretch_range = 0.1
            n_min_augmentations = 1
            n_max_augmentations = 2
        else:  # normal
            shift_range = 0.2
            pitch_range = 2
            noise_level = 0.005
            stretch_range = 0.2
            n_min_augmentations = 1
            n_max_augmentations = 3
        
        # Liste des techniques d'augmentation
        augmentation_techniques = [
            lambda x: self._time_shift(x, shift_range),
            lambda x: self._pitch_shift(x, pitch_range),
            lambda x: self._add_noise(x, noise_level),
            lambda x: self._time_stretch(x, stretch_range),
            lambda x: self._change_volume(x),
            lambda x: self._add_background_noise(x)
        ]
        
        # Appliquer aléatoirement plusieurs techniques
        n_augmentations = np.random.randint(n_min_augmentations, n_max_augmentations + 1)
        selected_techniques = random.sample(augmentation_techniques, n_augmentations)
        
        for technique in selected_techniques:
            augmented_audio = technique(augmented_audio)
        
        return augmented_audio
    
    def _time_shift(self, audio, shift_range=0.2):
        """Décale le signal dans le temps"""
        shift = int(np.random.uniform(-shift_range, shift_range) * len(audio))
        if shift > 0:
            shifted_audio = np.concatenate((np.zeros(shift), audio[:-shift]))
        else:
            shifted_audio = np.concatenate((audio[-shift:], np.zeros(-shift)))
        return shifted_audio
    
    def _pitch_shift(self, audio, pitch_range=2):
        """Modifie la hauteur du signal"""
        pitch_factor = np.random.uniform(-pitch_range, pitch_range)
        try:
            return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=pitch_factor)
        except:
            return audio
    
    def _add_noise(self, audio, noise_level=0.005):
        """Ajoute du bruit gaussien au signal"""
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
    
    def _time_stretch(self, audio, stretch_range=0.2):
        """Étire ou compresse le signal temporellement"""
        rate = np.random.uniform(1-stretch_range, 1+stretch_range)
        try:
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            # Ajuster la longueur
            if len(stretched) > self.target_samples:
                stretched = stretched[:self.target_samples]
            else:
                stretched = np.pad(stretched, (0, max(0, self.target_samples - len(stretched))), 'constant')
            return stretched
        except:
            return audio
    
    def _change_volume(self, audio, volume_range=0.3):
        """Modifie le volume du signal"""
        # Amplification aléatoire entre 70% et 130% du volume original
        gain = np.random.uniform(1-volume_range, 1+volume_range)
        return audio * gain
    
    def _add_background_noise(self, audio, intensity=0.05):
        """Simule un bruit de fond ambiant"""
        # Générer un bruit rose (adapté aux sons naturels)
        noise_len = len(audio)
        noise = np.random.randn(noise_len)
        # Filtrer pour obtenir un bruit plus naturel (bruit rose)
        noise = np.cumsum(noise)
        noise = noise / np.max(np.abs(noise))  # Normaliser
        
        # Mélanger le bruit avec le signal
        intensity = np.random.uniform(0.01, intensity)
        return audio * (1 - intensity) + noise * intensity
    
    def generate_augmented_samples(self, audio, class_name, num_samples=5):
        """
        Génère plusieurs échantillons augmentés à partir d'un seul audio
        
        Args:
            audio: Signal audio original
            class_name: Nom de la classe (pour adapter l'intensité)
            num_samples: Nombre d'échantillons à générer
            
        Returns:
            Liste de signaux audio augmentés
        """
        augmented_samples = []
        
        # Adapter l'intensité de l'augmentation selon la classe
        # Augmentation plus forte pour les classes sous-représentées
        if class_name in ["easmog1", "abethr1", "bagwea1", "darbar1"]:
            strength = "strong"
            num_samples = max(num_samples, 10)  # Plus d'échantillons pour les classes rares
        else:
            strength = "normal"
        
        for _ in range(num_samples):
            augmented = self.apply_data_augmentation(audio, strength)
            augmented_samples.append(augmented)
        
        return augmented_samples
    
    def prepare_dataset(self, augment=True, balance_classes=True):
        """
        Prépare le dataset en chargeant les fichiers audio et en créant les spectrogrammes
        
        Args:
            augment: Si True, applique la data augmentation
            balance_classes: Si True, équilibre les classes
            
        Returns:
            X: Liste des spectrogrammes (features)
            y: Liste des labels
            classes: Liste des classes
        """
        X, y = [], []
        file_mapping = []  # Pour tracer l'origine des échantillons
        
        # Trouver toutes les classes
        class_dirs = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        # Créer le mapping des classes
        self.classes = sorted(class_dirs)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        print(f"Classes trouvées: {self.classes}")
        print(f"Préparation du dataset depuis {self.dataset_path}")
        
        # Compteur d'échantillons par classe
        class_counts = {cls: 0 for cls in self.classes}
        
        # Parcourir les classes
        for class_name in class_dirs:
            class_path = os.path.join(self.dataset_path, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Lister les fichiers audio
            audio_files = [f for f in os.listdir(class_path) if f.endswith(('.wav', '.mp3', '.ogg'))]
            
            print(f"Traitement de la classe {class_name} ({len(audio_files)} fichiers)")
            
            for audio_file in tqdm(audio_files):
                file_path = os.path.join(class_path, audio_file)
                
                # Charger le fichier audio
                audio = self.load_audio_file(file_path)
                if audio is None:
                    continue
                
                # Créer le spectrogramme et ajouter aux données
                mel_spec = self.audio_to_melspectrogram(audio)
                if mel_spec is not None:
                    X.append(mel_spec)
                    y.append(class_idx)
                    file_mapping.append(file_path)
                    class_counts[class_name] += 1
                
                # Appliquer la data augmentation si demandé
                if augment:
                    # Nombre d'échantillons augmentés à générer
                    # Adapter selon la classe pour équilibrer
                    augmented_samples = self.generate_augmented_samples(
                        audio, 
                        class_name,
                        num_samples=5
                    )
                    
                    for aug_audio in augmented_samples:
                        aug_mel_spec = self.audio_to_melspectrogram(aug_audio)
                        if aug_mel_spec is not None:
                            X.append(aug_mel_spec)
                            y.append(class_idx)
                            file_mapping.append(f"{file_path} (augmented)")
                            class_counts[class_name] += 1
        
        # Afficher le nombre d'échantillons par classe après augmentation
        print("\nNombre d'échantillons par classe après augmentation:")
        for cls, count in class_counts.items():
            print(f"{cls}: {count} échantillons")
        
        X = np.array(X)
        y = np.array(y)
        
        # Calculer les poids de classe pour l'entraînement
        self._calculate_class_weights(y)
        
        # Ajouter une dimension pour les canaux (1 canal car c'est une image en niveaux de gris)
        X = X[..., np.newaxis]
        
        print(f"Dataset préparé: {X.shape[0]} échantillons, {len(self.classes)} classes")
        
        return X, y, self.classes
    
    def _calculate_class_weights(self, y):
        """Calcule les poids de classe pour l'entraînement"""
        class_counts = Counter(y)
        total_samples = len(y)
        n_classes = len(class_counts)
        
        # Calculer les poids de classe
        self.class_weights = {
            cls_idx: total_samples / (n_classes * count)
            for cls_idx, count in class_counts.items()
        }
        
        print("\nPoids des classes pour l'entraînement:")
        for cls_idx, weight in self.class_weights.items():
            cls_name = self.classes[cls_idx]
            print(f"{cls_name}: {weight:.2f}")
        
        return self.class_weights


def create_improved_model(input_shape, num_classes):
    """
    Crée un modèle CNN plus efficace avec une meilleure architecture
    
    Args:
        input_shape: Forme des données d'entrée (spectrogramme)
        num_classes: Nombre de classes à prédire
        
    Returns:
        model: Modèle compilé
    """
    # Activer le calcul en précision mixte pour accélérer l'entraînement et réduire la mémoire
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Calcul en précision mixte activé")
    except:
        print("Impossible d'activer le calcul en précision mixte, utilisation de float32")
    
    # Paramètre de régularisation L2
    reg = l2(0.0005)  # Réduit pour éviter la sur-régularisation
    
    model = Sequential([
        # Première couche de convolution
        Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg,
              input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Deuxième couche de convolution
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        # Troisième couche de convolution (réduite pour éviter les problèmes de mémoire)
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Aplatir pour les couches denses
        Flatten(),
        
        # Utilisation de couches denses optimisées
        Dense(256, activation='relu', kernel_regularizer=reg),
        BatchNormalization(),
        Dropout(0.3),
        
        # Couche de sortie
        Dense(num_classes, activation='softmax')
    ])
    
    # Compiler le modèle avec un taux d'apprentissage fixe initial
    # (pour éviter les problèmes avec le callback LRHistory)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Taux d'apprentissage simplifié
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_and_evaluate_improved(X, y, classes, class_weights, dataset_name, save_dir):
    """
    Entraîne et évalue un modèle sur un dataset avec des techniques avancées
    
    Args:
        X: Features (spectrogrammes)
        y: Labels
        classes: Liste des classes
        class_weights: Poids des classes pour l'entraînement
        dataset_name: Nom du dataset (pour la sauvegarde)
        save_dir: Répertoire de sauvegarde des résultats
        
    Returns:
        history: Historique d'entraînement
        model: Modèle entraîné
        evaluation: Résultats d'évaluation
    """
    # Diviser en train/validation/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Convertir les labels en one-hot encoding
    num_classes = len(classes)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    print(f"Forme des données d'entraînement: {X_train.shape}")
    print(f"Forme des données de validation: {X_val.shape}")
    print(f"Forme des données de test: {X_test.shape}")
    
    # Vérifier la distribution des classes
    print("\nDistribution des classes dans l'ensemble d'entraînement:")
    train_class_counts = Counter(y_train)
    for cls_idx, count in train_class_counts.items():
        cls_name = classes[cls_idx]
        print(f"{cls_name}: {count} échantillons ({count/len(y_train)*100:.1f}%)")
    
    # Créer le modèle
    input_shape = X_train.shape[1:]
    model = create_improved_model(input_shape, num_classes)
    
    # Résumé du modèle
    model.summary()
    
    # Callbacks pour l'entraînement
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(save_dir, f'improved_model_{dataset_name}.h5'),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Callback corrigé pour enregistrer le taux d'apprentissage
    class LRHistory(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            # Récupérer le learning rate actuel en toute sécurité
            try:
                lr = float(self.model.optimizer._decayed_lr(tf.float32).numpy())
            except:
                try:
                    lr = float(self.model.optimizer.lr)
                except:
                    lr = 0.0
            logs['lr'] = lr
    
    lr_history = LRHistory()
    
    # Entraîner le modèle
    print(f"Début de l'entraînement sur le dataset {dataset_name}...")
    start_time = time.time()
    
    # Utiliser des données réduites en taille si nécessaire
    try:
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[model_checkpoint, early_stopping, reduce_lr, lr_history],
            class_weight=class_weights,
            verbose=1
        )
    except Exception as e:
        print(f"Erreur pendant l'entraînement: {e}")
        print("Tentative avec un batch size réduit...")
        # Réduire la taille du batch et réessayer
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            batch_size=BATCH_SIZE // 2,  # Réduire de moitié
            epochs=EPOCHS,
            callbacks=[model_checkpoint, early_stopping, reduce_lr, lr_history],
            class_weight=class_weights,
            verbose=1
        )
    
    train_duration = time.time() - start_time
    print(f"Entraînement terminé en {train_duration:.2f} secondes")
    
    # Évaluer le modèle
    print("Évaluation du modèle sur les données de test...")
    evaluation = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test loss: {evaluation[0]:.4f}")
    print(f"Test accuracy: {evaluation[1]:.4f}")
    
    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    
    # Rapport de classification détaillé
    class_report = classification_report(y_test, y_pred_classes, target_names=classes, output_dict=True)
    
    # Afficher le rapport de classification
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred_classes, target_names=classes))
    
    # Visualiser les résultats
    plot_improved_training_history(history.history, dataset_name, save_dir)
    plot_confusion_matrix(conf_matrix, classes, dataset_name, save_dir)
    plot_class_metrics(class_report, classes, dataset_name, save_dir)
    if 'lr' in history.history:
        plot_learning_rate(history.history, dataset_name, save_dir)
    
    # Sauvegarder les résultats d'évaluation
    results = {
        'history': history.history,
        'evaluation': {
            'loss': evaluation[0],
            'accuracy': evaluation[1]
        },
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'train_duration': train_duration,
        'classes': classes,
        'class_distribution': {
            'train': Counter(y_train),
            'val': Counter(y_val),
            'test': Counter(y_test)
        }
    }
    
    with open(os.path.join(save_dir, f'improved_results_{dataset_name}.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    return history, model, results


def plot_improved_training_history(history, dataset_name, save_dir):
    """Visualise l'historique d'entraînement avec plus de détails"""
    # Créer la figure avec 2 sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Graphique de précision
    ax1.plot(history['accuracy'], label='Train')
    ax1.plot(history['val_accuracy'], label='Validation')
    ax1.set_title('Précision')
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Précision')
    ax1.legend()
    ax1.grid(True)
    
    # Ajouter des annotations pour les valeurs finales
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    ax1.annotate(f"{final_train_acc:.4f}", 
                xy=(len(history['accuracy'])-1, final_train_acc),
                xytext=(5, 5), textcoords='offset points')
    ax1.annotate(f"{final_val_acc:.4f}", 
                xy=(len(history['val_accuracy'])-1, final_val_acc),
                xytext=(5, 5), textcoords='offset points')
    
    # Graphique de perte
    ax2.plot(history['loss'], label='Train')
    ax2.plot(history['val_loss'], label='Validation')
    ax2.set_title('Perte')
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Perte')
    ax2.legend()
    ax2.grid(True)
    
    # Ajouter des annotations pour les valeurs finales
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    ax2.annotate(f"{final_train_loss:.4f}", 
                xy=(len(history['loss'])-1, final_train_loss),
                xytext=(5, 5), textcoords='offset points')
    ax2.annotate(f"{final_val_loss:.4f}", 
                xy=(len(history['val_loss'])-1, final_val_loss),
                xytext=(5, 5), textcoords='offset points')
    
    plt.suptitle(f'Historique d\'entraînement amélioré - {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'improved_history_{dataset_name}.png'))
    plt.close()


def plot_confusion_matrix(conf_matrix, classes, dataset_name, save_dir):
    """Visualise la matrice de confusion avec une colormap améliorée"""
    plt.figure(figsize=(12, 10))
    
    # Normaliser la matrice pour obtenir des pourcentages
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # Créer la heatmap
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.title(f'Matrice de confusion normalisée - {dataset_name}')
    plt.ylabel('Classe réelle')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{dataset_name}.png'))
    plt.close()
    
    # Version non normalisée pour voir les nombres absolus
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.title(f'Matrice de confusion (valeurs absolues) - {dataset_name}')
    plt.ylabel('Classe réelle')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_abs_{dataset_name}.png'))
    plt.close()


def plot_class_metrics(class_report, classes, dataset_name, save_dir):
    """Visualise les métriques par classe (précision, rappel, F1-score)"""
    # Extraire les métriques pour chaque classe
    metrics = {}
    for cls in classes:
        if cls in class_report:
            metrics[cls] = {
                'precision': class_report[cls]['precision'],
                'recall': class_report[cls]['recall'],
                'f1-score': class_report[cls]['f1-score']
            }
    
    # Convertir en DataFrame pour faciliter la visualisation
    df = pd.DataFrame(metrics).T
    
    # Plot
    plt.figure(figsize=(12, 8))
    df.plot(kind='bar', figsize=(15, 8))
    plt.title(f'Métriques par classe - {dataset_name}')
    plt.ylabel('Score')
    plt.xlabel('Classe')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'class_metrics_{dataset_name}.png'))
    plt.close()
    
    # Visualisation en heatmap
    plt.figure(figsize=(12, len(classes)*0.4 + 2))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlGnBu')
    plt.title(f'Heatmap des métriques par classe - {dataset_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'class_metrics_heatmap_{dataset_name}.png'))
    plt.close()


def plot_learning_rate(history, dataset_name, save_dir):
    """Visualise l'évolution du taux d'apprentissage pendant l'entraînement"""
    # Vérifier si le taux d'apprentissage a été enregistré
    if 'lr' not in history:
        print("Aucune information sur le taux d'apprentissage n'a été enregistrée.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['lr'])
    plt.title(f'Évolution du taux d\'apprentissage - {dataset_name}')
    plt.xlabel('Époque')
    plt.ylabel('Taux d\'apprentissage')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'learning_rate_{dataset_name}.png'))
    plt.close()


def plot_class_distribution(class_counts, classes, title, save_path):
    """Visualise la distribution des classes dans le dataset"""
    # Préparer les données pour le graphique
    counts = []
    labels = []
    for cls_idx in sorted(class_counts.keys()):
        if cls_idx < len(classes):
            counts.append(class_counts[cls_idx])
            labels.append(classes[cls_idx])
    
    # Créer le graphique
    plt.figure(figsize=(14, 8))
    bars = plt.bar(labels, counts, color='skyblue')
    
    # Ajouter les valeurs sur chaque barre
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}',
                ha='center', va='bottom', rotation=0)
    
    plt.title(title)
    plt.xlabel('Classe')
    plt.ylabel('Nombre d\'échantillons')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Version en camembert pour les classes ayant beaucoup d'échantillons
    plt.figure(figsize=(12, 12))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(f"{title} (Vue proportionnelle)")
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_pie.png'))
    plt.close()


def main():
    """Fonction principale"""
    print("=== ANALYSE ET CLASSIFICATION AMÉLIORÉE DES SONS D'OISEAUX ===")
    
    # Optimisation de la mémoire GPU
    tf.keras.backend.clear_session()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Limiter la mémoire GPU allouée
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs disponibles avec allocation mémoire optimisée: {len(gpus)}")
        except RuntimeError as e:
            print(e)
    
    # Tester si le dataset est disponible
    if not os.path.exists(ORIGINAL_DATASET_PATH):
        print(f"ERREUR: Le dataset n'a pas été trouvé: {ORIGINAL_DATASET_PATH}")
        return
    
    # Créer le dossier versionné pour cette exécution
    version_dir = get_next_version_dir()
    
    # Configurer le garbage collector pour libérer plus agressivement la mémoire
    gc.set_threshold(100, 5, 5)
        
    # Traiter le dataset
    print("\n=== TRAITEMENT DU DATASET ===")
    processor = ImprovedAudioDataProcessor(
        dataset_path=ORIGINAL_DATASET_PATH,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    X, y, classes = processor.prepare_dataset(
        augment=False,  # Désactiver l'augmentation des données
        balance_classes=False  # Désactiver l'équilibrage des classes
    )
    
    # Optimisation de la mémoire
    gc.collect()
    
    # Entraîner et évaluer le modèle sur le dataset
    history, model, results = train_and_evaluate_improved(
        X, y, classes, 
        processor.class_weights, "dataset", version_dir
    )
    
    print(f"\nAnalyse terminée. Le modèle amélioré a été sauvegardé dans le dossier: {version_dir}")
    print("Pour utiliser le modèle pour des prédictions, chargez-le avec:")
    print(f"  model = tf.keras.models.load_model('{os.path.join(version_dir, 'improved_model_dataset.h5')}')")


if __name__ == "__main__":
    main() 
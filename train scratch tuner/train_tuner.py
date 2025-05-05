import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from optuna.trial import TrialState
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import (AmplitudeToDB, FrequencyMasking,
                                   MelSpectrogram, TimeMasking)


class AudioAugmentation:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def add_noise(self, waveform, snr_db=10):
        # Ajouter du bruit blanc avec un SNR spécifié
        signal_power = torch.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise
    
    def time_shift(self, waveform, shift_factor=0.2):
        # Décaler le signal dans le temps
        shift = int(waveform.shape[1] * shift_factor)
        shifted = torch.zeros_like(waveform)
        
        if shift > 0:  # shift right
            shifted[:, shift:] = waveform[:, :-shift]
        elif shift < 0:  # shift left
            shift = abs(shift)  # Take absolute value for indexing
            shifted[:, :-shift] = waveform[:, shift:]
        else:  # shift is zero, no change
            return waveform
            
        return shifted
    
    def pitch_shift(self, waveform, pitch_factor=0.1):
        # Utiliser torchaudio quand disponible, sinon retourner l'original
        # Note: Cette implémentation est simplifiée
        try:
            freq_factor = 2 ** pitch_factor
            indices = torch.linspace(0, waveform.shape[1]-1, int(waveform.shape[1]/freq_factor))
            indices = indices.long().clamp(0, waveform.shape[1]-1)
            return waveform[:, indices]
        except:
            return waveform
    
    def spec_augment(self, spec):
        # SpecAugment pour les spectrogrammes (masking)
        freq_mask = FrequencyMasking(freq_mask_param=10)
        time_mask = TimeMasking(time_mask_param=20)
        
        augmented = spec.clone()
        if torch.rand(1) > 0.5:  # 50% chance
            augmented = freq_mask(augmented)
        if torch.rand(1) > 0.5:  # 50% chance
            augmented = time_mask(augmented)
            
        return augmented
    
    def random_augment(self, waveform):
        # Appliquer des augmentations aléatoirement
        augmented = waveform.clone()
        
        # Ajouter du bruit (30% chance)
        if torch.rand(1) > 0.7:
            snr_db = 15 + 10 * torch.rand(1)  # SNR entre 15-25dB
            augmented = self.add_noise(augmented, snr_db)
            
        # Time shift (30% chance)
        if torch.rand(1) > 0.7:
            shift_factor = 0.2 * (torch.rand(1) - 0.5)  # -0.1 à 0.1
            augmented = self.time_shift(augmented, shift_factor)
            
        # Pitch shift (20% chance) - si implémenté complètement
        if torch.rand(1) > 0.8:
            pitch_factor = 0.2 * (torch.rand(1) - 0.5)  # -0.1 à 0.1
            augmented = self.pitch_shift(augmented, pitch_factor)
            
        return augmented


# ---------------------------
#  Dataset pour sons d'oiseaux
# ---------------------------
class BirdAudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, n_mels=128, transform=None, augment=False):
        """
        root_dir doit pointer vers le dossier 'bird_audio' qui contient 
        un sous-dossier par classe (ex: 'abethr1', 'bagwea1', etc.)
        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.transform = transform
        self.augment = augment
        self.augmenter = AudioAugmentation(sample_rate) if augment else None
        self.n_mels = n_mels

        # Construire la liste des classes et mapping vers indices
        self.classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Lister tous les fichiers audio et leur label
        self.samples = []
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for filename in os.listdir(cls_folder):
                if filename.lower().endswith((".wav", ".ogg")):
                    path = os.path.join(cls_folder, filename)
                    self.samples.append((path, self.class_to_idx[cls]))

        # Si pas de transform fourni, créer un MelSpectrogram + DB
        if self.transform is None:
            self.transform = nn.Sequential(
                MelSpectrogram(
                    sample_rate=sample_rate,
                    n_mels=n_mels,
                    n_fft=2048,  # Augmenté pour une meilleure résolution fréquentielle
                    hop_length=512,
                    window_fn=torch.hann_window
                ),
                AmplitudeToDB(top_db=80)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        
        try:
            # Spécifier explicitement le backend pour torchaudio
            if filepath.lower().endswith(".ogg"):
                # Utiliser soundfile pour les fichiers OGG
                waveform, sr = torchaudio.load(filepath, backend="soundfile")
            else:
                # Backend par défaut pour les autres formats
                waveform, sr = torchaudio.load(filepath)
        except Exception as e:
            print(f"Erreur lors du chargement de {filepath}: {e}")
            # Créer un échantillon vide en cas d'erreur
            return torch.zeros(1, self.n_mels, 156), label
        
        # Assurez-vous que nous traitons un seul canal (mono)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Rééchantillonnage si nécessaire
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Limiter la durée pour éviter des spectrogrammes trop grands/variables
        max_length = 5 * self.sample_rate  # 5 secondes max
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
            
        # Padding si le son est trop court
        if waveform.shape[1] < max_length:
            padding = max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Appliquer l'augmentation si activée
        if self.augment:
            waveform = self.augmenter.random_augment(waveform)
        
        spec = self.transform(waveform)  # (1, n_mels, time_frames)
        
        # Appliquer SpecAugment si augmentation activée
        if self.augment:
            spec = self.augmenter.spec_augment(spec)
        
        # Normaliser le spectrogramme
        spec = (spec - spec.mean()) / (spec.std() + 1e-10)
        
        # Standardize the time dimension to a fixed size (156 frames is common for ~5 sec audio)
        target_length = 156
        current_length = spec.shape[2]
        
        if current_length > target_length:
            # If longer, crop to target length
            spec = spec[:, :, :target_length]
        elif current_length < target_length:
            # If shorter, pad to target length
            padding = target_length - current_length
            spec = F.pad(spec, (0, padding))
        
        return spec, label


# ---------------------------
#  Modèle CNN amélioré avec blocs résiduels et attention
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

# Modifié pour accepter des hyperparamètres
class ImprovedAudioClassifier(nn.Module):
    def __init__(self, n_classes, n_mels=128, initial_filters=32, filter_multiplier=2, fc_size=512, dropout=0.5, attention_reduction=16):
        super(ImprovedAudioClassifier, self).__init__()
        
        # Premier bloc de traitement
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, initial_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Calcul des filtres pour chaque couche
        filters_2 = initial_filters * filter_multiplier
        filters_3 = filters_2 * filter_multiplier
        filters_4 = filters_3 * filter_multiplier
        
        # Blocs résiduels
        self.layer2 = ResidualBlock(initial_filters, filters_2, stride=2)
        self.attention1 = ChannelAttention(filters_2, reduction=attention_reduction)
        
        self.layer3 = ResidualBlock(filters_2, filters_3, stride=2)
        self.attention2 = ChannelAttention(filters_3, reduction=attention_reduction)
        
        self.layer4 = ResidualBlock(filters_3, filters_4, stride=2)
        self.attention3 = ChannelAttention(filters_4, reduction=attention_reduction)
        
        # Classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(filters_4, fc_size)
        self.bn_fc = nn.BatchNorm1d(fc_size)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc_size, n_classes)

    def forward(self, x):
        x = self.layer1(x)
        
        x = self.layer2(x)
        x = self.attention1(x)
        
        x = self.layer3(x)
        x = self.attention2(x)
        
        x = self.layer4(x)
        x = self.attention3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ---------------------------
#  Fonctions d'entraînement et d'évaluation
# ---------------------------
def mixup_data(x, y, alpha=0.2):
    batch_size = x.size()[0]
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # Mix inputs
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(model, device, train_loader, criterion, optimizer, epoch, use_mixup=True, mixup_alpha=0.2):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Appliquer mixup si activé
        if use_mixup and torch.rand(1) > 0.5:
            data, targets_a, targets_b, lam = mixup_data(data, target, alpha=mixup_alpha)
            optimizer.zero_grad()
            output = model(data)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
            # Pour le calcul de précision dans le mixup
            _, predicted = output.max(1)
            total += target.size(0)
            correct += (lam * predicted.eq(targets_a).sum().float() + 
                       (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            # Calcul de précision standard
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        epoch_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {running_loss/10:.4f} "
                  f"Acc: {100.*correct/total:.2f}%")
            running_loss = 0.0
    
    return epoch_loss / len(train_loader), 100. * correct / total

def evaluate(model, device, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    print(f'Précision sur le jeu de test: {accuracy:.2f}%')
    return avg_loss, accuracy


# ---------------------------
#  Fonction objective pour Optuna (optimisation des hyperparamètres)
# ---------------------------
def objective(trial, train_dir, val_dir, device, n_classes, sample_rate=16000, epochs_per_trial=8):
    # Définir l'espace de recherche des hyperparamètres
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    
    # Hyperparamètres du modèle
    n_mels = trial.suggest_int("n_mels", 64, 256)
    initial_filters = trial.suggest_categorical("initial_filters", [16, 32, 64])
    filter_multiplier = trial.suggest_categorical("filter_multiplier", [1, 2])
    fc_size = trial.suggest_categorical("fc_size", [256, 512, 1024])
    dropout = trial.suggest_float("dropout", 0.2, 0.7)
    attention_reduction = trial.suggest_categorical("attention_reduction", [8, 16, 32])
    
    # Hyperparamètres de l'augmentation
    use_mixup = trial.suggest_categorical("use_mixup", [True, False])
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.1, 0.5) if use_mixup else 0.2
    
    # Scheduler
    scheduler_type = trial.suggest_categorical("scheduler_type", ["cosine", "reduceLR"])
    
    # Créer les datasets
    train_dataset = BirdAudioDataset(train_dir, sample_rate=sample_rate, n_mels=n_mels, augment=True)
    val_dataset = BirdAudioDataset(val_dir, sample_rate=sample_rate, n_mels=n_mels, augment=False)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Créer le modèle avec les hyperparamètres suggérés
    model = ImprovedAudioClassifier(
        n_classes=n_classes, 
        n_mels=n_mels,
        initial_filters=initial_filters,
        filter_multiplier=filter_multiplier,
        fc_size=fc_size,
        dropout=dropout,
        attention_reduction=attention_reduction
    ).to(device)
    
    # Criterion avec label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    else:  # reduceLR
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Boucle d'entraînement
    best_val_acc = 0
    
    for epoch in range(1, epochs_per_trial + 1):
        # Entraînement
        train_loss, train_acc = train(
            model, device, train_loader, criterion, optimizer, epoch, 
            use_mixup=use_mixup, mixup_alpha=mixup_alpha
        )
        
        # Évaluation
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        
        # Mettre à jour le scheduler
        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)
        
        # Rapport à Optuna
        trial.report(val_acc, epoch)
        
        # Pruning: arrêter les essais non prometteurs
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Sauvegarder la meilleure précision
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return best_val_acc


# ---------------------------
#  Préparation des données
# ---------------------------
def prepare_data_folders(original_dir, data_dir, val_split=0.2):
    print("Préparation des dossiers de données...")
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    # Supprimer les dossiers s'ils existent déjà
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    
    # Créer les dossiers principaux
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Lister toutes les classes
    classes = sorted(entry.name for entry in os.scandir(original_dir) if entry.is_dir())
    
    # Pour chaque classe, créer les dossiers et diviser les fichiers
    for cls in classes:
        # Créer les dossiers de classe dans train et val
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        
        # Lister tous les fichiers audio de cette classe
        cls_folder = os.path.join(original_dir, cls)
        audio_files = [f for f in os.listdir(cls_folder) 
                     if f.lower().endswith((".wav", ".ogg"))]
        
        # Diviser en train/val
        train_files, val_files = train_test_split(
            audio_files, test_size=val_split, random_state=42
        )
        
        # Copier les fichiers vers leurs dossiers respectifs
        for file in train_files:
            shutil.copy(
                os.path.join(cls_folder, file),
                os.path.join(train_dir, cls, file)
            )
        
        for file in val_files:
            shutil.copy(
                os.path.join(cls_folder, file),
                os.path.join(val_dir, cls, file)
            )
            
    print(f"Données préparées : {len(classes)} classes divisées en ensembles train/val")
    return classes, train_dir, val_dir


# ---------------------------
#  Visualisation des résultats
# ---------------------------
def plot_confusion_matrix(model, device, loader, class_names):
    """
    Génère une matrice de confusion simple avec les valeurs brutes.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # Collecte de toutes les prédictions
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Calculer la matrice de confusion avec les valeurs brutes
    cm = confusion_matrix(all_labels, all_preds)
    
    # Créer la visualisation
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '10'})
    
    plt.xlabel('Prédit', fontsize=14)
    plt.ylabel('Vrai', fontsize=14)
    plt.title('Matrice de Confusion', fontsize=16)
    
    # Améliorer la lisibilité des étiquettes
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    
    print("Matrice de confusion sauvegardée dans 'confusion_matrix.png'")


def plot_per_class_accuracy(model, device, loader, class_names):
    """
    Génère un graphique simple montrant la précision par classe.
    """
    model.eval()
    # Dictionnaire pour stocker les prédictions correctes et le total par classe
    class_correct = {i: 0 for i in range(len(class_names))}
    class_total = {i: 0 for i in range(len(class_names))}
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            # Mise à jour des statistiques par classe
            for i in range(len(target)):
                label = target[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
    
    # Calculer la précision par classe
    accuracies = []
    for i in range(len(class_names)):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
        else:
            acc = 0
        accuracies.append(acc)
    
    # Créer le graphique simple
    plt.figure(figsize=(16, 12))
    plt.bar(class_names, accuracies, color='royalblue')
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Précision (%)', fontsize=14)
    plt.title('Précision par classe', fontsize=16)
    
    # Rotation des noms de classes pour la lisibilité
    plt.xticks(rotation=90, fontsize=10)
    
    # Ajouter les valeurs sur chaque barre
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 1, f'{acc:.1f}%', 
                 ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('class_accuracy.png', dpi=300)
    plt.close()
    
    print("Précision par classe sauvegardée dans 'class_accuracy.png'")
    
    # Retourner un dictionnaire avec les précisions par classe
    return {class_names[i]: accuracies[i] for i in range(len(class_names))}


# ---------------------------
#  Fonction pour visualiser les résultats d'Optuna
# ---------------------------
def plot_optuna_results(study):
    # Importance des paramètres
    plt.figure(figsize=(10, 8))
    param_importances = optuna.visualization.plot_param_importances(study)
    plt.title('Importance des Hyperparamètres', fontsize=16)
    plt.tight_layout()
    plt.savefig('param_importances.png', dpi=300)
    plt.close()
    
    # Evolution des scores au fil des essais
    plt.figure(figsize=(10, 6))
    optuna.visualization.plot_optimization_history(study)
    plt.title('Historique d\'Optimisation', fontsize=16)
    plt.tight_layout()
    plt.savefig('optimization_history.png', dpi=300)
    plt.close()
    
    # Graphiques de corrélation des paramètres
    plt.figure(figsize=(12, 10))
    optuna.visualization.plot_contour(study, params=['lr', 'weight_decay'])
    plt.title('Graphique de Contour', fontsize=16)
    plt.tight_layout()
    plt.savefig('contour_plot.png', dpi=300)
    plt.close()


# ---------------------------
#  Programme principal
# ---------------------------
if __name__ == "__main__":
    # Chemin vers les données
    original_dir = "./bird_audio"
    data_dir = "./bird_audio_dataset"
    val_split = 0.2
    sample_rate = 16000
    
    # Utiliser GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du périphérique: {device}")
    
    # Préparer les dossiers et obtenir les classes
    classes, train_dir, val_dir = prepare_data_folders(original_dir, data_dir, val_split)
    n_classes = len(classes)
    
    # Configuration d'Optuna
    print("Configuration de l'optimisation des hyperparamètres avec Optuna...")
    study_name = "bird_classifier_optimization"
    storage_name = "sqlite:///optuna_bird_classifier.db"  # Stockage local pour visualisation ultérieure
    
    # Options: TPE (Tree-structured Parzen Estimator) et pruning
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=5)
    
    # Créer l'étude (mode maximisation car on optimise l'accuracy)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True  # Permettre de reprendre une étude existante
    )
    
    # Lancer l'optimisation des hyperparamètres
    print("Début de l'optimisation des hyperparamètres...")
    start_time = time.time()
    
    # Nombre d'essais et d'époques par essai
    n_trials = 30  # Nombre d'essais pour l'optimisation
    epochs_per_trial = 8  # Nombre d'époques par essai
    
    # Fonction partielle pour passer les paramètres constants à l'objectif
    objective_func = lambda trial: objective(
        trial, train_dir, val_dir, device, n_classes, 
        sample_rate=sample_rate, epochs_per_trial=epochs_per_trial
    )
    
    study.optimize(objective_func, n_trials=n_trials)
    
    optimization_time = time.time() - start_time
    print(f"Optimisation terminée en {optimization_time:.2f} secondes")
    
    # Afficher les meilleurs hyperparamètres trouvés
    print("Meilleurs hyperparamètres trouvés:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    print(f"\nMeilleure valeur obtenue: {study.best_value:.2f}%")
    
    # Visualiser les résultats d'Optuna
    plot_optuna_results(study)
    
    # Entraîner le modèle final avec les meilleurs hyperparamètres
    print("\nEntraînement du modèle final avec les meilleurs hyperparamètres...")
    
    # Extraire les meilleurs hyperparamètres
    best_params = study.best_params
    
    # Configurer le dataset avec les meilleurs paramètres
    batch_size = best_params['batch_size']
    n_mels = best_params['n_mels']
    
    # Créer les datasets finaux
    final_train_dataset = BirdAudioDataset(train_dir, sample_rate=sample_rate, n_mels=n_mels, augment=True)
    final_val_dataset = BirdAudioDataset(val_dir, sample_rate=sample_rate, n_mels=n_mels, augment=False)
    
    # DataLoaders pour l'entraînement final
    final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=4, pin_memory=True)
    final_val_loader = DataLoader(final_val_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)
    
    # Créer le modèle final avec les meilleurs hyperparamètres
    final_model = ImprovedAudioClassifier(
        n_classes=n_classes,
        n_mels=best_params['n_mels'],
        initial_filters=best_params['initial_filters'],
        filter_multiplier=best_params['filter_multiplier'],
        fc_size=best_params['fc_size'],
        dropout=best_params['dropout'],
        attention_reduction=best_params['attention_reduction']
    ).to(device)
    
    # Configurer l'entraînement avec les meilleurs hyperparamètres
    criterion = nn.CrossEntropyLoss(label_smoothing=best_params['label_smoothing'])
    optimizer = optim.AdamW(
        final_model.parameters(), 
        lr=best_params['lr'], 
        weight_decay=best_params['weight_decay']
    )
    
    # Meilleur scheduler trouvé
    if best_params['scheduler_type'] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Paramètres pour l'entraînement final
    final_epochs = 20  # Entraînement plus long pour le modèle final
    early_stopping_patience = 10
    early_stopping_counter = 0
    best_val_loss = float('inf')
    best_acc = 0
    
    # Pour stocker les métriques
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Entraînement final
    for epoch in range(1, final_epochs + 1):
        print(f"\n--- Epoch {epoch}/{final_epochs} ---")
        
        # Entraînement
        train_loss, train_acc = train(
            final_model, device, final_train_loader, criterion, optimizer, epoch,
            use_mixup=best_params['use_mixup'], mixup_alpha=best_params['mixup_alpha']
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Évaluation
        val_loss, val_acc = evaluate(final_model, device, final_val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Mettre à jour scheduler
        if best_params['scheduler_type'] == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)
        
        # Early stopping check and best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            
            # Save best model when accuracy improves
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"Nouveau meilleur modèle avec précision: {best_acc:.2f}%")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': final_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'classes': final_train_dataset.classes,
                    'class_to_idx': final_train_dataset.class_to_idx,
                    'hyperparameters': best_params
                }, "best_bird_audio_classifier.pth")
        else:
            early_stopping_counter += 1
            print(f"Compteur d'early stopping: {early_stopping_counter}/{early_stopping_patience}")
            
        # Check if early stopping criteria is met
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping déclenché! Arrêt de l'entraînement.")
            break
            
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"Meilleure Acc: {best_acc:.2f}%")
    
    # Visualisation des métriques
    plt.figure(figsize=(12, 5))

    # Graphique d'accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation')
    plt.title('Précision par époque')
    plt.xlabel('Époque')
    plt.ylabel('Précision (%)')
    plt.legend()
    plt.grid(True)

    # Graphique de loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation')
    plt.title('Loss par époque')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('bird_classifier_metrics.png')
    plt.show()
    
    # Charger le meilleur modèle pour les visualisations finales
    checkpoint = torch.load("best_bird_audio_classifier.pth")
    final_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Afficher la matrice de confusion
    print("Génération de la matrice de confusion...")
    plot_confusion_matrix(final_model, device, final_val_loader, final_train_dataset.classes)
    
    # Afficher la précision par classe
    print("Génération de la précision par classe...")
    class_accuracies = plot_per_class_accuracy(final_model, device, final_val_loader, final_train_dataset.classes)
    
    # Sauvegarder le modèle final
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'classes': final_train_dataset.classes,
        'class_to_idx': final_train_dataset.class_to_idx,
        'hyperparameters': best_params
    }, "final_bird_audio_classifier.pth")
    
    print(f"\nEntraînement terminé!")
    print(f"Meilleure précision: {best_acc:.2f}%")
    print(f"Hyperparamètres optimaux:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")
    print(f"Graphiques sauvegardés dans 'bird_classifier_metrics.png'")
    print(f"Résultats d'optimisation sauvegardés dans 'param_importances.png' et 'optimization_history.png'")
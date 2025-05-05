import os
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename


# Import model components needed for classifier
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
    def __init__(self, channels, reduction=32):  # Changed from 16 to 32
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

class ImprovedAudioClassifier(nn.Module):
    def __init__(self, n_classes, n_mels=92):  # Changed default from 128 to 92
        super(ImprovedAudioClassifier, self).__init__()
        
        # Using parameters
        initial_filters = 64  # Changed from 32 to 64
        filter_multiplier = 2
        dropout_rate = 0.2451735125777227  # Changed from 0.5
        fc_size = 1024  # Changed from 512
        
        # Premier bloc de traitement
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, initial_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Blocs résiduels with filter multiplier
        f1 = initial_filters
        f2 = int(f1 * filter_multiplier)
        f3 = int(f2 * filter_multiplier)
        f4 = int(f3 * filter_multiplier)
        
        self.layer2 = ResidualBlock(f1, f2, stride=2)
        self.attention1 = ChannelAttention(f2)
        
        self.layer3 = ResidualBlock(f2, f3, stride=2)
        self.attention2 = ChannelAttention(f3)
        
        self.layer4 = ResidualBlock(f3, f4, stride=2)
        self.attention3 = ChannelAttention(f4)
        
        # Classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(f4, fc_size)
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

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'bird-audio-classifier-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dictionary with bird descriptions
BIRD_INFO = {
    'barswa': {
        'name': 'Bar-tailed Swallow',
        'description': 'A small swallow with a distinctive barred tail.'
    },
    'comsan': {
        'name': 'Common Sandpiper',
        'description': 'A small wading bird found near freshwater and coastal areas.'
    },
    'eaywag1': {
        'name': 'Eastern Yellow Wagtail',
        'description': 'A small passerine bird with distinctive yellow underparts.'
    },
    'thrnig1': {
        'name': 'Thrush Nightingale',
        'description': 'A small passerine bird known for its beautiful song.'
    },
    'wlwwar': {
        'name': 'Willow Warbler',
        'description': 'A common and widespread leaf warbler with greenish upperparts.'
    },
    'woosan': {
        'name': 'Wood Sandpiper',
        'description': 'A small wader with long legs, found in marshy wetlands.'
    }
}

# Initialize the classifier when app starts
classifier = None

def init_classifier():
    global classifier
    from model import AudioClassifier
    classifier = AudioClassifier()

@app.route('/')
def home():
    """Render the home page"""
    return render_template('home.html')

@app.route('/analyze')
def analyze():
    """Render the analysis page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if file:
        # Generate unique filename
        extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{extension}"
        filename = secure_filename(unique_filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file
        file.save(file_path)
        
        # Classify the audio
        result = classifier.classify(file_path)
        
        # Add bird info to predictions
        if result['success']:
            for pred in result['predictions']:
                bird_code = pred['class']
                if bird_code in BIRD_INFO:
                    pred['info'] = BIRD_INFO[bird_code]
                    pred['image'] = f"/static/bird_images/{bird_code}.jpg"
                else:
                    pred['info'] = {
                        'name': bird_code,
                        'description': 'No description available'
                    }
                    pred['image'] = "/static/bird_images/unknown.jpg"
            
            # Add audio file path to result
            result['audio_path'] = os.path.join('/static/uploads', filename)
            
        return jsonify(result)
    
    

if __name__ == '__main__':
    init_classifier()
    app.run(debug=True)
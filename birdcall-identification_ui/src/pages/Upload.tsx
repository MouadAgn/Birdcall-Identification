import React, { useState } from 'react';
import {
  Container,
  Box,
  Typography,
  Paper,
  Button,
  CircularProgress,
  Alert,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import AudioFileIcon from '@mui/icons-material/AudioFile';

const Upload = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Veuillez sélectionner un fichier');
      return;
    }

    setIsUploading(true);
    setError(null);
    setUploadSuccess(false);

    try {
      // Simulation d'appel API
      await new Promise(resolve => setTimeout(resolve, 2000));
      setUploadSuccess(true);
    } catch (err) {
      setError('Échec du téléchargement. Veuillez réessayer.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ mt: 8, mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Télécharger un Enregistrement d'Oiseau
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" align="center" paragraph>
          Téléchargez votre enregistrement de chant d'oiseau pour identifier l'espèce. Nous supportons différents formats audio.
        </Typography>

        <Paper
          elevation={3}
          sx={{
            p: 4,
            mt: 4,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
        >
          <input
            accept="audio/*"
            style={{ display: 'none' }}
            id="raised-button-file"
            type="file"
            onChange={handleFileChange}
          />
          <label htmlFor="raised-button-file">
            <Button
              variant="contained"
              component="span"
              startIcon={<CloudUploadIcon />}
              sx={{ mb: 2 }}
            >
              Sélectionner un Fichier
            </Button>
          </label>

          {file && (
            <Box sx={{ mt: 2, display: 'flex', alignItems: 'center' }}>
              <AudioFileIcon sx={{ mr: 1 }} />
              <Typography variant="body1">{file.name}</Typography>
            </Box>
          )}

          {error && (
            <Alert severity="error" sx={{ mt: 2, width: '100%' }}>
              {error}
            </Alert>
          )}

          {uploadSuccess && (
            <Alert severity="success" sx={{ mt: 2, width: '100%' }}>
              Fichier téléchargé avec succès ! Analyse en cours...
            </Alert>
          )}

          <Button
            variant="contained"
            color="primary"
            onClick={handleUpload}
            disabled={isUploading || !file}
            sx={{ mt: 3 }}
          >
            {isUploading ? (
              <>
                <CircularProgress size={24} sx={{ mr: 1 }} />
                Téléchargement...
              </>
            ) : (
              'Télécharger et Analyser'
            )}
          </Button>
        </Paper>

        <Box sx={{ 
          mt: 4, 
          display: 'flex', 
          flexDirection: { xs: 'column', md: 'row' },
          gap: 4
        }}>
          <Box sx={{ flex: 1 }}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Formats Supportés
              </Typography>
              <Typography variant="body2" color="text.secondary">
                • WAV
                <br />
                • MP3
                <br />
                • FLAC
                <br />
                • OGG
              </Typography>
            </Paper>
          </Box>
          <Box sx={{ flex: 1 }}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Conseils pour de Meilleurs Résultats
              </Typography>
              <Typography variant="body2" color="text.secondary">
                • Enregistrez dans un environnement calme
                <br />
                • Gardez une durée d'enregistrement entre 5-30 secondes
                <br />
                • Assurez-vous d'avoir une bonne qualité de microphone
                <br />
                • Évitez les bruits de fond
              </Typography>
            </Paper>
          </Box>
        </Box>
      </Box>
    </Container>
  );
};

export default Upload; 
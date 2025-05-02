import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Container,
  Box,
  Typography,
  Paper,
  Button,
  CircularProgress,
  Alert,
  IconButton,
  Fade,
  LinearProgress
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import AudioFileIcon from '@mui/icons-material/AudioFile';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import RefreshIcon from '@mui/icons-material/Refresh';
import WaveSurfer from 'wavesurfer.js';

const Upload = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isWaveformReady, setIsWaveformReady] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisResult, setAnalysisResult] = useState<string | null>(null);

  // Nettoyage sécurisé de WaveSurfer
  const destroyWaveSurfer = useCallback(() => {
    try {
      if (wavesurferRef.current) {
        wavesurferRef.current.pause();
        wavesurferRef.current.destroy();
        wavesurferRef.current = null;
      }
    } catch (err) {
      console.error('Erreur lors du nettoyage de WaveSurfer:', err);
    }
  }, []);

  // Initialisation sécurisée de WaveSurfer
  const initWaveSurfer = useCallback((url: string) => {
    try {
      if (!waveformRef.current) return;

      destroyWaveSurfer();

      const wavesurfer = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: isAnalyzing ? '#bdbdbd' : '#88A47C',
        progressColor: isAnalyzing ? '#bdbdbd' : '#4a90e2',
        cursorColor: '#2c5282',
        barWidth: 3,
        barGap: 2,
        barRadius: 3,
        height: 150,
        normalize: true,
        fillParent: true,
        autoCenter: true,
        minPxPerSec: 100,
        interact: !isAnalyzing,
        hideScrollbar: false,
      });

      wavesurfer.on('ready', () => {
        setIsWaveformReady(true);
        setIsPlaying(false);
      });

      wavesurfer.on('play', () => setIsPlaying(true));
      wavesurfer.on('pause', () => setIsPlaying(false));
      wavesurfer.on('finish', () => setIsPlaying(false));
      wavesurfer.on('error', (err) => {
        console.error('Erreur WaveSurfer:', err);
        setError('Erreur lors du chargement de l\'audio');
      });

      wavesurfer.load(url);
      wavesurferRef.current = wavesurfer;
    } catch (err) {
      console.error('Erreur lors de l\'initialisation de WaveSurfer:', err);
      setError('Erreur lors de l\'initialisation de la visualisation audio');
    }
  }, [isAnalyzing, destroyWaveSurfer]);

  // Gestion du changement de fichier
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const selectedFile = event.target.files[0];
      
      // Nettoyage de l'URL précédente
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }

      const newAudioUrl = URL.createObjectURL(selectedFile);
      
      setFile(selectedFile);
      setError(null);
      setUploadSuccess(false);
      setIsAnalyzing(false);
      setIsWaveformReady(false);
      setAudioUrl(newAudioUrl);
    }
  };

  // Gestion de la lecture/pause
  const handlePlayPause = () => {
    if (!wavesurferRef.current || !isWaveformReady || isAnalyzing) return;
    
    try {
      if (isPlaying) {
        wavesurferRef.current.pause();
      } else {
        wavesurferRef.current.play();
      }
    } catch (err) {
      console.error('Erreur lors de la lecture:', err);
      setError('Erreur lors de la lecture audio');
    }
  };

  const simulateAnalysis = useCallback(async () => {
    const totalSteps = 5;
    const stepTime = 400;

    for (let i = 0; i < totalSteps; i++) {
      setAnalysisProgress((i / totalSteps) * 100);
      await new Promise(resolve => setTimeout(resolve, stepTime));
    }

    return "Merle noir (Turdus merula) - Confiance: 92%";
  }, []);

  const handleUpload = async () => {
    if (!file) {
      setError('Veuillez sélectionner un fichier');
      return;
    }

    try {
      setIsUploading(true);
      setError(null);
      setUploadSuccess(false);
      setIsAnalyzing(true);
      setAnalysisResult(null);
      
      if (wavesurferRef.current && isPlaying) {
        wavesurferRef.current.pause();
      }

      // Simulation de l'analyse
      const result = await simulateAnalysis();
      setAnalysisResult(result);
      setUploadSuccess(true);
      
      // Réinitialisation de la visualisation
      if (audioUrl) {
        setTimeout(() => {
          setIsAnalyzing(false);
          initWaveSurfer(audioUrl);
        }, 500);
      }
    } catch (err) {
      console.error('Erreur lors de l\'analyse:', err);
      setError('Échec de l\'analyse. Veuillez réessayer.');
      setIsAnalyzing(false);
    } finally {
      setIsUploading(false);
      setAnalysisProgress(0);
    }
  };

  const handleReset = useCallback(() => {
    if (wavesurferRef.current && isPlaying) {
      wavesurferRef.current.pause();
    }

    setIsUploading(false);
    setUploadSuccess(false);
    setError(null);
    setIsPlaying(false);
    setIsAnalyzing(false);
    setAnalysisProgress(0);
    setAnalysisResult(null);

    if (audioUrl) {
      initWaveSurfer(audioUrl);
    }
  }, [audioUrl, isPlaying, initWaveSurfer]);

  useEffect(() => {
    if (audioUrl && !isAnalyzing) {
      initWaveSurfer(audioUrl);
    }
    return () => {
      destroyWaveSurfer();
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl, isAnalyzing, initWaveSurfer, destroyWaveSurfer]);

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
            <Box sx={{ mt: 2, width: '100%' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <AudioFileIcon sx={{ mr: 1 }} />
                <Typography variant="body1">{file.name}</Typography>
              </Box>
              <Paper
                elevation={2}
                sx={{
                  p: 3,
                  backgroundColor: '#f5f5f5',
                  borderRadius: 2,
                  transition: 'all 0.3s ease',
                  minHeight: 200
                }}
              >
                {isAnalyzing && (
                  <Fade in={isAnalyzing} unmountOnExit>
                    <Box sx={{ width: '100%', mb: 3 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <CircularProgress size={20} sx={{ mr: 2 }} />
                        <Typography variant="body2">
                          Analyse en cours... {Math.round(analysisProgress)}%
                        </Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={analysisProgress}
                        sx={{ 
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: 'rgba(0,0,0,0.1)',
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: '#4a90e2'
                          }
                        }}
                      />
                    </Box>
                  </Fade>
                )}
                <div ref={waveformRef} style={{ width: '100%', minHeight: 150, opacity: isAnalyzing ? 0.3 : 1, transition: 'opacity 0.3s' }} />
                <Box sx={{
                  display: 'flex',
                  justifyContent: 'center',
                  mt: 2,
                  opacity: isWaveformReady && !isAnalyzing ? 1 : 0.5,
                  transition: 'opacity 0.3s ease'
                }}>
                  <IconButton
                    onClick={handlePlayPause}
                    color="primary"
                    size="large"
                    disabled={!isWaveformReady || isAnalyzing}
                    sx={{
                      backgroundColor: 'rgba(74, 144, 226, 0.1)',
                      '&:hover': {
                        backgroundColor: 'rgba(74, 144, 226, 0.2)',
                      },
                      transition: 'all 0.3s ease'
                    }}
                  >
                    {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                  </IconButton>
                </Box>
              </Paper>

              {analysisResult && (
                <Fade in={!!analysisResult}>
                  <Paper elevation={2} sx={{ 
                    mt: 2, 
                    p: 2, 
                    backgroundColor: '#e8f5e9',
                    borderRadius: 2
                  }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="h6" color="primary">
                        Résultat de l'analyse
                      </Typography>
                      <IconButton
                        onClick={handleReset}
                        color="primary"
                        size="small"
                        sx={{
                          backgroundColor: 'rgba(74, 144, 226, 0.1)',
                          '&:hover': {
                            backgroundColor: 'rgba(74, 144, 226, 0.2)',
                          }
                        }}
                      >
                        <RefreshIcon />
                      </IconButton>
                    </Box>
                    <Typography variant="body1">
                      {analysisResult}
                    </Typography>
                  </Paper>
                </Fade>
              )}
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
          <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleUpload}
              disabled={isUploading || !file || isAnalyzing}
            >
              {isUploading || isAnalyzing ? (
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <CircularProgress size={24} sx={{ mr: 1 }} color="inherit" />
                  Analyse en cours...
                </Box>
              ) : (
                'Télécharger et Analyser'
              )}
            </Button>

            {(uploadSuccess || analysisResult) && (
              <Button
                variant="outlined"
                color="primary"
                onClick={handleReset}
                startIcon={<RefreshIcon />}
                disabled={isAnalyzing}
              >
                Réinitialiser
              </Button>
            )}
          </Box>
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
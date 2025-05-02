import React from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Button, 

  Card,
  CardContent,
  CardMedia
} from '@mui/material';
import UploadIcon from '@mui/icons-material/Upload';
import AnalyticsIcon from '@mui/icons-material/Analytics';

const Home = () => {
  return (
    <Container maxWidth="lg">
      {/* Section d'en-tête */}
      <Box
        sx={{
          pt: 8,
          pb: 6,
          textAlign: 'center',
        }}
      >
        <Typography
          component="h1"
          variant="h2"
          color="primary"
          gutterBottom
          sx={{ fontWeight: 'bold' }}
        >
          Identification des Chants d'Oiseaux
        </Typography>
        <Typography variant="h5" color="text.secondary" paragraph>
          Identifiez les espèces d'oiseaux grâce à leurs chants en utilisant l'intelligence artificielle.
          Aidez les scientifiques à automatiser la surveillance des populations d'oiseaux.
        </Typography>
        <Box sx={{ mt: 4 }}>
          <Button
            variant="contained"
            size="large"
            startIcon={<UploadIcon />}
            href="/upload"
            sx={{ mr: 2 }}
          >
            Télécharger un Enregistrement
          </Button>
          <Button
            variant="outlined"
            size="large"
            startIcon={<AnalyticsIcon />}
            href="/about"
          >
            En Savoir Plus
          </Button>
        </Box>
      </Box>

      {/* Section des fonctionnalités */}
      <Box sx={{ 
        mt: 4,
        mb: 8,
        display: 'flex',
        flexDirection: { xs: 'column', md: 'row' },
        gap: 4
      }}>
        <Box sx={{ flex: 1 }}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardMedia
              component="img"
              height="200"
              image="https://images.unsplash.com/photo-1452570053594-1b985d6ea890?auto=format&fit=crop&w=800&q=80"
              alt="Enregistrement d'Oiseau"
              sx={{
                objectFit: 'cover',
                objectPosition: 'center',
              }}
            />
            <CardContent sx={{ flexGrow: 1 }}>
              <Typography gutterBottom variant="h5" component="h2">
                Téléchargez vos Enregistrements
              </Typography>
              <Typography>
                Téléchargez vos enregistrements de chants d'oiseaux et laissez notre IA identifier les espèces.
                Compatible avec différents formats audio et longues durées d'enregistrement.
              </Typography>
            </CardContent>
          </Card>
        </Box>

        <Box sx={{ flex: 1 }}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardMedia
              component="img"
              height="200"
              image="https://images.unsplash.com/photo-1620712943543-bcc4688e7485?auto=format&fit=crop&w=800&q=80"
              alt="Analyse IA"
              sx={{
                objectFit: 'cover',
                objectPosition: 'center',
              }}
            />
            <CardContent sx={{ flexGrow: 1 }}>
              <Typography gutterBottom variant="h5" component="h2">
                Analyse par IA
              </Typography>
              <Typography>
                Notre modèle d'apprentissage automatique avancé analyse les enregistrements
                pour identifier les espèces d'oiseaux avec une grande précision.
              </Typography>
            </CardContent>
          </Card>
        </Box>

        <Box sx={{ flex: 1 }}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardMedia
              component="img"
              height="200"
              image="https://images.unsplash.com/photo-1490199444786-9d1faf6fbeb8?auto=format&fit=crop&w=800&q=80"
              alt="Conservation"
              sx={{
                objectFit: 'cover',
                objectPosition: 'center',
              }}
            />
            <CardContent sx={{ flexGrow: 1 }}>
              <Typography gutterBottom variant="h5" component="h2">
                Soutien à la Conservation
              </Typography>
              <Typography>
                Contribuez à la surveillance des populations d'oiseaux et aux efforts de conservation
                en aidant les scientifiques à suivre la distribution des espèces.
              </Typography>
            </CardContent>
          </Card>
        </Box>
      </Box>
    </Container>
  );
};

export default Home; 
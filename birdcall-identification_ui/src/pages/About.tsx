import React from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import ScienceIcon from '@mui/icons-material/Science';
import PsychologyIcon from '@mui/icons-material/Psychology';
import NatureIcon from '@mui/icons-material/Nature';
import TimelineIcon from '@mui/icons-material/Timeline';

const About = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ mt: 8, mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          About BirdCall Identification
        </Typography>
        <Typography variant="h6" color="text.secondary" align="center" paragraph>
          Helping scientists automate the remote monitoring of bird populations
        </Typography>

        <Box sx={{ 
          mt: 4,
          display: 'flex',
          flexDirection: { xs: 'column', md: 'row' },
          gap: 4
        }}>
          <Box sx={{ flex: 1 }}>
            <Paper sx={{ p: 4, height: '100%' }}>
              <Typography variant="h5" gutterBottom>
                Our Mission
              </Typography>
              <Typography paragraph>
                BirdCall Identification is a cutting-edge platform that uses artificial intelligence
                to identify bird species through their calls. Our goal is to support scientists and
                conservationists in their efforts to monitor and protect bird populations worldwide.
              </Typography>
              <Typography paragraph>
                By automating the process of bird call identification, we help researchers:
              </Typography>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <TimelineIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText primary="Track bird population trends over time" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <NatureIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText primary="Monitor species distribution and migration patterns" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <ScienceIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText primary="Gather data for conservation research" />
                </ListItem>
              </List>
            </Paper>
          </Box>

          <Box sx={{ flex: 1 }}>
            <Paper sx={{ p: 4, height: '100%' }}>
              <Typography variant="h5" gutterBottom>
                How It Works
              </Typography>
              <Typography paragraph>
                Our platform uses advanced machine learning algorithms to analyze bird calls:
              </Typography>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <PsychologyIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="AI-Powered Analysis"
                    secondary="Our deep learning model processes audio recordings to identify unique bird call patterns"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <ScienceIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Scientific Validation"
                    secondary="Results are cross-referenced with extensive bird call databases"
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <NatureIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary="Conservation Impact"
                    secondary="Data contributes to global bird population monitoring efforts"
                  />
                </ListItem>
              </List>
            </Paper>
          </Box>
        </Box>

        <Paper sx={{ p: 4, mt: 4 }}>
          <Typography variant="h5" gutterBottom>
            Technical Details
          </Typography>
          <Typography paragraph>
            Our system is built on state-of-the-art machine learning models trained on thousands of
            bird call recordings from diverse environments. The model can identify multiple species
            in a single recording and works with various audio formats.
          </Typography>
          <Typography paragraph>
            Key features:
          </Typography>
          <List>
            <ListItem>
              <ListItemText
                primary="Multi-species Detection"
                secondary="Identify multiple bird species in a single recording"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Noise Filtering"
                secondary="Advanced algorithms to filter out background noise"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Real-time Processing"
                secondary="Quick analysis of uploaded recordings"
              />
            </ListItem>
          </List>
        </Paper>
      </Box>
    </Container>
  );
};

export default About; 
import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import AudiotrackIcon from '@mui/icons-material/Audiotrack';

const Navbar = () => {
  return (
    <AppBar position="static" elevation={0}>
      <Toolbar>
        <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
          <AudiotrackIcon sx={{ mr: 2 }} />
          <Typography variant="h6" component={RouterLink} to="/" sx={{ 
            textDecoration: 'none', 
            color: 'inherit',
            '&:hover': {
              color: 'inherit',
            }
          }}>
            BirdCall Identification
          </Typography>
        </Box>
        <Box>
          <Button color="inherit" component={RouterLink} to="/">
            Home
          </Button>
          <Button color="inherit" component={RouterLink} to="/upload">
            Upload
          </Button>
          <Button color="inherit" component={RouterLink} to="/about">
            About
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar; 
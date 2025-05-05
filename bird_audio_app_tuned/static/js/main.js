// Format time in seconds to MM:SS format
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
}

// Add audio visualization with Web Audio API
function setupAudioVisualization() {
    const audioElement = document.getElementById('audio-player');
    if (!audioElement) return;
    
    // Create audio context when play starts
    audioElement.addEventListener('play', function() {
        if (!window.audioContext) {
            window.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = window.audioContext.createMediaElementSource(audioElement);
            
            // Create analyzer
            const analyser = window.audioContext.createAnalyser();
            analyser.fftSize = 256;
            
            // Connect source to analyzer and then to destination
            source.connect(analyser);
            analyser.connect(window.audioContext.destination);
            
            // Initialize visualization
            const canvas = document.getElementById('audio-visualizer');
            if (canvas) {
                const canvasCtx = canvas.getContext('2d');
                const bufferLength = analyser.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);
                
                function draw() {
                    requestAnimationFrame(draw);
                    
                    analyser.getByteFrequencyData(dataArray);
                    
                    canvasCtx.fillStyle = 'rgb(34, 34, 34)';
                    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    const barWidth = (canvas.width / bufferLength) * 2.5;
                    let x = 0;
                    
                    for (let i = 0; i < bufferLength; i++) {
                        const barHeight = dataArray[i] / 2;
                        
                        canvasCtx.fillStyle = `rgb(${barHeight + 100}, 50, 50)`;
                        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                        
                        x += barWidth + 1;
                    }
                }
                
                draw();
            }
        }
    });
}

// Handle page load
document.addEventListener('DOMContentLoaded', function() {
    // Setup audio visualization if canvas is present
    if (document.getElementById('audio-visualizer')) {
        setupAudioVisualization();
    }
    
    // Add tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Handle copy buttons (if any)
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        console.log('Text copied to clipboard');
    }, function(err) {
        console.error('Could not copy text: ', err);
    });
}
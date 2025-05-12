document.addEventListener('DOMContentLoaded', function() {
    const startWebcamBtn = document.getElementById('startWebcam');
    const stopWebcamBtn = document.getElementById('stopWebcam');
    const webcamElement = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const webcamResult = document.getElementById('webcamResult');
    
    let stream = null;
    let isWebcamActive = false;
    let lastPredictionTime = 0;
    const MIN_PREDICTION_INTERVAL = 200; // 5 FPS

    // Handle webcam
    startWebcamBtn.addEventListener('click', async function() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcamElement.srcObject = stream;
            webcamElement.style.display = 'block';
            startWebcamBtn.style.display = 'none';
            stopWebcamBtn.style.display = 'block';
            isWebcamActive = true;
            processWebcam();
        } catch (error) {
            console.error('Error:', error);
            webcamResult.innerHTML = 'Error accessing webcam. Please check your camera.';
        }
    });

    stopWebcamBtn.addEventListener('click', function() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcamElement.style.display = 'none';
            startWebcamBtn.style.display = 'block';
            stopWebcamBtn.style.display = 'none';
            isWebcamActive = false;
            webcamResult.innerHTML = '';
        }
    });

    async function processWebcam() {
        if (!isWebcamActive) return;

        const currentTime = Date.now();
        if (currentTime - lastPredictionTime < MIN_PREDICTION_INTERVAL) {
            setTimeout(processWebcam, MIN_PREDICTION_INTERVAL - (currentTime - lastPredictionTime));
            return;
        }

        canvas.width = webcamElement.videoWidth;
        canvas.height = webcamElement.videoHeight;
        canvas.getContext('2d').drawImage(webcamElement, 0, 0);
        
        const imageData = canvas.toDataURL('image/jpeg');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ frame: imageData })
            });
            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();
            lastPredictionTime = currentTime;
            
            if (data.confidence > 40.0) {
                webcamResult.innerHTML = `
                    Prediction: ${data.prediction}<br>
                    Confidence: ${data.confidence.toFixed(1)}%
                `;
            } else {
                webcamResult.innerHTML = 'Low confidence';
            }
        } catch (error) {
            console.error('Error:', error);
            webcamResult.innerHTML = 'Error processing webcam frame. Please try again.';
        }

        if (isWebcamActive) {
            setTimeout(processWebcam, MIN_PREDICTION_INTERVAL);
        }
    }
});
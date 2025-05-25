// Door Configuration Debugger
console.log('Door configuration debugger loaded');

document.addEventListener('DOMContentLoaded', () => {
    // Check for elements
    const videoFeed = document.getElementById('video-feed');
    const saveDoorBtn = document.getElementById('save-door');
    const doorStatus = document.getElementById('door-status');
    const x1Input = document.getElementById('x1');
    const y1Input = document.getElementById('y1');
    const x2Input = document.getElementById('x2');
    const y2Input = document.getElementById('y2');
    
    console.log('Video feed element:', videoFeed ? 'Found' : 'Missing');
    console.log('Save door button:', saveDoorBtn ? 'Found' : 'Missing');
    console.log('Door status element:', doorStatus ? 'Found' : 'Missing');
    console.log('Coordinate inputs:', 
        x1Input ? 'X1 Found' : 'X1 Missing',
        y1Input ? 'Y1 Found' : 'Y1 Missing',
        x2Input ? 'X2 Found' : 'X2 Missing',
        y2Input ? 'Y2 Found' : 'Y2 Missing'
    );
    
    // Override save function to log data
    if (saveDoorBtn) {
        saveDoorBtn.addEventListener('click', function() {
            const data = {
                x1: parseInt(x1Input?.value || '0'),
                y1: parseInt(y1Input?.value || '0'),
                x2: parseInt(x2Input?.value || '0'),
                y2: parseInt(y2Input?.value || '0'),
                inside_direction: document.getElementById('inside-direction')?.value || 'right'
            };
            
            console.log('Saving door data:', data);
            
            // Make the API call
            fetch('/api/door-area', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => {
                console.log('Response status:', response.status);
                return response.json();
            })
            .then(result => {
                console.log('API response:', result);
                if (doorStatus) {
                    doorStatus.textContent = result.success ? result.message : result.message || 'Error saving door area';
                    doorStatus.className = `mt-2 ${result.success ? 'success' : 'error'}`;
                }
            })
            .catch(error => {
                console.error('Error saving door area:', error);
                if (doorStatus) {
                    doorStatus.textContent = 'Error: ' + error.message;
                    doorStatus.className = 'mt-2 error';
                }
            });
        });
    }
});

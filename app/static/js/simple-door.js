// Simple Door Area Selection
document.addEventListener('DOMContentLoaded', function() {
    console.log('Simple door selector loaded');
    
    // Get elements
    const videoFeed = document.getElementById('video-feed');
    const selectionBox = document.getElementById('selection-box');
    const x1Input = document.getElementById('x1');
    const y1Input = document.getElementById('y1');
    const x2Input = document.getElementById('x2');
    const y2Input = document.getElementById('y2');
    const saveBtn = document.getElementById('save-door');
    const statusElem = document.getElementById('door-status');
    
    if (!videoFeed || !selectionBox) {
        console.error('Video feed or selection box not found');
        return;
    }
    
    // Simple state
    let isDrawing = false;
    let startX = 0;
    let startY = 0;
    
    // Log elements
    console.log('Video feed found:', videoFeed);
    console.log('Selection box found:', selectionBox);
    console.log('Input fields:', x1Input, y1Input, x2Input, y2Input);
    
    // Simple drawing
    videoFeed.onmousedown = function(e) {
        console.log('Mouse down', e);
        const rect = videoFeed.getBoundingClientRect();
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;
        isDrawing = true;
        
        // Set initial position
        selectionBox.style.display = 'block';
        selectionBox.style.left = startX + 'px';
        selectionBox.style.top = startY + 'px';
        selectionBox.style.width = '0px';
        selectionBox.style.height = '0px';
        
        // Set inputs        if (x1Input) x1Input.value = Math.round(startX);
        if (y1Input) y1Input.value = Math.round(startY);
        
        if (statusElem) statusElem.textContent = 'Menggambar area pintu...';
    };
    
    videoFeed.onmousemove = function(e) {
        if (!isDrawing) return;
        console.log('Mouse move while drawing');
        
        const rect = videoFeed.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;
        
        // Update the width and height
        const width = currentX - startX;
        const height = currentY - startY;
        
        // Update the display box
        if (width >= 0) {
            selectionBox.style.width = width + 'px';
            if (x2Input) x2Input.value = Math.round(currentX);
        } else {
            selectionBox.style.left = currentX + 'px';
            selectionBox.style.width = Math.abs(width) + 'px';
            if (x1Input) x1Input.value = Math.round(currentX);
            if (x2Input) x2Input.value = Math.round(startX);
        }
        
        if (height >= 0) {
            selectionBox.style.height = height + 'px';
            if (y2Input) y2Input.value = Math.round(currentY);
        } else {
            selectionBox.style.top = currentY + 'px';
            selectionBox.style.height = Math.abs(height) + 'px';
            if (y1Input) y1Input.value = Math.round(currentY);
            if (y2Input) y2Input.value = Math.round(startY);
        }
    };
    
    videoFeed.onmouseup = function(e) {
        console.log('Mouse up');
        if (!isDrawing) return;
        isDrawing = false;
        
        // Check if area is valid
        const x1 = parseInt(x1Input.value);
        const y1 = parseInt(y1Input.value);
        const x2 = parseInt(x2Input.value);
        const y2 = parseInt(y2Input.value);
          if (Math.abs(x2-x1) < 10 || Math.abs(y2-y1) < 10) {
            if (statusElem) statusElem.textContent = 'Area pintu terlalu kecil, silakan coba lagi';
            return;
        }
        
        if (statusElem) statusElem.textContent = 'Area pintu telah ditentukan. Klik Simpan Area Pintu untuk menyimpan.';
    };
    
    // Same for mouse leave
    videoFeed.onmouseleave = videoFeed.onmouseup;
    
    // When save button is clicked
    if (saveBtn) {
        saveBtn.onclick = function() {
            const x1 = parseInt(x1Input.value);
            const y1 = parseInt(y1Input.value);
            const x2 = parseInt(x2Input.value);
            const y2 = parseInt(y2Input.value);
              if (isNaN(x1) || isNaN(y1) || isNaN(x2) || isNaN(y2)) {
                if (statusElem) statusElem.textContent = 'Silakan tentukan area pintu terlebih dahulu';
                return;
            }
            
            // Create the data to send
            const data = {
                x1: x1,
                y1: y1,
                x2: x2,
                y2: y2,
                inside_direction: document.getElementById('inside-direction')?.value || 'right'
            };
            
            console.log('Saving door data:', data);
            
            // Send the data to the server
            fetch('/api/door-area', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(result => {
                console.log('Save result:', result);                if (statusElem) {
                    statusElem.textContent = result.message || 'Area pintu berhasil disimpan';
                }
            })
            .catch(error => {
                console.error('Error saving door area:', error);
                if (statusElem) {
                    statusElem.textContent = 'Gagal menyimpan area pintu: ' + error.message;
                }
            });
        };
    }
});

class DoorAreaConfig {
    constructor() {
        this.isSelecting = false;
        this.startX = null;
        this.startY = null;
        this.setupElements();
        this.setupEventListeners();
        this.loadExistingConfig();
        
        // Add door orientation detection
        this.checkDoorOrientation = this.checkDoorOrientation.bind(this);
    }

    setupElements() {
        this.videoFeed = document.getElementById('video-feed');
        this.selectionBox = document.getElementById('selection-box');
        this.x1Input = document.getElementById('x1');
        this.y1Input = document.getElementById('y1');
        this.x2Input = document.getElementById('x2');
        this.y2Input = document.getElementById('y2');
        this.saveDoorBtn = document.getElementById('save-door');
        this.doorStatus = document.getElementById('door-status');
        this.insideDirection = document.getElementById('inside-direction');
        
        // Log elements to help debugging
        console.log("Elements found:");
        console.log("Video feed:", this.videoFeed);
        console.log("Selection box:", this.selectionBox);
        console.log("Input fields:", this.x1Input, this.y1Input, this.x2Input, this.y2Input);
        console.log("Save button:", this.saveDoorBtn);
    }

    setupEventListeners() {
        if (!this.videoFeed) {
            console.error("Video feed element not found!");
            return;
        }
        
        if (!this.selectionBox) {
            console.error("Selection box element not found!");
            return;
        }        // Use direct references to 'this' in event handlers
        const self = this;
        
        this.videoFeed.onmousedown = function(e) {
            // Prevent default browser behavior (drag, selection)
            e.preventDefault();
            
            console.log("Mouse down event");
            const rect = self.videoFeed.getBoundingClientRect();
            self.startX = e.clientX - rect.left;
            self.startY = e.clientY - rect.top;
            self.isSelecting = true;

            // Initialize selection box
            self.selectionBox.style.display = 'block';
            self.selectionBox.style.left = self.startX + 'px';
            self.selectionBox.style.top = self.startY + 'px';
            self.selectionBox.style.width = '0px';
            self.selectionBox.style.height = '0px';

            self.x1Input.value = Math.round(self.startX);
            self.y1Input.value = Math.round(self.startY);
        };

        this.videoFeed.onmousemove = function(e) {
            if (!self.isSelecting) return;
            
            console.log("Mouse move while selecting");
            const rect = self.videoFeed.getBoundingClientRect();
            const currentX = e.clientX - rect.left;
            const currentY = e.clientY - rect.top;

            // Calculate dimensions
            const width = currentX - self.startX;
            const height = currentY - self.startY;

            // Update selection box
            if (width < 0) {
                self.selectionBox.style.left = currentX + 'px';
                self.selectionBox.style.width = Math.abs(width) + 'px';
                self.x1Input.value = Math.round(currentX);
                self.x2Input.value = Math.round(self.startX);
            } else {
                self.selectionBox.style.left = self.startX + 'px';
                self.selectionBox.style.width = width + 'px';
                self.x1Input.value = Math.round(self.startX);
                self.x2Input.value = Math.round(currentX);
            }

            if (height < 0) {
                self.selectionBox.style.top = currentY + 'px';
                self.selectionBox.style.height = Math.abs(height) + 'px';
                self.y1Input.value = Math.round(currentY);
                self.y2Input.value = Math.round(self.startY);
            } else {
                self.selectionBox.style.top = self.startY + 'px';
                self.selectionBox.style.height = height + 'px';
                self.y1Input.value = Math.round(self.startY);
                self.y2Input.value = Math.round(currentY);
            }
        };

        this.videoFeed.onmouseup = function(e) {
            if (!self.isSelecting) return;
            console.log("Mouse up event - ending selection");
            
            // Stop the selection process
            self.isSelecting = false;
            
            // Make sure we have valid coordinates
            const x1 = parseInt(self.x1Input.value);
            const y1 = parseInt(self.y1Input.value);
            const x2 = parseInt(self.x2Input.value);
            const y2 = parseInt(self.y2Input.value);
            
            console.log("Final coordinates:", x1, y1, x2, y2);            // Validate that we have a proper area
            if (Math.abs(x2 - x1) < 10 || Math.abs(y2 - y1) < 10) {
                self.updateStatus('Area pintu terlalu kecil. Silakan gambar area yang lebih besar.', 'error');
                return;
            }
            
            self.updateStatus('Area pintu telah ditentukan. Klik "Simpan Area Pintu" untuk menyimpan.', 'success');
        };

        // Also handle mouse leave
        this.videoFeed.onmouseleave = this.videoFeed.onmouseup;
        
        // Handle save button
        if (this.saveDoorBtn) {
            this.saveDoorBtn.onclick = function() {
                self.saveDoorArea();
            };
        }
    }

    loadExistingConfig() {
        fetch('/api/door-area')
            .then(response => response.json())
            .then(data => {
                if (data.door_defined) {
                    this.x1Input.value = data.x1;
                    this.y1Input.value = data.y1;
                    this.x2Input.value = data.x2;
                    this.y2Input.value = data.y2;
                    this.insideDirection.value = data.inside_direction;

                    // Display the door area
                    this.selectionBox.style.display = 'block';
                    this.selectionBox.style.left = data.x1 + 'px';
                    this.selectionBox.style.top = data.y1 + 'px';
                    this.selectionBox.style.width = (data.x2 - data.x1) + 'px';
                    this.selectionBox.style.height = (data.y2 - data.y1) + 'px';
                    
                    // Check door orientation and show relevant tip
                    this.checkDoorOrientation();
                }
            });
    }    checkDoorOrientation() {
        if (!this.x1Input.value || !this.y1Input.value || !this.x2Input.value || !this.y2Input.value) {
            return;
        }
        
        const width = Math.abs(parseInt(this.x2Input.value) - parseInt(this.x1Input.value));
        const height = Math.abs(parseInt(this.y2Input.value) - parseInt(this.y1Input.value));
        const orientationInfo = document.getElementById('door-orientation-info');
        
        if (height > width) {
            // Vertical door detected
            orientationInfo.style.display = 'flex';
            // Add direction arrows if they don't exist
            this.showDirectionIndicators('vertical');
        } else {
            // Horizontal door
            orientationInfo.style.display = 'none';
            this.showDirectionIndicators('horizontal');
        }
    }
    
    showDirectionIndicators(orientation) {
        // Remove any existing indicators
        const existingArrows = document.querySelectorAll('.direction-arrow');
        existingArrows.forEach(arrow => arrow.remove());
        
        if (!this.selectionBox || this.selectionBox.style.display === 'none') {
            return;
        }
        
        const x1 = parseInt(this.x1Input.value);
        const y1 = parseInt(this.y1Input.value);
        const x2 = parseInt(this.x2Input.value);
        const y2 = parseInt(this.y2Input.value);
        
        // Create container for arrows
        const arrowContainer = document.createElement('div');
        arrowContainer.className = 'direction-arrow';
        document.getElementById('video-wrapper').appendChild(arrowContainer);
        
        if (orientation === 'vertical') {
            // For vertical doors, place arrows on left and right sides
            const leftArrow = document.createElement('div');
            leftArrow.className = 'arrow left-arrow';
            leftArrow.innerHTML = '◀';
            leftArrow.style.top = ((y1 + y2) / 2) + 'px';
            leftArrow.style.left = (x1 - 30) + 'px';
            leftArrow.title = 'Arah kiri dianggap dalam ruangan jika dipilih "Sisi Kiri"';
            
            const rightArrow = document.createElement('div');
            rightArrow.className = 'arrow right-arrow';
            rightArrow.innerHTML = '▶';
            rightArrow.style.top = ((y1 + y2) / 2) + 'px';
            rightArrow.style.left = (x2 + 10) + 'px';
            rightArrow.title = 'Arah kanan dianggap dalam ruangan jika dipilih "Sisi Kanan"';
            
            arrowContainer.appendChild(leftArrow);
            arrowContainer.appendChild(rightArrow);
        } else {
            // For horizontal doors, place arrows on top and bottom
            const topArrow = document.createElement('div');
            topArrow.className = 'arrow top-arrow';
            topArrow.innerHTML = '▲';
            topArrow.style.top = (y1 - 30) + 'px';
            topArrow.style.left = ((x1 + x2) / 2) + 'px';
            topArrow.title = 'Arah atas dianggap dalam ruangan jika dipilih "Sisi Atas"';
            
            const bottomArrow = document.createElement('div');
            bottomArrow.className = 'arrow bottom-arrow';
            bottomArrow.innerHTML = '▼';
            bottomArrow.style.top = (y2 + 10) + 'px';
            bottomArrow.style.left = ((x1 + x2) / 2) + 'px';
            bottomArrow.title = 'Arah bawah dianggap dalam ruangan jika dipilih "Sisi Bawah"';
            
            arrowContainer.appendChild(topArrow);
            arrowContainer.appendChild(bottomArrow);
        }
    }
    
    saveDoorArea(callback) {
        const x1 = parseInt(this.x1Input.value);
        const y1 = parseInt(this.y1Input.value);
        const x2 = parseInt(this.x2Input.value);
        const y2 = parseInt(this.y2Input.value);

        if (isNaN(x1) || isNaN(y1) || isNaN(x2) || isNaN(y2)) {
            this.updateStatus('Silakan gambar area pintu terlebih dahulu', 'error');
            if (callback) callback(false, 'Silakan gambar area pintu terlebih dahulu');
            return;
        }

        const data = {
            x1, y1, x2, y2,
            inside_direction: this.insideDirection.value
        };

        fetch('/api/door-area', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    this.updateStatus(result.message, 'success');
                    
                    // Check if we have a modal function available from the parent page
                    if (typeof showSuccessDoorModal === 'function') {
                        // Use modal display
                        showSuccessDoorModal(data);
                    } else if (typeof showToast === 'function') {
                        // Use toast notification if available
                        showToast('✅ ' + result.message, 'success');
                    }
                    
                    if (callback) callback(true, result.message);
                } else {
                    this.updateStatus(result.message, 'error');
                    
                    // Use toast for error if available
                    if (typeof showToast === 'function') {
                        showToast('❌ ' + result.message, 'error');
                    }
                    
                    if (callback) callback(false, result.message);
                }
            })
            .catch(error => {
                this.updateStatus('Error: ' + error.message, 'error');
                
                // Use toast for error if available
                if (typeof showToast === 'function') {
                    showToast('❌ Error: ' + error.message, 'error');
                }
                
                if (callback) callback(false, error.message);
            });
    }

    updateStatus(message, type) {
        if (this.doorStatus) {
            this.doorStatus.textContent = message;
            this.doorStatus.className = `mt-2 ${type}`;
        }
    }
}

// Initialize door configuration when the page loads
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('video-feed')) {
        new DoorAreaConfig();
    }
});

// Create a new doorConfig instance when script is loaded
let doorAreaInstance = null;
document.addEventListener('DOMContentLoaded', function() {
    // Create a global doorAreaConfig instance if the video feed exists
    if (document.getElementById('video-feed')) {
        doorAreaInstance = new DoorAreaConfig();
    }
});
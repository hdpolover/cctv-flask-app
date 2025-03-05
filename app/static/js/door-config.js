class DoorAreaConfig {
    constructor() {
        this.isSelecting = false;
        this.startX = null;
        this.startY = null;
        this.setupElements();
        this.setupEventListeners();
        this.loadExistingConfig();
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
    }

    setupEventListeners() {
        if (!this.videoFeed) return;

        this.videoFeed.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.videoFeed.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.videoFeed.addEventListener('mouseup', this.handleMouseUp.bind(this));
        
        if (this.saveDoorBtn) {
            this.saveDoorBtn.addEventListener('click', this.saveDoorArea.bind(this));
        }
    }

    handleMouseDown(e) {
        const rect = this.videoFeed.getBoundingClientRect();
        this.startX = e.clientX - rect.left;
        this.startY = e.clientY - rect.top;
        this.isSelecting = true;

        // Initialize selection box
        this.selectionBox.style.display = 'block';
        this.selectionBox.style.left = this.startX + 'px';
        this.selectionBox.style.top = this.startY + 'px';
        this.selectionBox.style.width = '0px';
        this.selectionBox.style.height = '0px';

        this.x1Input.value = Math.round(this.startX);
        this.y1Input.value = Math.round(this.startY);
    }

    handleMouseMove(e) {
        if (!this.isSelecting) return;

        const rect = this.videoFeed.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;

        // Calculate dimensions
        const width = currentX - this.startX;
        const height = currentY - this.startY;

        // Update selection box
        if (width < 0) {
            this.selectionBox.style.left = currentX + 'px';
            this.selectionBox.style.width = Math.abs(width) + 'px';
            this.x1Input.value = Math.round(currentX);
            this.x2Input.value = Math.round(this.startX);
        } else {
            this.selectionBox.style.left = this.startX + 'px';
            this.selectionBox.style.width = width + 'px';
            this.x1Input.value = Math.round(this.startX);
            this.x2Input.value = Math.round(currentX);
        }

        if (height < 0) {
            this.selectionBox.style.top = currentY + 'px';
            this.selectionBox.style.height = Math.abs(height) + 'px';
            this.y1Input.value = Math.round(currentY);
            this.y2Input.value = Math.round(this.startY);
        } else {
            this.selectionBox.style.top = this.startY + 'px';
            this.selectionBox.style.height = height + 'px';
            this.y1Input.value = Math.round(this.startY);
            this.y2Input.value = Math.round(currentY);
        }
    }

    handleMouseUp() {
        this.isSelecting = false;
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
                }
            })
            .catch(error => {
                console.error('Error loading door configuration:', error);
            });
    }

    saveDoorArea() {
        const x1 = parseInt(this.x1Input.value);
        const y1 = parseInt(this.y1Input.value);
        const x2 = parseInt(this.x2Input.value);
        const y2 = parseInt(this.y2Input.value);

        if (isNaN(x1) || isNaN(y1) || isNaN(x2) || isNaN(y2)) {
            this.updateStatus('Please draw a door area first', 'error');
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
            } else {
                this.updateStatus(result.message, 'error');
            }
        })
        .catch(error => {
            this.updateStatus('Error: ' + error.message, 'error');
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
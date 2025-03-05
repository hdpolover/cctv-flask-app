class VideoFeed {
    constructor(options = {}) {
        this.socketUrl = options.socketUrl || window.location.origin;
        this.socket = io(this.socketUrl);
        this.counterElement = document.getElementById('people-count');
        this.setupSocketHandlers();
    }

    setupSocketHandlers() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
        });

        // Handle video frame updates
        this.socket.on('video_frame', (frameData) => {
            const videoFeed = document.getElementById('video-feed');
            if (videoFeed) {
                videoFeed.src = `data:image/jpeg;base64,${frameData}`;
            }
        });

        // Handle counter updates
        this.socket.on('counter_update', (data) => {
            if (this.counterElement) {
                this.counterElement.textContent = `People in the room: ${data.people_in_room}`;
            }
            
            // Update other counter elements if they exist
            const entriesElement = document.querySelector('.entries');
            const exitsElement = document.querySelector('.exits');
            
            if (entriesElement) {
                entriesElement.textContent = `Entries: ${data.entries}`;
            }
            if (exitsElement) {
                exitsElement.textContent = `Exits: ${data.exits}`;
            }
        });
    }

    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
        }
    }
}

// Initialize video feed when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const videoFeed = new VideoFeed();
    
    // Clean up when the page is unloaded
    window.addEventListener('beforeunload', () => {
        videoFeed.disconnect();
    });
});
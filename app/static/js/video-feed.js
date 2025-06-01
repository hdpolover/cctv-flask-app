class VideoFeed {
    constructor(options = {}) {
        this.socketUrl = options.socketUrl || window.location.origin;
        this.socket = io(this.socketUrl);
        this.counterElement = document.getElementById('people-count');
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.lastUpdateTime = Date.now();
        this.setupSocketHandlers();
        this.initializeUI();
    }

    initializeUI() {
        // Add loading state to video feed
        const videoFeed = document.getElementById('video-feed');
        if (videoFeed) {
            videoFeed.addEventListener('load', () => {
                this.removeVideoLoading();
            });

            videoFeed.addEventListener('error', () => {
                this.showVideoError();
            });
        }

        // Initialize connection status
        this.createConnectionStatusElement();

        // Add click to refresh functionality
        this.addRefreshButton();
    }

    setupSocketHandlers() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.updateConnectionStatus('connected', 'Terhubung - Update Real-time Aktif');
            this.showNotification('Koneksi berhasil!', 'success');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.isConnected = false;
            this.updateConnectionStatus('disconnected', 'Terputus - Mencoba menghubungkan kembali...');
            this.attemptReconnect();
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.updateConnectionStatus('error', 'Error koneksi - Periksa jaringan Anda');
        });

        // Handle video frame updates with error handling
        this.socket.on('video_frame', (frameData) => {
            try {
                const videoFeed = document.getElementById('video-feed');
                if (videoFeed && frameData) {
                    videoFeed.src = `data:image/jpeg;base64,${frameData}`;
                    this.lastUpdateTime = Date.now();
                    this.removeVideoError();
                }
            } catch (error) {
                console.error('Error updating video frame:', error);
                this.showVideoError();
            }
        });

        // Handle counter updates with enhanced animations
        this.socket.on('counter_update', (data) => {
            try {
                this.updateCounters(data);
                this.updatePageTitle(data.people_in_room);
                this.lastUpdateTime = Date.now();
            } catch (error) {
                console.error('Error updating counters:', error);
            }
        });

        // Handle system status updates
        this.socket.on('system_status', (status) => {
            this.updateSystemStatus(status);
        });
    }

    updateCounters(data) {
        // Update people count with animation
        if (this.counterElement) {
            const oldCount = parseInt(this.counterElement.textContent.match(/\d+/) || [0])[0];
            const newCount = data.people_in_room;

            if (oldCount !== newCount) {
                this.animateCounterChange(this.counterElement, oldCount, newCount);
            }
            // Subtle visual feedback without animation
            const counterDiv = this.counterElement.closest('.counter');
            if (counterDiv) {
                // No animation class adding/removing
            }
        }
        
        // Update entries and exits if available
        const entriesElement = document.querySelector('.entries span');
        const exitsElement = document.querySelector('.exits span');
        
        if (entriesElement && data.entries !== undefined) {
            entriesElement.textContent = data.entries;
        }
        
        if (exitsElement && data.exits !== undefined) {
            exitsElement.textContent = data.exits;
        }
        
        // Update video details if available
        this.updateVideoDetails(data);
    }
      updateVideoDetails(data) {
        // Update connection status
        const connectionStatusElement = document.getElementById('connection-status');
        if (connectionStatusElement) {
            if (this.isConnected) {
                connectionStatusElement.textContent = 'Terhubung';
                connectionStatusElement.classList.add('connected');
                connectionStatusElement.classList.remove('disconnected');
            } else {
                connectionStatusElement.textContent = 'Terputus';
                connectionStatusElement.classList.add('disconnected');
                connectionStatusElement.classList.remove('connected');
            }
        }
        // Update other details if they're provided in the data
        if (data.video_source) {
            const videoSourceElement = document.getElementById('video-source');
            if (videoSourceElement) {
                videoSourceElement.textContent = data.video_source;
            }
        }
        
        if (data.resolution) {
            const resolutionElement = document.getElementById('video-resolution');
            if (resolutionElement) {
                resolutionElement.textContent = data.resolution;
            }
        }
    }

    animateCounterChange(element, oldValue, newValue) {
        // Just update the text without animation
        element.textContent = `Orang di ruangan: ${newValue}`;
        element.style.color = 'var(--success-color)';
    }

    updateStatistics(data) {
        // Update any additional statistics elements
        const statsElements = {
            'total-detections': data.total_detections,
            'detection-rate': data.detection_rate,
            'avg-occupancy': data.avg_occupancy,
            'peak-occupancy': data.peak_occupancy
        };

        Object.entries(statsElements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element && value !== undefined) {
                element.textContent = value;
            }
        });
    }

    updatePageTitle(count) {
        const baseTitle = 'Beranda - CCTV Monitor';
        document.title = count !== undefined ?
            `${baseTitle} (${count} orang)` : baseTitle;
    }

    createConnectionStatusElement() {
        let statusIndicator = document.getElementById('connection-status');
        if (!statusIndicator) {
            statusIndicator = document.createElement('div');
            statusIndicator.id = 'connection-status';
            statusIndicator.className = 'connection-status';

            const container = document.querySelector('.home-container') ||
                document.querySelector('.main-content') ||
                document.body;
            container.prepend(statusIndicator);
        }
        return statusIndicator;
    }

    updateConnectionStatus(status, message) {
        const statusIndicator = this.createConnectionStatusElement();

        statusIndicator.innerHTML = `
            <span class="status-indicator status-${status}"></span>
            <span class="status-text">${message}</span>
            <span class="status-timestamp">${new Date().toLocaleTimeString()}</span>
        `;

        statusIndicator.className = `connection-status ${status}`;
    } updateSystemStatus(status) {
        const systemStatusElement = document.getElementById('system-status');

        // Handle fallback video notifications
        if (status.status === 'fallback') {
            this.showNotification('Menggunakan video cadangan karena masalah koneksi kamera', 'warning');
            // Add a visible indicator to the video container
            const videoContainer = document.querySelector('.video-container');
            if (videoContainer) {
                const fallbackIndicator = document.createElement('div');
                fallbackIndicator.className = 'fallback-indicator';
                fallbackIndicator.textContent = '⚠️ Menggunakan Video Cadangan';

                // Remove existing indicator if any
                const existingIndicator = videoContainer.querySelector('.fallback-indicator');
                if (existingIndicator) {
                    existingIndicator.remove();
                }

                videoContainer.appendChild(fallbackIndicator);
            }
            return;
        }

        // Handle reconnection notification
        if (status.status === 'reconnected') {
            this.showNotification('Berhasil terhubung kembali ke sumber video asli', 'success');
            // Remove fallback indicator if exists
            const fallbackIndicator = document.querySelector('.fallback-indicator');
            if (fallbackIndicator) {
                fallbackIndicator.remove();
            }
            return;
        }

        // Handle regular system status updates
        if (systemStatusElement && status.cpu_usage !== undefined && status.memory_usage !== undefined) {
            systemStatusElement.innerHTML = `
                <span class="status-label">Sistem:</span>
                <span class="status-value ${status.cpu_usage > 80 ? 'warning' : 'normal'}">
                    CPU: ${status.cpu_usage}%
                </span>
                <span class="status-value ${status.memory_usage > 80 ? 'warning' : 'normal'}">
                    RAM: ${status.memory_usage}%
                </span>
            `;
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

            this.updateConnectionStatus('reconnecting',
                `Mencoba koneksi ulang... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

            setTimeout(() => {
                if (!this.isConnected) {
                    this.socket.connect();
                }
            }, delay);
        } else {
            this.updateConnectionStatus('failed', 'Koneksi gagal - Silakan refresh halaman');
            this.showNotification('Koneksi terputus. Silakan refresh halaman.', 'error');
        }
    }

    addRefreshButton() {
        const refreshBtn = document.createElement('button');
        refreshBtn.id = 'refresh-feed';
        refreshBtn.className = 'btn btn-secondary refresh-btn';
        refreshBtn.innerHTML = '<span class="btn-icon">🔄</span> Refresh';
        refreshBtn.title = 'Refresh video feed';

        refreshBtn.addEventListener('click', () => {
            this.refreshVideoFeed();
        });

        const videoContainer = document.querySelector('.video-container');
        if (videoContainer) {
            videoContainer.appendChild(refreshBtn);
        }
    }

    refreshVideoFeed() {
        const videoFeed = document.getElementById('video-feed');
        if (videoFeed) {
            this.showVideoLoading();

            // Force refresh by adding timestamp
            const currentSrc = videoFeed.src;
            if (currentSrc.includes('?')) {
                videoFeed.src = currentSrc.split('?')[0] + '?t=' + Date.now();
            } else {
                videoFeed.src = currentSrc + '?t=' + Date.now();
            }
        }

        // Reconnect socket if needed
        if (!this.isConnected) {
            this.socket.connect();
        }
    }

    showVideoLoading() {
        const videoContainer = document.querySelector('.video-container');
        if (videoContainer) {
            videoContainer.classList.add('loading');
        }
    }

    removeVideoLoading() {
        const videoContainer = document.querySelector('.video-container');
        if (videoContainer) {
            videoContainer.classList.remove('loading');
        }
    }

    showVideoError() {
        const videoContainer = document.querySelector('.video-container');
        if (videoContainer) {
            videoContainer.classList.add('error');
        }
    }

    removeVideoError() {
        const videoContainer = document.querySelector('.video-container');
        if (videoContainer) {
            videoContainer.classList.remove('error');
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <span class="notification-message">${message}</span>
            <button class="notification-close">&times;</button>
        `;

        // Add to page
        document.body.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);

        // Manual close button
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.remove();
        });
    }    // Health check for video feed - reduced frequency to avoid UI updates
    startHealthCheck() {
        setInterval(() => {
            const timeSinceLastUpdate = Date.now() - this.lastUpdateTime;

            // If no updates for 60 seconds, show warning (increased threshold)
            if (timeSinceLastUpdate > 60000 && this.isConnected) {
                // Update connection status without animation
                const statusIndicator = this.createConnectionStatusElement();
                statusIndicator.innerHTML = `
                    <span class="status-indicator status-warning"></span>
                    <span class="status-text">Feed mungkin bermasalah - Tidak ada update</span>
                    <span class="status-timestamp">${new Date().toLocaleTimeString()}</span>
                `;
                statusIndicator.className = `connection-status warning`;
            }
        }, 30000); // Check less frequently - every 30 seconds
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

    // Start health monitoring
    videoFeed.startHealthCheck();

    // Clean up when the page is unloaded
    window.addEventListener('beforeunload', () => {
        videoFeed.disconnect();
    });

    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Press 'R' to refresh feed
        if (e.key === 'r' || e.key === 'R') {
            if (e.ctrlKey || e.metaKey) return; // Don't interfere with browser refresh
            e.preventDefault();
            videoFeed.refreshVideoFeed();
        }

        // Press 'F' for fullscreen video
        if (e.key === 'f' || e.key === 'F') {
            e.preventDefault();
            const video = document.getElementById('video-feed');
            if (video && video.requestFullscreen) {
                video.requestFullscreen();
            }
        }
    });
});
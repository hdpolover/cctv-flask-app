class VideoFeed {
    constructor(options = {}) {
        console.log('[VideoFeed] Initializing with options:', options);
        this.socketUrl = options.socketUrl || window.location.origin;
        this.socket = io(this.socketUrl);
        console.log('[VideoFeed] Socket created for URL:', this.socketUrl);
        this.counterElement = document.getElementById('people-count');
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.lastUpdateTime = Date.now();
        this.videoStatusInterval = null;
        this.wasFallbackMode = false;
        this.frameCounter = 0;
        this.setupSocketHandlers();
        this.initializeUI();
        this.startVideoStatusMonitoring();
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
            console.log('Connected to server - WebSocket connection established');
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
                console.log('Received video frame data, length:', frameData ? frameData.length : 'null');
                const videoFeed = document.getElementById('video-feed');
                if (videoFeed && frameData) {
                    videoFeed.src = `data:image/jpeg;base64,${frameData}`;
                    this.lastUpdateTime = Date.now();
                    this.removeVideoError();
                    console.log('Video frame updated successfully');
                    
                    // Update frame counter for debugging
                    if (!this.frameCounter) this.frameCounter = 0;
                    this.frameCounter++;
                    
                    // Update a debug counter on page if element exists
                    const debugCounter = document.getElementById('debug-frame-counter');
                    if (debugCounter) {
                        debugCounter.textContent = `Frames received: ${this.frameCounter}`;
                    }
                } else {
                    console.warn('Video feed element not found or no frame data');
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
        });        // Handle system status updates
        this.socket.on('system_status', (status) => {
            this.updateSystemStatus(status);
        });
        
        // Handle RTSP monitor status updates
        this.socket.on('rtsp_status', (status) => {
            this.updateRTSPStatus(status);
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
        const statusIndicator = document.getElementById('connection-status');
        if (!statusIndicator) return;

        // Update just the text content for the existing template element
        statusIndicator.textContent = message;
        statusIndicator.className = `detail-value status-${status}`;
    }    updateSystemStatus(status) {
        const systemStatusElement = document.getElementById('system-status');

        // Let the video status monitoring handle fallback states
        // This method now focuses on system performance metrics
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

    updateRTSPStatus(status) {
        // Update RTSP monitor status in the UI
        if (status.monitor_active) {
            this.showNotification('RTSP monitor active - automatic reconnection enabled', 'info');
        }
        
        if (status.reconnection_performed) {
            this.showNotification('RTSP stream automatically reconnected', 'success');
        }
        
        if (status.stream_unhealthy) {
            this.showNotification('RTSP stream health warning - monitoring for reconnection', 'warning');
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
    }    addRefreshButton() {
        const refreshBtn = document.createElement('button');
        refreshBtn.id = 'refresh-feed';
        refreshBtn.className = 'btn btn-secondary refresh-btn';
        refreshBtn.innerHTML = '<span class="btn-icon">🔄</span> Refresh';
        refreshBtn.title = 'Refresh video feed (or press R)';

        refreshBtn.addEventListener('click', () => {
            this.refreshVideoFeed();
        });

        const videoContainer = document.querySelector('.video-container');
        if (videoContainer) {
            videoContainer.appendChild(refreshBtn);
        }
        
        // Show initial tip about refresh shortcut
        setTimeout(() => {
            this.showNotification('Tip: Press "R" to refresh video feed or click the refresh button', 'info');
        }, 3000);
    }refreshVideoFeed() {
        const refreshBtn = document.getElementById('refresh-feed');
        const videoFeed = document.getElementById('video-feed');
        
        if (refreshBtn) {
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '<span class="btn-icon">⏳</span> Refreshing...';
        }
        
        if (videoFeed) {
            this.showVideoLoading();
        }

        // Call backend to refresh video service
        fetch('/refresh_video', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                this.showNotification(data.message, 'success');
                
                // Force refresh the video feed
                if (videoFeed) {
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
                
                // Reset connection status
                this.updateConnectionStatus('connected', 'Video service refreshed successfully');
            } else {
                this.showNotification('Failed to refresh video: ' + data.message, 'error');
            }
        })
        .catch(error => {
            console.error('Error refreshing video:', error);
            this.showNotification('Network error while refreshing video', 'error');
            
            // Fallback: just refresh the image source
            if (videoFeed) {
                const currentSrc = videoFeed.src;
                if (currentSrc.includes('?')) {
                    videoFeed.src = currentSrc.split('?')[0] + '?t=' + Date.now();
                } else {
                    videoFeed.src = currentSrc + '?t=' + Date.now();
                }
            }
        })
        .finally(() => {
            // Re-enable refresh button
            if (refreshBtn) {
                refreshBtn.disabled = false;
                refreshBtn.innerHTML = '<span class="btn-icon">🔄</span> Refresh';
            }
            
            // Remove loading state after a delay
            setTimeout(() => {
                this.removeVideoLoading();
            }, 1000);
        });
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

    startVideoStatusMonitoring() {
        this.videoStatusInterval = setInterval(() => {
            this.checkVideoStatus();
        }, 5000); // Check every 5 seconds
    }

    checkVideoStatus() {
        fetch('/api/video-status')
            .then(response => response.json())
            .then(data => {
                this.handleVideoStatusUpdate(data);
            })
            .catch(error => {
                console.error('Error checking video status:', error);
            });
    }

    handleVideoStatusUpdate(status) {
        const videoContainer = document.querySelector('.video-container');
        const statsContainer = document.querySelector('.stats-container') || 
                              document.querySelector('.counter-section');
        
        // Remove existing status indicators
        this.removeFallbackIndicators();
        
        if (status.is_fallback_active) {
            // Show fallback warning
            this.showFallbackWarning(status.fallback_reason, status.original_video_path);
            
            // Hide or disable detection stats
            if (statsContainer) {
                this.disableDetectionStats(statsContainer);
            }
            
            // Update connection status
            this.updateConnectionStatus('fallback', 
                `Demo mode aktif - ${status.fallback_reason || 'Kamera tidak tersedia'}`);
                
        } else {
            // Normal operation - ensure stats are enabled
            if (statsContainer) {
                this.enableDetectionStats(statsContainer);
            }
            
            // Update connection status to show normal operation
            this.updateConnectionStatus('connected', 'Terhubung - Deteksi aktif');
            
            // Update connection status if we were in fallback
            if (this.wasFallbackMode) {
                this.showNotification('Berhasil terhubung ke sumber video', 'success');
            }
        }
        
        this.wasFallbackMode = status.is_fallback_active;
    }

    showFallbackWarning(reason, originalPath) {
        const videoContainer = document.querySelector('.video-container');
        if (!videoContainer) return;
        
        const fallbackIndicator = document.createElement('div');
        fallbackIndicator.className = 'fallback-indicator';
        fallbackIndicator.innerHTML = `
            <div class="fallback-content">
                <span class="fallback-icon">⚠️</span>
                <span class="fallback-title">Mode Demo Aktif</span>
                <span class="fallback-reason">${reason || 'Kamera tidak tersedia'}</span>
                <small class="fallback-note">Deteksi otomatis dinonaktifkan</small>
            </div>
        `;
        
        videoContainer.appendChild(fallbackIndicator);
        
        // Show notification
        this.showNotification(
            `Mode demo aktif: ${reason || 'Kamera tidak tersedia'}. Deteksi dinonaktifkan.`, 
            'warning'
        );
    }

    removeFallbackIndicators() {
        const indicators = document.querySelectorAll('.fallback-indicator');
        indicators.forEach(indicator => indicator.remove());
    }

    disableDetectionStats(container) {
        container.classList.add('stats-disabled');
        
        // Add overlay message
        let overlay = container.querySelector('.stats-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'stats-overlay';
            overlay.innerHTML = `
                <div class="overlay-content">
                    <span class="overlay-icon">📊</span>
                    <span class="overlay-text">Statistik deteksi tidak tersedia dalam mode demo</span>
                </div>
            `;
            container.appendChild(overlay);
        }
    }

    enableDetectionStats(container) {
        container.classList.remove('stats-disabled');
        
        // Remove overlay
        const overlay = container.querySelector('.stats-overlay');
        if (overlay) {
            overlay.remove();
        }
    }

    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
        }

        if (this.videoStatusInterval) {
            clearInterval(this.videoStatusInterval);
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
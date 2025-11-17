let map;
let mineMarkers = [];
let predictedMarker = null;

// Initialize the map
function initMap() {
    map = L.map('map').setView([0, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
}

// Initialize when the page loads
document.addEventListener('DOMContentLoaded', function() {
    initMap();
    updateStatistics();
    loadLocations();
});

// Add a new mine location
async function addLocation() {
    const lat = parseFloat(document.getElementById('latitude').value);
    const lon = parseFloat(document.getElementById('longitude').value);

    if (isNaN(lat) || isNaN(lon)) {
        alert('Please enter valid coordinates');
        return;
    }

    try {
        const response = await fetch('/add_location', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                latitude: lat,
                longitude: lon
            })
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Clear inputs
        document.getElementById('latitude').value = '';
        document.getElementById('longitude').value = '';

        // Update the map and lists
        loadLocations();
        updateStatistics();

        if (data.pattern) {
            alert(`Pattern detected: ${data.pattern.pattern} (Confidence: ${data.pattern.confidence.toFixed(1)}%)`);
        }
    } catch (error) {
        alert('Error adding location: ' + error.message);
    }
}

// Predict next mine location
async function predict() {
    try {
        const response = await fetch('/predict');
        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Remove previous prediction marker if it exists
        if (predictedMarker) {
            map.removeLayer(predictedMarker);
        }

        // Add new prediction marker
        predictedMarker = L.marker([data.latitude, data.longitude], {
            icon: L.divIcon({
                className: 'predicted-marker',
                html: '⭐',
                iconSize: [20, 20]
            })
        }).addTo(map);

        map.setView([data.latitude, data.longitude], map.getZoom());
        
        alert(`Prediction confidence: ${data.confidence.toFixed(1)}%`);
    } catch (error) {
        alert('Error making prediction: ' + error.message);
    }
}

// Detect pattern
async function detectPattern() {
    try {
        const response = await fetch('/detect_pattern');
        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        alert(`Detected pattern: ${data.pattern} (Confidence: ${data.confidence.toFixed(1)}%)`);
    } catch (error) {
        alert('Error detecting pattern: ' + error.message);
    }
}

// Provide feedback on prediction
async function provideFeedback(success) {
    if (!predictedMarker) {
        alert('No prediction to provide feedback for');
        return;
    }

    try {
        const response = await fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                success: success
            })
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Update statistics
        updateStatistics();
        loadLocations();

        // Remove prediction marker
        map.removeLayer(predictedMarker);
        predictedMarker = null;

        if (data.message === "Maximum failure attempts reached") {
            alert('Maximum failure attempts reached. Please try a new prediction.');
        }
    } catch (error) {
        alert('Error providing feedback: ' + error.message);
    }
}

// Update statistics display
async function updateStatistics() {
    try {
        const response = await fetch('/statistics');
        const data = await response.json();

        const statsHtml = `
            <div class="stat-item">
                <span>Total Locations:</span>
                <span>${data.total_locations}</span>
            </div>
            <div class="stat-item">
                <span>Pattern Type:</span>
                <span>${data.pattern_type}</span>
            </div>
            <div class="stat-item">
                <span>Success Count:</span>
                <span>${data.success_count}</span>
            </div>
            <div class="stat-item">
                <span>Failure Count:</span>
                <span>${data.failure_count}</span>
            </div>
            <div class="stat-item">
                <span>Accuracy:</span>
                <span>${data.accuracy.toFixed(1)}%</span>
            </div>
        `;

        document.getElementById('statsDisplay').innerHTML = statsHtml;
    } catch (error) {
        console.error('Error updating statistics:', error);
    }
}

// Load all locations
async function loadLocations() {
    try {
        const response = await fetch('/locations');
        const data = await response.json();

        // Clear existing markers
        mineMarkers.forEach(marker => map.removeLayer(marker));
        mineMarkers = [];

        // Add mine locations
        data.mine_locations.forEach((loc, index) => {
            const marker = L.marker([loc[0], loc[1]], {
                icon: L.divIcon({
                    className: 'mine-marker',
                    html: '📍',
                    iconSize: [20, 20]
                })
            }).addTo(map);
            mineMarkers.push(marker);
        });

        // Update locations list
        const locationsHtml = data.mine_locations.map((loc, index) => `
            <div class="location-item">
                Mine ${index + 1}: Lat ${loc[0].toFixed(6)}, Lon ${loc[1].toFixed(6)}
            </div>
        `).join('');
        document.getElementById('locationsList').innerHTML = locationsHtml;

        // If there are locations, center the map on the last one
        if (data.mine_locations.length > 0) {
            const lastLoc = data.mine_locations[data.mine_locations.length - 1];
            map.setView([lastLoc[0], lastLoc[1]], 15);
        }
    } catch (error) {
        console.error('Error loading locations:', error);
    }
}
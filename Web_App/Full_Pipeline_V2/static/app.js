let map;
let homeMarker, schoolMarker;

// Initialize Leaflet map
function initMap() {
    map = L.map('map').setView([37.779, -122.246], 12); // default center

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);
}

// Call initMap on page load
document.addEventListener('DOMContentLoaded', initMap);

// Helper function to geocode an address (using Nominatim API)
async function geocode(address) {
    const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(address)}`;
    const res = await fetch(url);
    const data = await res.json();
    if (data.length > 0) {
        return [parseFloat(data[0].lat), parseFloat(data[0].lon)];
    } else {
        alert(`Address not found: ${address}`);
        return null;
    }
}

// Plot addresses on map
document.getElementById('plotMapBtn').addEventListener('click', async () => {
    const homeAddr = document.getElementById('home').value;
    const schoolAddr = document.getElementById('school').value;

    const homeCoords = await geocode(homeAddr);
    const schoolCoords = await geocode(schoolAddr);

    if (!homeCoords || !schoolCoords) return;

    // Remove existing markers
    if (homeMarker) map.removeLayer(homeMarker);
    if (schoolMarker) map.removeLayer(schoolMarker);

    // Add new markers
    homeMarker = L.marker(homeCoords).addTo(map).bindPopup('Home').openPopup();
    schoolMarker = L.marker(schoolCoords).addTo(map).bindPopup('School');

    // Fit map to markers
    const bounds = L.latLngBounds([homeCoords, schoolCoords]);
    map.fitBounds(bounds, {padding: [50, 50]});
});

// Generate Demographic Data
document.getElementById("getDataBtn").addEventListener("click", async () => {
    const home = document.getElementById("home").value;
    const school = document.getElementById("school").value;
    console.log("clicked")

    const res = await fetch("/get_data", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({home, school})
    });

    const data = await res.json();

    let out = "";
    for (const [key, value] of Object.entries(data)) {
        out += `${key}: ${value}\n`;
    }

    const box = document.getElementById("dataOutput");
    box.style.display = "block";
    box.textContent = out;
});

// Predict Bus Need
document.getElementById("predictBtn").addEventListener("click", async () => {
    const home = document.getElementById("home").value;
    const school = document.getElementById("school").value;

    const res = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({home, school})
    });

    const result = await res.json();

    const needsBus = result.needs_bus ? "Needs Bus" : "Does NOT Need Bus";
    const prob = (result.probability * 100).toFixed(2);

    const box = document.getElementById("predictionOutput");
    box.style.display = "block";
    box.textContent = `${needsBus}\nProbability: ${prob}%`;
});

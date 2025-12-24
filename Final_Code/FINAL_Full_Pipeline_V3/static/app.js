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

document.getElementById("getDataBtn").addEventListener("click", async () => {
    const home = document.getElementById("home").value;
    const school = document.getElementById("school").value;

    const res = await fetch("/get_data", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({home, school})
    });

    const data = await res.json();

    const box = document.getElementById("dataOutput");
    box.innerHTML = ""; // clear previous content
    box.style.display = "block";

    // Create a separate line for each key/value
    for (const [key, value] of Object.entries(data)) {
        const line = document.createElement("div");
        line.className = "py-1 border-b border-black/20"; // optional styling
        line.innerHTML = `<strong>${key}:</strong> ${value}`;
        box.appendChild(line);
    }
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


//HEATMAP
// Heatmap logic (requires Leaflet + chroma.js)
let heatmapLayer = null;
let mapHeat = null;

// initialize map for heatmap page (separate div id heatmap_map)
function initHeatmapLeaflet() {
    if (mapHeat) return;
    mapHeat = L.map('heatmap_map', { preferCanvas: true }).setView([37.75, -122.25], 11);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(mapHeat);
}

document.addEventListener('DOMContentLoaded', () => {
    initHeatmapLeaflet();

    document.getElementById("genHeatBtn").addEventListener("click", async () => {
        const school = document.getElementById("heat_school").value;
        const radius = parseFloat(document.getElementById("heat_radius").value);

        if (!school || !radius) {
            alert("Please provide school address and radius (miles).");
            return;
        }
        await generateHeatmap(school, radius);
    });

    document.getElementById("clearHeatBtn").addEventListener("click", () => {
        if (heatmapLayer) {
            mapHeat.removeLayer(heatmapLayer);
            heatmapLayer = null;
        }
        document.getElementById("heat_detail_panel").classList.add("hidden");
    });

    document.getElementById("closeDetail").addEventListener("click", () => {
        document.getElementById("heat_detail_panel").classList.add("hidden");
    });
});

async function generateHeatmap(school, radius) {
    try {
        // show spinner/disable button
        const btn = document.getElementById("genHeatBtn");
        btn.disabled = true;
        btn.textContent = "Generating...";

        const res = await fetch("/heatmap_data", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ school: school, radius: radius })
        });
        
        console.log("reached2")
        const body = await res.json();
        btn.disabled = false;
        btn.textContent = "Generate Heatmap";

        if (!res.ok) {
            alert("Server error: " + (body.error || res.statusText));
            return;
        }
        

        const geojson = body.geojson;
        console.log("reached3")
        console.log(geojson)
        renderGeoJSONHeatmap(geojson);
        addHeatmapLegend(chroma.scale(['magenta', 'orange', 'yellow']).domain([0, 1]));

    } catch (err) {
        console.error(err);
        alert("Error generating heatmap: " + err);
    }
}

// color scale (inferno) mapping from 0..1 -> color (chroma.js)



function renderGeoJSONHeatmap(geojson) {
    // Remove previous layer
    if (heatmapLayer) {
        mapHeat.removeLayer(heatmapLayer);
        heatmapLayer = null;
    }

    // Make sure colorScale is properly initialized
    // Replace these colors with your preferred gradient
    const colorScale = chroma.scale(['magenta', 'orange', 'yellow']).domain([0, 1]);

    // create layer
    heatmapLayer = L.geoJSON(geojson, {
        style: function(feature) {
            const props = feature.properties || {};
            const predRaw = props.pred;
            const pred = predRaw === null || predRaw === undefined ? NaN : Number(predRaw);

            let c; // color to use
            if (isFinite(pred)) {
                try {
                    c = colorScale(pred).hex();
                } catch (err) {
                    console.error("Chroma error for pred:", pred, err);
                    c = "#ff00ff"; // fallback magenta for debugging
                }
            } else {
                c = "#cccccc"; // gray for missing data
            }

            return {
                color: c,
                weight: 0.2,
                fillColor: c,
                fillOpacity: isFinite(pred) ? 0.8 : 0.7
            };
        },
        onEachFeature: function(feature, layer) {
            const props = feature.properties || {};
            const hoverFields = ["geoid","rf_prob","rf_pred","pred","state_fips","county_fips","tract","block_group"];
            let hoverText = "";
            hoverFields.forEach(f => {
                if (props[f] !== undefined) {
                    hoverText += `<b>${f}:</b> ${props[f]}<br/>`;
                }
            });

            layer.bindTooltip(hoverText + "<i>Click for full details</i>", {
                direction: "auto",
                sticky: true,
                opacity: 0.9
            });

            layer.on('click', function(e) {
                openDetailPanel(props);
            });
        }
    }).addTo(mapHeat);

    // auto-zoom to layer bounds (if features exist)
    try {
        const bounds = heatmapLayer.getBounds();
        if (bounds.isValid()) {
            mapHeat.fitBounds(bounds, {padding: [30,30]});
        }
    } catch (err) {
        console.warn("Could not auto-zoom:", err);
    }
}

function addHeatmapLegend(colorScale) {
    const legend = L.control({ position: 'bottomright' });

    legend.onAdd = function(map) {
        const div = L.DomUtil.create('div', 'info legend p-2 rounded-lg text-black bg-white/80');

        // Create a gradient bar
        const gradient = document.createElement('div');
        gradient.style.width = '120px';
        gradient.style.height = '20px';
        gradient.style.background = `linear-gradient(to right, ${colorScale(0).hex()}, ${colorScale(1).hex()})`;
        gradient.style.border = '1px solid #000';
        gradient.style.marginBottom = '4px';
        div.appendChild(gradient);

        // Add min/max labels
        const labels = document.createElement('div');
        labels.style.display = 'flex';
        labels.style.justifyContent = 'space-between';
        labels.style.fontSize = '12px';
        labels.innerHTML = `<span>0</span><span>1</span>`; // change 0 and 1 to your scale range if needed
        div.appendChild(labels);

        // Add title
        const title = document.createElement('div');
        title.style.fontSize = '12px';
        title.style.fontWeight = 'bold';
        title.style.marginBottom = '2px';
        title.innerText = 'Probability';
        div.prepend(title);

        return div;
    };

    legend.addTo(mapHeat);
}



// show full props in side panel transposed (vertical list)
function openDetailPanel(props) {
    const panel = document.getElementById("heat_detail_panel");
    const content = document.getElementById("detail_content");
    // create a transposed list
    let html = '<div class="space-y-2">';
    // Sort keys for determinism (you can change ordering)
    const keys = Object.keys(props).sort();
    keys.forEach(k => {
        let v = props[k];
        // nicely format JSON/arrays
        if (typeof v === 'object') v = JSON.stringify(v);
        html += `<div class="border-b pb-1"><div class="font-semibold text-xs text-gray-600">${k}</div><div class="text-sm">${v}</div></div>`;
    });
    html += '</div>';
    content.innerHTML = html;
    panel.classList.remove("hidden");
}

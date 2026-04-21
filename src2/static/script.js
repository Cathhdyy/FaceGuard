let currentModel = 'haar';

function setModel(model) {
    currentModel = model;

    // Update UI
    document.querySelectorAll('.model-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.model === model) {
            btn.classList.add('active');
        }
    });

    // Call API
    fetch('/api/config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model: model }),
    })
        .then(response => response.json())
        .then(data => console.log('Model switched to:', data.model))
        .catch((error) => console.error('Error:', error));
}

function updateStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            // Update Metrics
            document.getElementById('frame-count').textContent = data.frame_count || 0;
            document.getElementById('face-count').textContent = data.faces || 0;
            document.getElementById('processing-fps').textContent = data.processing_fps !== undefined ? data.processing_fps.toFixed(1) : '0.0';

            // Update Names List
            const namesContainer = document.getElementById('names-list');
            namesContainer.innerHTML = '';

            if (data.names.length === 0) {
                namesContainer.innerHTML = '<span class="placeholder">Scanning...</span>';
            } else {
                data.names.forEach((name, index) => {
                    const tag = document.createElement('span');
                    const isUnknown = ['Unknown', 'Scanning...', 'Analyzing...'].includes(name);
                    tag.className = `name-tag ${isUnknown ? 'unknown' : ''}`;
                    tag.textContent = name;
                    
                    if (name === 'Unknown' || name === 'Scanning...' || name === 'Analyzing...') {
                        const regBtn = document.createElement('button');
                        regBtn.className = 'mini-reg-btn';
                        regBtn.innerHTML = '+';
                        regBtn.title = 'Register this face';
                        regBtn.onclick = (e) => {
                            e.stopPropagation();
                            openModal(index);
                        };
                        tag.appendChild(regBtn);
                    }
                    
                    namesContainer.appendChild(tag);
                });
            }

            // Update Emotions List
            const emotionsContainer = document.getElementById('emotions-list');
            emotionsContainer.innerHTML = '';
            if (!data.emotions || data.emotions.length === 0) {
                emotionsContainer.innerHTML = '<span class="placeholder">Analyzing...</span>';
            } else {
                data.emotions.forEach(emotion => {
                    const tag = document.createElement('span');
                    tag.className = 'name-tag';
                    tag.style.background = 'rgba(52, 152, 219, 0.2)';
                    tag.style.color = '#3498db';
                    tag.style.borderColor = '#3498db';
                    tag.textContent = emotion;
                    emotionsContainer.appendChild(tag);
                });
            }
        })
        .catch(err => console.error(err));
}

function backToLive() {
    document.getElementById('main-display').style.display = 'block';
    document.getElementById('result-view').style.display = 'none';
}

// File upload handling
document.getElementById('file-input').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const status = document.getElementById('upload-status');
    status.textContent = 'Processing...';
    status.className = 'upload-status processing';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', currentModel);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.status === 'success') {
            if (data.type === 'image') {
                // Show processed image
                document.getElementById('main-display').style.display = 'none';
                document.getElementById('result-view').style.display = 'block';
                document.getElementById('result-image').src = `/uploads/${data.filename}?t=${Date.now()}`;

                status.textContent = `✓ Found ${data.faces} faces`;
                status.className = 'upload-status success';
            } else {
                status.textContent = '✓ ' + data.message;
                status.className = 'upload-status success';
            }
        } else {
            status.textContent = '✗ ' + data.error;
            status.className = 'upload-status error';
        }
    } catch (err) {
        status.textContent = '✗ Upload failed';
        status.className = 'upload-status error';
    }

    // Clear input
    e.target.value = '';

    // Clear status after 5s
    setTimeout(() => {
        status.textContent = '';
        status.className = '';
    }, 5000);
});

// Modal Logic
function openModal(index) {
    document.getElementById('reg-index').value = index;
    document.getElementById('register-modal').style.display = 'flex';
    document.getElementById('reg-name').focus();
}

function closeModal() {
    document.getElementById('register-modal').style.display = 'none';
    document.getElementById('reg-name').value = '';
}

async function submitRegistration() {
    const name = document.getElementById('reg-name').value.trim();
    const faceIdx = document.getElementById('reg-index').value;
    const btn = document.getElementById('confirm-reg');
    
    if (!name) {
        alert("Please enter a name");
        return;
    }

    btn.disabled = true;
    btn.textContent = "Capturing...";

    try {
        // Step 1: Capture
        const regRes = await fetch('/api/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, face_idx: parseInt(faceIdx) })
        });
        
        const regData = await regRes.json();
        if (regData.status !== 'success') throw new Error(regData.message);

        // Step 2: Retrain
        btn.textContent = "Training AI...";
        const trainRes = await fetch('/api/retrain', { method: 'POST' });
        const trainData = await trainRes.json();

        // Show success
        btn.style.background = "#2ecc71";
        btn.textContent = "Success!";
        
        setTimeout(() => {
            closeModal();
            btn.disabled = false;
            btn.style.background = "";
            btn.textContent = "Register & Train";
        }, 1500);

    } catch (err) {
        alert("Error: " + err.message);
        btn.disabled = false;
        btn.textContent = "Register & Train";
    }
}

// Close modal on escape
window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
});

async function manualRetrain() {
    const btn = document.getElementById('manual-retrain');
    const status = document.getElementById('retrain-status');
    
    btn.disabled = true;
    btn.textContent = "⚙ Processing...";
    status.textContent = "Regenerating AI Index...";

    try {
        const response = await fetch('/api/retrain', { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'success') {
            btn.style.borderColor = "#2ecc71";
            btn.style.color = "#2ecc71";
            btn.textContent = "✓ Optimized";
            status.textContent = "FAISS Index Ready";
        } else {
            throw new Error(data.message);
        }
    } catch (err) {
        btn.style.borderColor = "#e74c3c";
        btn.style.color = "#e74c3c";
        btn.textContent = "✗ Error";
        status.textContent = err.message;
    }

    setTimeout(() => {
        btn.disabled = false;
        btn.style.borderColor = "";
        btn.style.color = "";
        btn.textContent = "🔄 Retrain AI Index";
        status.textContent = "";
    }, 3000);
}

// Poll stats every 1 second
setInterval(updateStats, 1000);

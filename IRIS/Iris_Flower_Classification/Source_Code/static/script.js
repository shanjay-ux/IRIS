document.addEventListener('DOMContentLoaded', () => {
    // Feature keys
    const features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'];

    // DOM Elements
    const form = document.getElementById('prediction-form');
    const resetBtn = document.getElementById('reset-btn');
    const refreshAnalyticsBtn = document.getElementById('refresh-analytics-btn');
    const speciesCard = document.getElementById('species-card');
    const resultDisplayName = document.getElementById('result-display-name');
    const resultConfidence = document.getElementById('result-confidence');
    const resultDescription = document.getElementById('result-description');
    const historyTbody = document.getElementById('history-tbody');
    const liveIndicator = document.getElementById('live-indicator');
    
    // Analytics Elements
    const anaTotalInferences = document.getElementById('ana-total-inferences');
    const anaAvgConfidence = document.getElementById('ana-avg-confidence');
    const anaMostCommon = document.getElementById('ana-most-common');
    const rateLimitAlert = document.getElementById('rate-limit-alert');
    const rateLimitMsg = document.getElementById('rate-limit-msg');

    // Anatomy Stats
    const statPetalRatio = document.getElementById('stat-petal-ratio');
    const statSepalRatio = document.getElementById('stat-sepal-ratio');
    const statArea = document.getElementById('stat-area');

    // 1. Synchronize Range Sliders and Numeric Inputs
    features.forEach(feature => {
        const slider = document.getElementById(feature);
        const numInput = document.getElementById(`${feature}-num`);

        if (slider && numInput) {
            slider.addEventListener('input', () => {
                numInput.value = slider.value;
                clearFieldError(feature);
                updateAnatomyCalculations();
                triggerAutoPredict();
            });

            numInput.addEventListener('input', () => {
                let val = parseFloat(numInput.value);
                const min = parseFloat(slider.min);
                const max = parseFloat(slider.max);

                if (!isNaN(val)) {
                    if (val >= min && val <= max) {
                        slider.value = val;
                        clearFieldError(feature);
                        updateAnatomyCalculations();
                        triggerAutoPredict();
                    }
                }
            });
        }
    });

    // 2. Preset Buttons Click Handler
    const presetBtns = document.querySelectorAll('.preset-btn');
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const sl = btn.getAttribute('data-sl');
            const sw = btn.getAttribute('data-sw');
            const pl = btn.getAttribute('data-pl');
            const pw = btn.getAttribute('data-pw');

            setFeatureValue('sepal_length', sl);
            setFeatureValue('sepal_width', sw);
            setFeatureValue('petal_length', pl);
            setFeatureValue('petal_width', pw);

            updateAnatomyCalculations();
            performPrediction();
        });
    });

    function setFeatureValue(feature, val) {
        const slider = document.getElementById(feature);
        const numInput = document.getElementById(`${feature}-num`);
        if (slider) slider.value = val;
        if (numInput) numInput.value = val;
        clearFieldError(feature);
    }

    // 3. Anatomy Calculations
    function updateAnatomyCalculations() {
        const sl = parseFloat(document.getElementById('sepal_length').value) || 0;
        const sw = parseFloat(document.getElementById('sepal_width').value) || 1;
        const pl = parseFloat(document.getElementById('petal_length').value) || 0;
        const pw = parseFloat(document.getElementById('petal_width').value) || 1;

        const petalRatio = (pl / pw).toFixed(2);
        const sepalRatio = (sl / sw).toFixed(2);
        const estArea = ((sl * sw * 0.75) + (pl * pw * 0.75)).toFixed(2);

        if (statPetalRatio) statPetalRatio.textContent = petalRatio;
        if (statSepalRatio) statSepalRatio.textContent = sepalRatio;
        if (statArea) statArea.textContent = `${estArea} cm²`;
    }

    // Debounce timer for auto predict on slider move
    let autoPredictTimer = null;
    function triggerAutoPredict() {
        if (autoPredictTimer) clearTimeout(autoPredictTimer);
        autoPredictTimer = setTimeout(() => {
            performPrediction();
        }, 250);
    }

    // 4. Form Submit & Prediction API Fetch
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        performPrediction();
    });

    resetBtn.addEventListener('click', () => {
        setFeatureValue('sepal_length', '5.1');
        setFeatureValue('sepal_width', '3.5');
        setFeatureValue('petal_length', '1.4');
        setFeatureValue('petal_width', '0.2');
        updateAnatomyCalculations();
        performPrediction();
    });

    async function performPrediction() {
        hideRateLimitAlert();
        
        const payload = {};
        let hasError = false;

        features.forEach(feature => {
            const val = parseFloat(document.getElementById(feature).value);
            if (isNaN(val)) {
                showFieldError(feature, 'Required number');
                hasError = true;
            } else {
                payload[feature] = val;
            }
        });

        if (hasError) return;

        if (liveIndicator) liveIndicator.textContent = 'Analyzing...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (liveIndicator) liveIndicator.textContent = 'Live Inference';

            if (response.status === 429) {
                showRateLimitAlert(data.error || 'Rate limit exceeded (Max 60 requests/min)');
                return;
            }

            if (!response.ok || !data.success) {
                if (data.field_errors) {
                    Object.keys(data.field_errors).forEach(key => {
                        showFieldError(key, data.field_errors[key]);
                    });
                }
                return;
            }

            // Render prediction output
            renderPredictionResults(data);
            fetchAnalyticsAndHistory();

        } catch (err) {
            console.error('Prediction API Error:', err);
            if (liveIndicator) liveIndicator.textContent = 'Connection Error';
        }
    }

    // 5. Update UI Results
    function renderPredictionResults(data) {
        const species = data.predicted_species.toLowerCase();

        // Update Card Theme Class
        speciesCard.className = 'species-card';
        if (species === 'setosa') speciesCard.classList.add('setosa-theme');
        else if (species === 'versicolor') speciesCard.classList.add('versicolor-theme');
        else if (species === 'virginica') speciesCard.classList.add('virginica-theme');

        // Text & Metadata Updates
        resultDisplayName.textContent = data.display_name;
        resultConfidence.textContent = `${data.confidence.toFixed(1)}%`;
        if (data.species_metadata && data.species_metadata.description) {
            resultDescription.textContent = data.species_metadata.description;
        }

        // Update Probability Bars
        const probs = data.probabilities;
        for (const [sp, val] of Object.entries(probs)) {
            const spLower = sp.toLowerCase();
            const barFill = document.getElementById(`prob-bar-${spLower}`);
            const valText = document.getElementById(`prob-val-${spLower}`);

            if (barFill && valText) {
                barFill.style.width = `${val}%`;
                valText.textContent = `${val.toFixed(1)}%`;
            }
        }
    }

    // 6. Fetch Analytics & Database History
    async function fetchAnalyticsAndHistory() {
        try {
            const res = await fetch('/api/analytics');
            if (!res.ok) return;
            const data = await res.json();
            
            if (data.success) {
                const summary = data.analytics;
                if (anaTotalInferences) anaTotalInferences.textContent = summary.total_inferences;
                if (anaAvgConfidence) anaAvgConfidence.textContent = `${summary.average_confidence}%`;
                if (anaMostCommon) anaMostCommon.textContent = capitalize(summary.most_common_species);

                renderHistoryTable(data.history);
            }
        } catch (err) {
            console.error('Analytics Fetch Error:', err);
        }
    }

    function renderHistoryTable(logs) {
        if (!logs || logs.length === 0) {
            historyTbody.innerHTML = `
                <tr class="empty-row">
                    <td colspan="7">No prediction logs recorded in database yet. Adjust features above to perform predictions!</td>
                </tr>
            `;
            return;
        }

        historyTbody.innerHTML = logs.map(log => {
            const spLower = log.predicted_species.toLowerCase();
            const displayName = `Iris ${capitalize(spLower)}`;
            return `
                <tr>
                    <td><strong>#${log.id}</strong></td>
                    <td>${log.sepal_length.toFixed(1)} cm</td>
                    <td>${log.sepal_width.toFixed(1)} cm</td>
                    <td>${log.petal_length.toFixed(1)} cm</td>
                    <td>${log.petal_width.toFixed(1)} cm</td>
                    <td><span class="species-pill pill-${spLower}">${displayName}</span></td>
                    <td><strong>${log.confidence.toFixed(1)}%</strong></td>
                </tr>
            `;
        }).join('');
    }

    if (refreshAnalyticsBtn) {
        refreshAnalyticsBtn.addEventListener('click', () => {
            fetchAnalyticsAndHistory();
        });
    }

    // Error UI helpers
    function showFieldError(feature, msg) {
        const group = document.getElementById(`group-${feature}`);
        const errSpan = document.getElementById(`error-${feature}`);
        if (group) group.classList.add('has-error');
        if (errSpan) errSpan.textContent = msg;
    }

    function clearFieldError(feature) {
        const group = document.getElementById(`group-${feature}`);
        const errSpan = document.getElementById(`error-${feature}`);
        if (group) group.classList.remove('has-error');
        if (errSpan) errSpan.textContent = '';
    }

    function showRateLimitAlert(msg) {
        if (rateLimitAlert && rateLimitMsg) {
            rateLimitMsg.textContent = msg;
            rateLimitAlert.classList.remove('hidden');
        }
    }

    function hideRateLimitAlert() {
        if (rateLimitAlert) rateLimitAlert.classList.add('hidden');
    }

    function capitalize(str) {
        if (!str) return '';
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    // Initial setup
    updateAnatomyCalculations();
    performPrediction();
    fetchAnalyticsAndHistory();
});

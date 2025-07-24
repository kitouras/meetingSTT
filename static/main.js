const form = document.getElementById('uploadForm');
const audioFileInput = document.getElementById('audioFile');
const summaryResultDiv = document.getElementById('summaryResult');
const summaryTextP = document.getElementById('summaryText');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorMessageDiv = document.getElementById('errorMessage');
const downloadSummaryLink = document.getElementById('downloadSummaryLink');
const downloadTransriptionLink = document.getElementById('downloadTransriptionLink');
const resourceUsageDiv = document.getElementById('resourceUsage');
const cpuUsageSpan = document.getElementById('cpuUsage');
const memUsageSpan = document.getElementById('memUsage');
const processMemUsageSpan = document.getElementById('processMemUsage');
const gpuUtilUsageSpan = document.getElementById('gpuUtilUsage');
const gpuMemUsageSpan = document.getElementById('gpuMemUsage');
const gpuMemUsedSpan = document.getElementById('gpuMemUsed');
const gpuMemTotalSpan = document.getElementById('gpuMemTotal');


let resourcePollInterval = null;
let resourcePollEnabled = false;

const fetchResourceUsage = async () => {
    if (!resourcePollEnabled) return;

    try {
        const response = await fetch('/resources');
        if (response.ok) {
            const data = await response.json();

            if (!resourcePollEnabled) return;

            cpuUsageSpan.textContent = data.cpu_percent != null ? data.cpu_percent.toFixed(1) : '--';
            memUsageSpan.textContent = data.mem_percent != null ? data.mem_percent.toFixed(1) : '--';
            processMemUsageSpan.textContent = data.process_mem_mb != null ? data.process_mem_mb.toFixed(2) : '--';
            
            const gpuAvailable = data.gpu_utilization_percent != null || data.gpu_mem_percent != null;
            gpuUtilUsageSpan.textContent = data.gpu_utilization_percent != null ? data.gpu_utilization_percent.toFixed(1) : 'N/A';
            gpuMemUsageSpan.textContent = data.gpu_mem_percent != null ? data.gpu_mem_percent.toFixed(1) : 'N/A';
            gpuMemUsedSpan.textContent = data.gpu_mem_used_mb != null ? data.gpu_mem_used_mb.toFixed(2) : '--';
            gpuMemTotalSpan.textContent = data.gpu_mem_total_mb != null ? data.gpu_mem_total_mb.toFixed(2) : '--';

            const gpuUtilP = gpuUtilUsageSpan.closest('p');
            const gpuMemP = gpuMemUsageSpan.closest('p');
            const gpuErrorP = document.getElementById('gpuError');

            if (gpuUtilP) gpuUtilP.style.display = gpuAvailable ? 'block' : 'none';
            if (gpuMemP) gpuMemP.style.display = gpuAvailable ? 'block' : 'none';

            if (data.gpu_error) {
                if (gpuErrorP) {
                    gpuErrorP.textContent = `GPU Info: ${data.gpu_error}`;
                    gpuErrorP.style.display = 'block';
                }
            } else {
                if (gpuErrorP) gpuErrorP.style.display = 'none';
            }

            resourceUsageDiv.style.display = 'block';
        } else {
            console.error('Failed to fetch resource usage:', response.status, response.statusText, await response.text());
        }
    } catch (error) {
        console.error('Error fetching resource usage:', error);
    }
};

const stopResourcePolling = () => {
    if (resourcePollInterval) {
        resourcePollEnabled = false;
        clearInterval(resourcePollInterval);
        resourcePollInterval = null;
        console.log("Stopped resource polling.");
        cpuUsageSpan.textContent = '--';
        memUsageSpan.textContent = '--';
        processMemUsageSpan.textContent = '--';
        gpuUtilUsageSpan.textContent = 'N/A';
        gpuMemUsageSpan.textContent = 'N/A';
        gpuMemUsedSpan.textContent = '--';
        gpuMemTotalSpan.textContent = '--';
        resourceUsageDiv.style.display = 'none';
    }
};


form.addEventListener('submit', async (event) => {
    event.preventDefault();
    summaryResultDiv.style.display = 'none';
    errorMessageDiv.textContent = '';
    errorMessageDiv.style.display = 'none';
    downloadSummaryLink.style.display = 'none';
    downloadTransriptionLink.style.display = 'none';
    loadingIndicator.style.display = 'block';
    resourceUsageDiv.style.display = 'none';
    stopResourcePolling();

    resourcePollEnabled = true;
    resourcePollInterval = setInterval(fetchResourceUsage, 1000);
    console.log("Started resource polling.");


    const formData = new FormData();
    formData.append('audio_file', audioFileInput.files[0]);

    try {
        const response = await fetch('/summarize', {
            method: 'POST',
            body: formData,
        });
        
        stopResourcePolling();
        loadingIndicator.style.display = 'none';
        const result = await response.json();

        if (response.ok) {
            summaryTextP.innerHTML = result.summary ? marked.parse(result.summary) : "No summary generated (transcription might have been empty).";
            summaryResultDiv.style.display = 'block';
            downloadSummaryLink.style.display = 'inline-block';
            downloadTransriptionLink.style.display = 'inline-block';
            window.open('/download/summary', '_blank');
        } else {
            errorMessageDiv.textContent = `Error: ${result.error || response.statusText}`;
            errorMessageDiv.style.display = 'block';
        }
    } catch (error) {
        stopResourcePolling();
        loadingIndicator.style.display = 'none';
        errorMessageDiv.textContent = `Network or server error: ${error.message}`;
        errorMessageDiv.style.display = 'block';
        console.error('Error submitting form:', error);
    }
});

window.addEventListener('beforeunload', function(event) {
    if (navigator.sendBeacon) {
        navigator.sendBeacon('/shutdown', new Blob());
        console.log("Beacon sent to /shutdown");
    } else {
        var xhr = new XMLHttpRequest();
        xhr.open('POST', '/shutdown', false);
        xhr.send(null);
        console.log("XHR sent to /shutdown (fallback)");
    }
});
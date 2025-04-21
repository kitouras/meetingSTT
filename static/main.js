const form = document.getElementById('uploadForm');
const audioFileInput = document.getElementById('audioFile');
const summaryResultDiv = document.getElementById('summaryResult');
const summaryTextP = document.getElementById('summaryText');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorMessageDiv = document.getElementById('errorMessage');
const resourceUsageDiv = document.getElementById('resourceUsage');
const cpuUsageSpan = document.getElementById('cpuUsage');
const memUsageSpan = document.getElementById('memUsage');
const processMemUsageSpan = document.getElementById('processMemUsage');
const gpuUtilUsageSpan = document.getElementById('gpuUtilUsage');
const gpuMemUsageSpan = document.getElementById('gpuMemUsage');
const gpuMemUsedSpan = document.getElementById('gpuMemUsed');
const gpuMemTotalSpan = document.getElementById('gpuMemTotal');


let resourcePollInterval = null;

const fetchResourceUsage = async () => {
    try {
        const response = await fetch('/resources');
        if (response.ok) {
            const data = await response.json();
            cpuUsageSpan.textContent = data.cpu_percent != null ? data.cpu_percent.toFixed(1) : '--';
            memUsageSpan.textContent = data.mem_percent != null ? data.mem_percent.toFixed(1) : '--';
            processMemUsageSpan.textContent = data.process_mem_mb != null ? data.process_mem_mb.toFixed(2) : '--';

            gpuUtilUsageSpan.textContent = data.gpu_utilization_percent != null ? data.gpu_utilization_percent.toFixed(1) : 'N/A';
            gpuMemUsageSpan.textContent = data.gpu_mem_percent != null ? data.gpu_mem_percent.toFixed(1) : 'N/A';
            gpuMemUsedSpan.textContent = data.gpu_mem_used_mb != null ? data.gpu_mem_used_mb.toFixed(2) : '--';
            gpuMemTotalSpan.textContent = data.gpu_mem_total_mb != null ? data.gpu_mem_total_mb.toFixed(2) : '--';

            gpuUtilUsageSpan.closest('p').style.display = data.gpu_utilization_percent != null ? 'block' : 'none';
            gpuMemUsageSpan.closest('p').style.display = data.gpu_mem_percent != null ? 'block' : 'none';


            resourceUsageDiv.style.display = 'block';
        } else {
            console.error('Failed to fetch resource usage:', response.statusText);
        }
    } catch (error) {
        console.error('Error fetching resource usage:', error);
    }
};

const stopResourcePolling = () => {
    if (resourcePollInterval) {
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
    loadingIndicator.style.display = 'block';
    resourceUsageDiv.style.display = 'none';
    stopResourcePolling();

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
            summaryTextP.textContent = result.summary || "No summary generated (transcription might have been empty).";
            summaryResultDiv.style.display = 'block';
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
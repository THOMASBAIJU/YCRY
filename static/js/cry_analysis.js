// --- UI & AJAX Logic ---
function updateFileName() {
    const input = document.getElementById('fileInput');
    const display = document.getElementById('fileNameDisplay');
    const previewContainer = document.getElementById('audio-preview-container');
    const audioPreview = document.getElementById('audio-preview');

    if (input.files.length > 0) {
        display.innerHTML = '<span class="text-warm-teal">✅ Ready:</span> <br><span id="safe-filename"></span>';
        document.getElementById('safe-filename').textContent = input.files[0].name;

        // Set Audio Preview
        const fileURL = URL.createObjectURL(input.files[0]);
        audioPreview.src = fileURL;
        previewContainer.classList.remove('hidden');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('cry-analysis-form');
    if (!form) return;

    async function postCryAnalysis(formData, retries = 1) {
        for (let attempt = 0; attempt <= retries; attempt++) {
            try {
                return await fetch('/cry', {
                    method: 'POST',
                    body: formData
                });
            } catch (err) {
                const isLastAttempt = attempt === retries;
                if (isLastAttempt) {
                    throw err;
                }
                // Small delay allows backend cold-start/warmup to complete.
                await new Promise(resolve => setTimeout(resolve, 1500));
            }
        }
    }

    const resultContainer = document.getElementById('result-container');
    const resultContent = document.getElementById('result-content');
    const resultContentEmpty = document.getElementById('result-content-empty');

    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        const fileInput = document.getElementById('fileInput');
        if (fileInput.files.length === 0) {
            showToast("⚠️ Please upload a file or record audio first!", 'error');
            return;
        }

        // Show Loading State
        const btn = form.querySelector('button[type="submit"]');
        const originalBtnContent = btn.innerHTML;
        btn.innerHTML = '<span class="animate-spin inline-block mr-2">⏳</span> Analyzing...';
        btn.disabled = true;

        const formData = new FormData(form);

        try {
            const response = await postCryAnalysis(formData, 1);

            const text = await response.text();
            let data;

            try {
                data = JSON.parse(text);
            } catch (jsonErr) {
                console.error("Server Response (Not JSON):", text);
                let msg = "Server returned non-JSON response.";
                if (text.includes("<title>")) {
                    const matches = text.match(/<title>(.*?)<\/title>/);
                    if (matches) msg += " Title: " + matches[1];
                } else {
                    msg += " Preview: " + text.substring(0, 50) + "...";
                }
                showToast("Server Error: " + msg, 'error');
                return;
            }

            if (data.redirect) {
                window.location.href = data.redirect;
                return;
            }

            if (data.error) {
                showToast("Error: " + data.error, 'error');
            } else if (data.success) {
                // Update UI with Results
                resultContentEmpty.classList.add('hidden');
                resultContent.classList.remove('hidden');

                // Map Prediction to Icon
                const icons = {
                    "Hunger": "🍼",
                    "Pain": "🩹",
                    "Burping": "🫧",
                    "Discomfort": "🧸",
                    "Tired": "🌙"
                };
                const icon = icons[data.prediction] || "👶";

                document.getElementById('res-icon').innerText = icon;
                document.getElementById('res-pred').innerText = data.prediction;
                document.getElementById('res-conf').innerText = data.confidence + "%";
                document.getElementById('res-advice').innerText = data.advice;

                // Scroll to results using Lenis if available, or native
                if (typeof lenis !== 'undefined') {
                    lenis.scrollTo('#result-container', { offset: -100 });
                } else {
                    resultContainer.scrollIntoView({ behavior: 'smooth' });
                }
            }

        } catch (error) {
            console.error("Error:", error);
            showToast("Network/Client Error: " + error.message, 'error');
        } finally {
            // Reset Button
            btn.innerHTML = originalBtnContent;
            btn.disabled = false;
        }
    });

    function getSupportedMimeType() {
        const types = [
            'audio/webm',
            'audio/mp4',
            'audio/aac',
            'audio/ogg'
        ];
        // Ensure MediaRecorder is defined before checking
        if (typeof MediaRecorder !== 'undefined') {
            for (let t of types) {
                if (MediaRecorder.isTypeSupported(t)) {
                    return t;
                }
            }
        }
        return ''; // Fallback, let browser choose default
    }

    // Export toggleRecording and setMode globally
    window.setMode = function (mode) {
        const uploadDiv = document.getElementById('mode-upload');
        const recordDiv = document.getElementById('mode-record');
        const btnUpload = document.getElementById('btn-mode-upload');
        const btnRecord = document.getElementById('btn-mode-record');

        if (mode === 'upload') {
            uploadDiv.classList.remove('hidden');
            recordDiv.classList.add('hidden');

            btnUpload.className = "flex-1 py-2 rounded-lg text-sm font-bold transition-all bg-white shadow-sm text-warm-teal";
            btnRecord.className = "flex-1 py-2 rounded-lg text-sm font-bold transition-all text-warm-gray hover:bg-white/50";
        } else {
            uploadDiv.classList.add('hidden');
            recordDiv.classList.remove('hidden');

            btnRecord.className = "flex-1 py-2 rounded-lg text-sm font-bold transition-all bg-white shadow-sm text-warm-teal";
            btnUpload.className = "flex-1 py-2 rounded-lg text-sm font-bold transition-all text-warm-gray hover:bg-white/50";
        }
    };

    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let timerInterval;
    let startTime;

    window.toggleRecording = async function () {
        const btn = document.getElementById('recordBtn');
        const status = document.getElementById('recordStatus');
        const visual = document.getElementById('recording-visual');
        const timer = document.getElementById('timer');

        if (!isRecording) {
            // START Recording
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const mimeType = getSupportedMimeType();
                const options = mimeType ? { mimeType: mimeType } : undefined;

                // Note: File extension depends on format
                const extension = mimeType.includes('mp4') ? 'mp4' : (mimeType.includes('aac') ? 'aac' : 'webm');

                mediaRecorder = new MediaRecorder(stream, options);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, options);
                    const audioFile = new File([audioBlob], `recorded_cry.${extension}`, { type: mimeType || "audio/webm" });

                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(audioFile);
                    document.getElementById('fileInput').files = dataTransfer.files;

                    status.innerHTML = "<span class='text-warm-teal'>✅ Recorded!</span> Click Analyze Below";
                    btn.classList.remove('animate-pulse');
                    visual.classList.add('hidden');
                    clearInterval(timerInterval);

                    // Set Audio Preview
                    const audioURL = URL.createObjectURL(audioBlob);
                    const audioPreview = document.getElementById('audio-preview');
                    const previewContainer = document.getElementById('audio-preview-container');
                    audioPreview.src = audioURL;
                    previewContainer.classList.remove('hidden');
                };

                mediaRecorder.start();
                isRecording = true;

                btn.innerHTML = "⬛";
                btn.classList.add('animate-pulse', 'bg-warm-dark');
                btn.classList.remove('bg-red-500');
                status.innerText = "Recording...";
                visual.classList.remove('hidden');
                timer.classList.remove('hidden');

                startTime = Date.now();
                timerInterval = setInterval(() => {
                    const diff = Date.now() - startTime;
                    const secs = Math.floor(diff / 1000);
                    const mins = Math.floor(secs / 60);
                    timer.innerText = `${mins.toString().padStart(2, '0')}:${(secs % 60).toString().padStart(2, '0')}`;
                }, 1000);

            } catch (err) {
                console.error("Mic Error:", err);
                showToast("Could not access microphone: " + err.message, 'error');
            }
        } else {
            // STOP Recording
            mediaRecorder.stop();
            isRecording = false;

            btn.innerHTML = "🎙️";
            btn.classList.remove('animate-pulse', 'bg-warm-dark');
            btn.classList.add('bg-red-500');
            timer.classList.add('hidden');

            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    };
});

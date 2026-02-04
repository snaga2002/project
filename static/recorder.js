let mediaRecorder;
let audioChunks = [];

async function startRecording() {
    let stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();
    
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
}

function stopRecording() {
    mediaRecorder.stop();
    mediaRecorder.onstop = async () => {
        let audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        audioChunks = [];

        let reader = new FileReader();
        reader.readAsDataURL(audioBlob);
        reader.onloadend = function() {
            document.getElementById("audioData").value = reader.result;
            let audioURL = URL.createObjectURL(audioBlob);
            document.getElementById("audioPlayback").src = audioURL;
        }
    };
}

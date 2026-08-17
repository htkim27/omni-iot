const startButton = document.querySelector("#startButton");
const stopButton = document.querySelector("#stopButton");
const resetButton = document.querySelector("#resetButton");
const stateDot = document.querySelector("#stateDot");
const stateText = document.querySelector("#stateText");
const meterBar = document.querySelector("#meterBar");
const replyAudio = document.querySelector("#replyAudio");
const turnList = document.querySelector("#turnList");
const thresholdSlider = document.querySelector("#thresholdSlider");
const thresholdValue = document.querySelector("#thresholdValue");

const preRollSeconds = 0.45;
const silenceEndMs = 850;
const maxTurnMs = 14000;

let audioContext;
let analyser;
let source;
let processor;
let zeroGain;
let stream;
let animationFrame;
let sessionId = localStorage.getItem("omni_iot_session_id");
let state = "idle";
let threshold = Number(thresholdSlider.value) / 100;
let preRollChunks = [];
let preRollLength = 0;
let turnChunks = [];
let turnLength = 0;
let speechStartedAt = 0;
let lastVoiceAt = 0;
let latestRms = 0;

thresholdValue.value = threshold.toFixed(2);

startButton.addEventListener("click", startHarness);
stopButton.addEventListener("click", stopHarness);
resetButton.addEventListener("click", resetSession);
thresholdSlider.addEventListener("input", () => {
  threshold = Number(thresholdSlider.value) / 100;
  thresholdValue.value = threshold.toFixed(2);
});

async function startHarness() {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  audioContext = new AudioContext({ sampleRate: 16000 });
  source = audioContext.createMediaStreamSource(stream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 512;
  processor = audioContext.createScriptProcessor(2048, 1, 1);
  zeroGain = audioContext.createGain();
  zeroGain.gain.value = 0;

  processor.onaudioprocess = handleAudioFrame;
  source.connect(analyser);
  source.connect(processor);
  processor.connect(zeroGain);
  zeroGain.connect(audioContext.destination);

  startButton.disabled = true;
  stopButton.disabled = false;
  resetTurnBuffers();
  setState("listening", "Listening");
  drawMeter();
}

async function stopHarness() {
  cancelAnimationFrame(animationFrame);
  window.speechSynthesis?.cancel();
  replyAudio.pause();
  replyAudio.removeAttribute("src");

  processor?.disconnect();
  analyser?.disconnect();
  source?.disconnect();
  zeroGain?.disconnect();
  stream?.getTracks().forEach((track) => track.stop());
  await audioContext?.close();

  startButton.disabled = false;
  stopButton.disabled = true;
  resetTurnBuffers();
  setState("idle", "Idle");
}

function handleAudioFrame(event) {
  if (!audioContext || state === "processing") {
    return;
  }

  const input = new Float32Array(event.inputBuffer.getChannelData(0));
  const now = performance.now();
  latestRms = rms(input);
  pushPreRoll(input);

  if (state === "speaking" && latestRms >= threshold * 1.4) {
    stopAssistantAudio();
    beginSpeech(now);
  }

  if (state === "listening" && latestRms >= threshold) {
    beginSpeech(now);
  }

  if (state === "recording") {
    turnChunks.push(input);
    turnLength += input.length;

    if (latestRms >= threshold * 0.72) {
      lastVoiceAt = now;
    }

    const silenceElapsed = now - lastVoiceAt;
    const turnElapsed = now - speechStartedAt;
    if (silenceElapsed >= silenceEndMs || turnElapsed >= maxTurnMs) {
      void finishSpeech();
    }
  }
}

function beginSpeech(now) {
  turnChunks = [...preRollChunks];
  turnLength = preRollLength;
  speechStartedAt = now;
  lastVoiceAt = now;
  setState("recording", "Recording");
}

async function finishSpeech() {
  if (state !== "recording" || turnLength === 0) {
    return;
  }

  setState("processing", "Thinking");
  const wavBlob = encodeWav(turnChunks, turnLength, audioContext.sampleRate);
  resetTurnBuffers();
  await sendTurn(wavBlob);
}

async function sendTurn(wavBlob) {
  const userTurn = addTurn("user", "Voice turn");

  try {
    const response = await fetch("/api/turn", {
      method: "POST",
      headers: {
        "Content-Type": "audio/wav",
        ...(sessionId ? { "X-Session-Id": sessionId } : {}),
      },
      body: wavBlob,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed");
    }

    sessionId = payload.session_id;
    localStorage.setItem("omni_iot_session_id", sessionId);
    if (payload.user_text) {
      updateTurn(userTurn, payload.user_text);
    }
    addTurn("assistant", payload.text, payload.timings);
    await playReply(payload);
  } catch (error) {
    addTurn("system", error.message);
    setState("listening", "Listening");
  }
}

async function playReply(payload) {
  if (payload.audio_url) {
    setState("speaking", "Speaking");
    replyAudio.src = payload.audio_url;
    replyAudio.currentTime = 0;
    replyAudio.onended = () => setState("listening", "Listening");
    await replyAudio.play();
    return;
  }

  if ("speechSynthesis" in window) {
    setState("speaking", payload.used_mock_omni ? "Mock speaking" : "Speaking");
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(payload.text);
    utterance.lang = "en-US";
    utterance.onend = () => setState("listening", "Listening");
    window.speechSynthesis.speak(utterance);
    return;
  }

  setState("listening", "Listening");
}

async function resetSession() {
  const response = await fetch("/api/session/reset", {
    method: "POST",
    headers: sessionId ? { "X-Session-Id": sessionId } : {},
  });
  const payload = await response.json();
  sessionId = payload.session_id;
  localStorage.setItem("omni_iot_session_id", sessionId);
  turnList.replaceChildren();
}

function stopAssistantAudio() {
  replyAudio.pause();
  replyAudio.removeAttribute("src");
  window.speechSynthesis?.cancel();
}

function pushPreRoll(input) {
  preRollChunks.push(input);
  preRollLength += input.length;

  const maxLength = Math.floor((audioContext?.sampleRate || 16000) * preRollSeconds);
  while (preRollLength > maxLength && preRollChunks.length > 0) {
    const removed = preRollChunks.shift();
    preRollLength -= removed.length;
  }
}

function resetTurnBuffers() {
  preRollChunks = [];
  preRollLength = 0;
  turnChunks = [];
  turnLength = 0;
}

function addTurn(role, text, timings = null) {
  const item = document.createElement("li");
  item.className = `turn ${role}`;

  const roleLabel = document.createElement("span");
  roleLabel.className = "role";
  roleLabel.textContent = role;

  const content = document.createElement("p");
  content.textContent = timings ? `${text}  (${timings.total_seconds}s)` : text;

  item.append(roleLabel, content);
  turnList.append(item);
  item.scrollIntoView({ block: "end", behavior: "smooth" });
  return item;
}

function updateTurn(item, text) {
  const content = item?.querySelector("p");
  if (content) {
    content.textContent = text;
  }
}

function drawMeter() {
  const tick = () => {
    meterBar.style.transform = `scaleX(${Math.min(1, latestRms / 0.12)})`;
    animationFrame = requestAnimationFrame(tick);
  };
  tick();
}

function setState(nextState, label) {
  state = nextState;
  stateDot.className = `dot ${nextState}`;
  stateText.textContent = label;
}

function rms(buffer) {
  let sum = 0;
  for (let i = 0; i < buffer.length; i += 1) {
    sum += buffer[i] * buffer[i];
  }
  return Math.sqrt(sum / buffer.length);
}

function encodeWav(channelBuffers, length, sampleRate) {
  const samples = mergeBuffers(channelBuffers, length);
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  floatTo16BitPcm(view, 44, samples);
  return new Blob([view], { type: "audio/wav" });
}

function mergeBuffers(buffers, length) {
  const result = new Float32Array(length);
  let offset = 0;
  for (const buffer of buffers) {
    result.set(buffer, offset);
    offset += buffer.length;
  }
  return result;
}

function floatTo16BitPcm(view, offset, input) {
  for (let i = 0; i < input.length; i += 1, offset += 2) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i += 1) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

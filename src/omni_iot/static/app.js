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
const silenceSlider = document.querySelector("#silenceSlider");
const silenceValue = document.querySelector("#silenceValue");
const responseTokensSlider = document.querySelector("#responseTokensSlider");
const responseTokensValue = document.querySelector("#responseTokensValue");
const ttsStepsSlider = document.querySelector("#ttsStepsSlider");
const ttsStepsValue = document.querySelector("#ttsStepsValue");

const TARGET_SAMPLE_RATE = 16000;

let preRollSeconds = 0.45;
let silenceEndMs = Number(silenceSlider.value);
let maxTurnMs = 14000;
let bargeInMultiplier = 1.4;
let continueMultiplier = 0.72;
let followUpTimeoutMs = 8000;
let wakeWordLabel = "Hey Jarvis";
let responseTokenLimit = Number(responseTokensSlider.value);
let ttsNumSteps = Number(ttsStepsSlider.value);

let audioContext;
let source;
let processor;
let zeroGain;
let stream;
let socket;
let resampler;
let animationFrame;
let followUpTimer;
let commandWaitTimer;
let sessionId = localStorage.getItem("omni_iot_session_id");
let state = "idle";
let threshold = Number(thresholdSlider.value) / 100;
let preRollChunks = [];
let preRollLength = 0;
let speechStartedAt = 0;
let lastVoiceAt = 0;
let latestRms = 0;
let pendingUserTurn;
let lastUtteranceVoiceAt = 0;
let awaitingCommandVoice = false;
let commandArmed = false;
let stopping = false;

thresholdValue.value = threshold.toFixed(2);

startButton.addEventListener("click", startHarness);
stopButton.addEventListener("click", stopHarness);
resetButton.addEventListener("click", resetSession);
thresholdSlider.addEventListener("input", () => {
  threshold = Number(thresholdSlider.value) / 100;
  thresholdValue.value = threshold.toFixed(2);
  saveSetting("vad_threshold", thresholdSlider.value);
});
silenceSlider.addEventListener("input", () => {
  silenceEndMs = Number(silenceSlider.value);
  silenceValue.value = `${silenceEndMs}ms`;
  saveSetting("vad_silence_end_ms", silenceSlider.value);
});
responseTokensSlider.addEventListener("input", () => {
  responseTokenLimit = Number(responseTokensSlider.value);
  responseTokensValue.value = `${responseTokenLimit} tokens`;
  saveSetting("response_token_limit", responseTokensSlider.value);
});
ttsStepsSlider.addEventListener("input", () => {
  ttsNumSteps = Number(ttsStepsSlider.value);
  ttsStepsValue.value = `${ttsNumSteps} steps`;
  saveSetting("tts_num_steps", ttsStepsSlider.value);
});

void initializeSettings();

async function startHarness() {
  stopping = false;
  startButton.disabled = true;
  setState("connecting", "Connecting");

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    audioContext = new AudioContext({ latencyHint: "interactive" });
    await audioContext.resume();
    resampler = new Pcm16Resampler(audioContext.sampleRate, TARGET_SAMPLE_RATE);
    source = audioContext.createMediaStreamSource(stream);
    processor = audioContext.createScriptProcessor(2048, 1, 1);
    zeroGain = audioContext.createGain();
    zeroGain.gain.value = 0;

    processor.onaudioprocess = handleAudioFrame;
    source.connect(processor);
    processor.connect(zeroGain);
    zeroGain.connect(audioContext.destination);

    await connectAudioSocket();
    stopButton.disabled = false;
    resetAudioState();
    setState("sleeping", `Say “${wakeWordLabel}”`);
    drawMeter();
  } catch (error) {
    addTurn("system", error.message || String(error));
    await stopHarness();
  }
}

function connectAudioSocket() {
  return new Promise((resolve, reject) => {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${scheme}://${window.location.host}/ws/audio`);

    socket.onopen = () => {
      socket.send(JSON.stringify({
        type: "start",
        sample_rate: TARGET_SAMPLE_RATE,
        session_id: sessionId,
        max_response_tokens: responseTokenLimit,
        tts_num_steps: ttsNumSteps,
      }));
      resolve();
    };
    socket.onmessage = handleServerMessage;
    socket.onerror = () => reject(new Error("Audio WebSocket connection failed."));
    socket.onclose = () => {
      if (!stopping && state !== "idle") {
        addTurn("system", "Audio connection closed. Press Start to reconnect.");
        void stopHarness();
      }
    };
  });
}

async function stopHarness() {
  stopping = true;
  clearTimeout(followUpTimer);
  clearTimeout(commandWaitTimer);
  cancelAnimationFrame(animationFrame);
  window.speechSynthesis?.cancel();
  replyAudio.pause();
  replyAudio.removeAttribute("src");

  if (socket?.readyState === WebSocket.OPEN) {
    socket.close(1000, "client stopped");
  }
  processor?.disconnect();
  source?.disconnect();
  zeroGain?.disconnect();
  stream?.getTracks().forEach((track) => track.stop());
  await audioContext?.close();

  socket = undefined;
  processor = undefined;
  source = undefined;
  zeroGain = undefined;
  stream = undefined;
  audioContext = undefined;
  startButton.disabled = false;
  stopButton.disabled = true;
  resetAudioState();
  setState("idle", "Idle");
}

function handleAudioFrame(event) {
  if (!audioContext || !socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }

  const input = event.inputBuffer.getChannelData(0);
  const pcm = resampler.process(input);
  if (pcm.length === 0) {
    return;
  }

  const now = performance.now();
  latestRms = rms(input);
  pushPreRoll(pcm);
  if (latestRms >= threshold * continueMultiplier) {
    lastVoiceAt = now;
  }

  if (state === "sleeping") {
    sendPcm(pcm);
    return;
  }

  if (state === "speaking" && latestRms >= threshold * bargeInMultiplier) {
    stopAssistantAudio();
    beginActiveSpeech(now, "Recording interruption");
    return;
  }

  if (state === "follow_up" && latestRms >= threshold) {
    beginActiveSpeech(now, "Recording follow-up");
    return;
  }

  if (state !== "recording") {
    return;
  }

  if (awaitingCommandVoice) {
    if (!commandArmed) {
      if (latestRms < threshold * continueMultiplier) {
        commandArmed = true;
        resetPreRoll();
      }
      return;
    }
    if (latestRms < threshold) {
      return;
    }
    awaitingCommandVoice = false;
    commandArmed = false;
    clearTimeout(commandWaitTimer);
    speechStartedAt = now;
    lastVoiceAt = now;
    for (const chunk of preRollChunks) {
      sendPcm(chunk);
    }
    resetPreRoll();
    setState("recording", "Recording command");
    return;
  }

  sendPcm(pcm);
  const silenceElapsed = now - lastVoiceAt;
  const turnElapsed = now - speechStartedAt;
  if (silenceElapsed >= silenceEndMs || turnElapsed >= maxTurnMs) {
    finishSpeech();
  }
}

function beginActiveSpeech(now, label) {
  clearTimeout(followUpTimer);
  clearTimeout(commandWaitTimer);
  awaitingCommandVoice = false;
  commandArmed = false;
  socket.send(JSON.stringify({ type: "speech_started" }));
  setState("recording", label);
  speechStartedAt = now;
  lastVoiceAt = now;
  for (const chunk of preRollChunks) {
    sendPcm(chunk);
  }
}

function finishSpeech() {
  if (state !== "recording") {
    return;
  }
  clearTimeout(commandWaitTimer);
  awaitingCommandVoice = false;
  commandArmed = false;
  lastUtteranceVoiceAt = lastVoiceAt;
  pendingUserTurn = addTurn("user", "Voice turn");
  setState("processing", "Thinking");
  socket.send(JSON.stringify({ type: "speech_ended" }));
  resetPreRoll();
}

function handleServerMessage(event) {
  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch (error) {
    console.warn("Invalid server event", error);
    return;
  }

  if (payload.session_id) {
    sessionId = payload.session_id;
    localStorage.setItem("omni_iot_session_id", sessionId);
  }

  if (payload.type === "wake_detected") {
    clearTimeout(followUpTimer);
    clearTimeout(commandWaitTimer);
    awaitingCommandVoice = true;
    commandArmed = false;
    speechStartedAt = 0;
    lastVoiceAt = 0;
    resetPreRoll();
    setState("recording", "Listening for command");
    commandWaitTimer = setTimeout(() => {
      if (state === "recording" && awaitingCommandVoice) {
        socket?.send(JSON.stringify({ type: "sleep" }));
        resetAudioState();
        setState("sleeping", `Say “${wakeWordLabel}”`);
      }
    }, followUpTimeoutMs);
  } else if (payload.type === "state") {
    applyServerState(payload.state);
  } else if (payload.type === "reply_audio") {
    if (payload.user_text) {
      updateTurn(pendingUserTurn, payload.user_text);
    }
    pendingUserTurn = undefined;
    addTurn("assistant", payload.text, payload.timings);
    void playReply(payload);
  } else if (payload.type === "session_reset") {
    turnList.replaceChildren();
  } else if (payload.type === "error") {
    addTurn("system", payload.message || "Audio pipeline error");
  }
}

function applyServerState(serverState) {
  if (serverState === "processing") {
    if (!pendingUserTurn) {
      pendingUserTurn = addTurn("user", "Voice turn");
    }
    setState("processing", "Thinking");
  } else if (serverState === "recording") {
    setState("recording", "Recording");
  } else if (serverState === "sleeping") {
    clearTimeout(followUpTimer);
    resetAudioState();
    setState("sleeping", `Say “${wakeWordLabel}”`);
  } else if (serverState === "follow_up" && state !== "speaking") {
    beginFollowUpWindow();
  }
}

async function playReply(payload) {
  setState("speaking", "Speaking");
  if (payload.url || payload.audio_url) {
    replyAudio.src = payload.url || payload.audio_url;
    replyAudio.currentTime = 0;
    replyAudio.onended = finishAssistantReply;
    replyAudio.onplaying = () => {
      if (!lastUtteranceVoiceAt) {
        return;
      }
      const speechEndToAudioStartSeconds =
        (performance.now() - lastUtteranceVoiceAt) / 1000;
      socket?.send(JSON.stringify({
        type: "playback_started",
        turn_id: payload.turn_id,
        browser_speech_end_to_audio_start_seconds:
          Number(speechEndToAudioStartSeconds.toFixed(3)),
      }));
      lastUtteranceVoiceAt = 0;
      replyAudio.onplaying = null;
    };
    try {
      await replyAudio.play();
    } catch (error) {
      addTurn("system", `Reply playback failed: ${error.message}`);
      finishAssistantReply();
    }
    return;
  }

  if ("speechSynthesis" in window) {
    const utterance = new SpeechSynthesisUtterance(payload.text);
    utterance.lang = "ko-KR";
    utterance.onend = finishAssistantReply;
    utterance.onerror = finishAssistantReply;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
    return;
  }

  finishAssistantReply();
}

function finishAssistantReply() {
  if (state !== "speaking") {
    return;
  }
  socket?.send(JSON.stringify({ type: "reply_ended" }));
  beginFollowUpWindow();
}

function stopAssistantAudio() {
  replyAudio.onended = null;
  replyAudio.onplaying = null;
  replyAudio.pause();
  replyAudio.removeAttribute("src");
  window.speechSynthesis?.cancel();
}

function beginFollowUpWindow() {
  clearTimeout(followUpTimer);
  resetPreRoll();
  setState("follow_up", "Listening for follow-up");
  followUpTimer = setTimeout(() => {
    if (state === "follow_up") {
      socket?.send(JSON.stringify({ type: "sleep" }));
      setState("sleeping", `Say “${wakeWordLabel}”`);
      resetAudioState();
    }
  }, followUpTimeoutMs);
}

async function resetSession() {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "reset_session" }));
    return;
  }

  const response = await fetch("/api/session/reset", {
    method: "POST",
    headers: sessionId ? { "X-Session-Id": sessionId } : {},
  });
  const payload = await response.json();
  sessionId = payload.session_id;
  localStorage.setItem("omni_iot_session_id", sessionId);
  turnList.replaceChildren();
}

async function initializeSettings() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) {
      throw new Error("Failed to load client config");
    }
    const config = await response.json();
    preRollSeconds = Number(config.vad.pre_roll_ms) / 1000;
    maxTurnMs = Number(config.vad.max_turn_ms);
    bargeInMultiplier = Number(config.vad.barge_in_multiplier);
    continueMultiplier = Number(config.vad.continue_multiplier);
    followUpTimeoutMs = Number(config.wakeword.follow_up_timeout_ms);
    wakeWordLabel = config.wakeword.label || wakeWordLabel;

    setControl(thresholdSlider, "vad_threshold", Math.round(Number(config.vad.threshold) * 100));
    setControl(silenceSlider, "vad_silence_end_ms", config.vad.silence_end_ms);
    setControl(responseTokensSlider, "response_token_limit", config.generation.max_response_tokens);
    setControl(ttsStepsSlider, "tts_num_steps", config.generation.tts_num_steps);
  } catch (error) {
    console.warn(error);
  }

  threshold = Number(thresholdSlider.value) / 100;
  silenceEndMs = Number(silenceSlider.value);
  responseTokenLimit = Number(responseTokensSlider.value);
  ttsNumSteps = Number(ttsStepsSlider.value);
  thresholdValue.value = threshold.toFixed(2);
  silenceValue.value = `${silenceEndMs}ms`;
  responseTokensValue.value = `${responseTokenLimit} tokens`;
  ttsStepsValue.value = `${ttsNumSteps} steps`;
}

function setControl(control, key, serverDefault) {
  const saved = localStorage.getItem(`omni_iot_${key}`);
  control.value = saved ?? String(serverDefault);
}

function saveSetting(key, value) {
  localStorage.setItem(`omni_iot_${key}`, String(value));
}

function sendPcm(pcm) {
  socket.send(pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength));
}

function pushPreRoll(input) {
  preRollChunks.push(input);
  preRollLength += input.length;
  const maxLength = Math.floor(TARGET_SAMPLE_RATE * preRollSeconds);
  while (preRollLength > maxLength && preRollChunks.length > 0) {
    const removed = preRollChunks.shift();
    preRollLength -= removed.length;
  }
}

function resetPreRoll() {
  preRollChunks = [];
  preRollLength = 0;
}

function resetAudioState() {
  clearTimeout(commandWaitTimer);
  awaitingCommandVoice = false;
  commandArmed = false;
  resetPreRoll();
  resampler?.reset();
  speechStartedAt = 0;
  lastVoiceAt = 0;
  latestRms = 0;
  pendingUserTurn = undefined;
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

class Pcm16Resampler {
  constructor(inputRate, outputRate) {
    this.ratio = inputRate / outputRate;
    this.pending = new Float32Array(0);
    this.position = 0;
  }

  process(input) {
    const joined = new Float32Array(this.pending.length + input.length);
    joined.set(this.pending);
    joined.set(input, this.pending.length);
    const samples = [];
    while (this.position + 1 < joined.length) {
      const left = Math.floor(this.position);
      const fraction = this.position - left;
      const value = joined[left] + (joined[left + 1] - joined[left]) * fraction;
      samples.push(Math.max(-1, Math.min(1, value)));
      this.position += this.ratio;
    }

    const consumed = Math.min(Math.floor(this.position), joined.length - 1);
    this.pending = joined.slice(consumed);
    this.position -= consumed;
    const output = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i += 1) {
      output[i] = samples[i] < 0 ? samples[i] * 0x8000 : samples[i] * 0x7fff;
    }
    return output;
  }

  reset() {
    this.pending = new Float32Array(0);
    this.position = 0;
  }
}

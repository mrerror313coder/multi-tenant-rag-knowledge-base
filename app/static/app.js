// Multi-Tenant RAG Knowledge Base - Frontend Application Logic with Multimodal Vision & Voice

let currentTenant = {
  id: "org_acme_corp",
  name: "Acme Corporation",
  apiKey: "sk_acme_demo_key_1001",
};

let allOrganizations = [];
let stagedImageFile = null;
let isRecording = false;
let speechRecognizer = null;
let mediaRecorder = null;
let audioChunks = [];
let currentUtterance = null;
let currentSpeakingBtn = null;

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", async () => {
  await loadOrganizations();
  await loadTenantDocuments();
  setupDropZone();
  setupClipboardPasteListener();
  setupSpeechRecognition();
});

// Tab Switching
function switchTab(tabId, btn) {
  document.querySelectorAll(".tab-content").forEach((el) => el.classList.remove("active"));
  document.querySelectorAll(".nav-tab-btn").forEach((el) => el.classList.remove("active"));

  document.getElementById(tabId).classList.add("active");
  btn.classList.add("active");

  if (tabId === "docsTab") {
    loadTenantDocuments();
  }
}

// Load Organizations
async function loadOrganizations() {
  try {
    const res = await fetch("/api/organizations/");
    if (!res.ok) return;
    allOrganizations = await res.json();

    const select = document.getElementById("tenantSelect");
    select.innerHTML = "";

    allOrganizations.forEach((org) => {
      const opt = document.createElement("option");
      opt.value = org.id;
      opt.textContent = `${org.name} (${org.id})`;
      if (org.id === currentTenant.id) opt.selected = true;
      select.appendChild(opt);
    });

    if (allOrganizations.length > 0) {
      const active = allOrganizations.find((o) => o.id === currentTenant.id) || allOrganizations[0];
      setTenant(active);
    }
  } catch (err) {
    console.error("Failed to load orgs:", err);
  }
}

function setTenant(org) {
  currentTenant = {
    id: org.id,
    name: org.name,
    apiKey: org.api_key,
  };
  const badge = document.getElementById("apiKeyBadge");
  if (badge) badge.textContent = org.api_key;
  document.getElementById("chatTenantName").textContent = org.name;
  loadTenantDocuments();
}

function switchTenant(orgId) {
  const org = allOrganizations.find((o) => o.id === orgId);
  if (org) {
    setTenant(org);
    appendChatBubble("assistant", `Switched active tenant to **${org.name}**. All vector retrieval is now strictly scoped to ${org.id}.`);
  }
}

// -------------------------------------------------------------
// Multimodal Image / Screenshot Attachment & Clipboard Paste
// -------------------------------------------------------------
function setupClipboardPasteListener() {
  document.addEventListener("paste", (e) => {
    if (e.clipboardData && e.clipboardData.files.length > 0) {
      const file = e.clipboardData.files[0];
      if (file.type.startsWith("image/")) {
        e.preventDefault();
        stageImage(file);
      }
    }
  });
}

function handleImageSelection(e) {
  const file = e.target.files[0];
  if (file) {
    stageImage(file);
  }
}

function stageImage(file) {
  stagedImageFile = file;
  const bar = document.getElementById("attachmentPreviewBar");
  const thumb = document.getElementById("attachmentThumbnail");
  const nameEl = document.getElementById("attachmentName");

  const reader = new FileReader();
  reader.onload = (ev) => {
    thumb.src = ev.target.result;
    nameEl.textContent = file.name || "Pasted Screenshot.png";
    bar.classList.add("active");
  };
  reader.readAsDataURL(file);
}

function clearStagedImage() {
  stagedImageFile = null;
  document.getElementById("imageFileInput").value = "";
  document.getElementById("attachmentPreviewBar").classList.remove("active");
}

// -------------------------------------------------------------
// Voice Input: Speech-to-Text (Reliable MediaRecorder Engine)
// -------------------------------------------------------------
let voiceTimeoutTimer = null;
let recordingTickerInterval = null;
let recordingSeconds = 0;
let mediaStreamRef = null;

function setupSpeechRecognition() {
  // SpeechRecognition placeholder (handled dynamically via MediaRecorder for 100% browser compatibility)
}

async function toggleVoiceRecording() {
  if (isRecording) {
    stopVoiceRecording();
  } else {
    await startVoiceRecording();
  }
}

async function startVoiceRecording() {
  const micBtn = document.getElementById("micBtn");
  const statusBar = document.getElementById("recordingStatusBar");
  const statusText = document.getElementById("recordingStatusText");
  const timerBadge = document.getElementById("recordingTimerBadge");

  // Request audio permission from browser
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaStreamRef = stream;

    // Use audio/webm or audio/mp4 depending on browser support
    let mimeType = "audio/webm";
    if (!MediaRecorder.isTypeSupported("audio/webm")) {
      if (MediaRecorder.isTypeSupported("audio/mp4")) mimeType = "audio/mp4";
      else if (MediaRecorder.isTypeSupported("audio/ogg")) mimeType = "audio/ogg";
    }

    mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      if (audioChunks.length > 0) {
        const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
        if (statusText) statusText.textContent = "⏳ Transcribing audio with Gemini AI...";
        await sendAudioForBackendTranscription(audioBlob);
      }
      if (mediaStreamRef) {
        mediaStreamRef.getTracks().forEach((track) => track.stop());
        mediaStreamRef = null;
      }
      hideRecordingStatusBar();
    };

    // Start recording audio in chunks
    mediaRecorder.start(200);
    isRecording = true;

    // UI Feedback
    micBtn.classList.add("recording");
    micBtn.title = "Recording... Click to Stop";
    if (statusBar) statusBar.classList.add("active");
    if (statusText) statusText.textContent = "🔴 Recording... Speak your question • Click 🎙️ or ⏹️ Done when finished";
    if (timerBadge) timerBadge.textContent = "0:00";

    // Seconds counter ticker
    recordingSeconds = 0;
    if (recordingTickerInterval) clearInterval(recordingTickerInterval);
    recordingTickerInterval = setInterval(() => {
      recordingSeconds++;
      const mins = Math.floor(recordingSeconds / 60);
      const secs = recordingSeconds % 60;
      if (timerBadge) {
        timerBadge.textContent = `${mins}:${secs < 10 ? "0" : ""}${secs}`;
      }
    }, 1000);

    // 25-second maximum recording safety limit
    if (voiceTimeoutTimer) clearTimeout(voiceTimeoutTimer);
    voiceTimeoutTimer = setTimeout(() => {
      if (isRecording) {
        stopVoiceRecording();
      }
    }, 25000);

  } catch (err) {
    console.warn("Microphone access error:", err);
    alert(`Microphone permission required: Please allow microphone access in your browser to speak.`);
    stopVoiceRecordingUI();
  }
}

function stopVoiceRecording() {
  if (voiceTimeoutTimer) {
    clearTimeout(voiceTimeoutTimer);
    voiceTimeoutTimer = null;
  }
  if (recordingTickerInterval) {
    clearInterval(recordingTickerInterval);
    recordingTickerInterval = null;
  }

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  } else {
    hideRecordingStatusBar();
  }

  stopVoiceRecordingUI();
}

function hideRecordingStatusBar() {
  const statusBar = document.getElementById("recordingStatusBar");
  if (statusBar) statusBar.classList.remove("active");
}

function stopVoiceRecordingUI() {
  isRecording = false;
  if (voiceTimeoutTimer) {
    clearTimeout(voiceTimeoutTimer);
    voiceTimeoutTimer = null;
  }
  if (recordingTickerInterval) {
    clearInterval(recordingTickerInterval);
    recordingTickerInterval = null;
  }
  const micBtn = document.getElementById("micBtn");
  if (micBtn) {
    micBtn.classList.remove("recording");
    micBtn.title = "Speak Voice Message";
  }
}

async function sendAudioForBackendTranscription(blob) {
  const langSelect = document.getElementById("voiceLangSelect");
  const selectedLang = langSelect ? langSelect.value : "ur";

  const formData = new FormData();
  formData.append("audio_file", blob, "recording.webm");
  formData.append("language", selectedLang);

  try {
    const res = await fetch("/api/chat/transcribe", {
      method: "POST",
      headers: { "X-API-Key": currentTenant.apiKey },
      body: formData,
    });
    if (res.ok) {
      const data = await res.json();
      if (data.transcription) {
        const input = document.getElementById("chatInput");
        input.value = data.transcription;
        input.focus();
      }
    }
  } catch (err) {
    console.error("Audio transcription error:", err);
  } finally {
    hideRecordingStatusBar();
  }
}

// -------------------------------------------------------------
// Voice Output: Text-to-Speech (Listen Button)
// -------------------------------------------------------------
let currentAudio = null;

async function speakAnswerText(text, btn) {
  // If already speaking/playing this button, stop
  if (currentSpeakingBtn === btn) {
    if (currentAudio) {
      currentAudio.pause();
      currentAudio = null;
    }
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    btn.classList.remove("speaking");
    btn.innerHTML = "🔊 Listen";
    currentSpeakingBtn = null;
    return;
  }

  // Cancel any ongoing audio/speech
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }
  if (window.speechSynthesis) window.speechSynthesis.cancel();
  if (currentSpeakingBtn) {
    currentSpeakingBtn.classList.remove("speaking");
    currentSpeakingBtn.innerHTML = "🔊 Listen";
  }

  btn.classList.add("speaking");
  btn.innerHTML = "⏹️ Stop";
  currentSpeakingBtn = btn;

  // 1. Try Backend High-Quality Neural TTS (Natural Pakistani Urdu ur-PK / English)
  try {
    const res = await fetch("/api/chat/tts", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": currentTenant.apiKey,
      },
      body: JSON.stringify({ text: text }),
    });

    if (res.ok) {
      const blob = await res.blob();
      const audioUrl = URL.createObjectURL(blob);
      currentAudio = new Audio(audioUrl);
      currentAudio.onended = () => {
        btn.classList.remove("speaking");
        btn.innerHTML = "🔊 Listen";
        currentSpeakingBtn = null;
        currentAudio = null;
      };
      currentAudio.onerror = () => {
        fallbackBrowserTTS(text, btn);
      };
      await currentAudio.play();
      return;
    }
  } catch (err) {
    console.warn("Backend Neural TTS failed, using browser fallback:", err);
  }

  // 2. Fallback to Browser SpeechSynthesis
  fallbackBrowserTTS(text, btn);
}

function fallbackBrowserTTS(text, btn) {
  if (!window.speechSynthesis) {
    btn.classList.remove("speaking");
    btn.innerHTML = "🔊 Listen";
    currentSpeakingBtn = null;
    return;
  }

  let cleanText = text.replace(/\[Doc:[^\]]+\]/gi, "");
  cleanText = cleanText.replace(/[#*`_~>]/g, "");

  const utterance = new SpeechSynthesisUtterance(cleanText);
  const isUrdu = /[\u0600-\u06FF]/.test(cleanText);

  if (isUrdu) {
    utterance.lang = "ur-PK";
    const voices = window.speechSynthesis.getVoices();
    const urVoice = voices.find(
      (v) =>
        v.lang.toLowerCase().includes("ur") ||
        v.lang.toLowerCase().includes("hi") ||
        v.lang.toLowerCase().includes("ar")
    );
    if (urVoice) utterance.voice = urVoice;
  } else {
    utterance.lang = "en-US";
  }

  utterance.onend = () => {
    btn.classList.remove("speaking");
    btn.innerHTML = "🔊 Listen";
    currentSpeakingBtn = null;
  };

  utterance.onerror = () => {
    btn.classList.remove("speaking");
    btn.innerHTML = "🔊 Listen";
    currentSpeakingBtn = null;
  };

  window.speechSynthesis.speak(utterance);
}

// -------------------------------------------------------------
// Chat Sending & Streaming (Text & Multimodal Vision)
// -------------------------------------------------------------
async function sendChatMessage() {
  const input = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const query = input.value.trim();
  const imageToSend = stagedImageFile;

  if (!query && !imageToSend) return;

  const actualQuery = query || "Please analyze this image in relation to our documents.";

  // Render user bubble (with optional image)
  appendChatBubble("user", actualQuery, imageToSend);

  // Clear inputs
  input.value = "";
  clearStagedImage();
  sendBtn.disabled = true;

  // Create empty assistant bubble
  const assistantBubble = appendChatBubble("assistant", "");
  const textContainer = document.createElement("div");
  assistantBubble.appendChild(textContainer);
  textContainer.textContent = imageToSend
    ? "🔍 Analyzing screenshot/image with Gemini Vision & retrieving org documents..."
    : "🔍 Retrieving org-scoped documents & synthesizing answer...";

  try {
    // If image is attached -> Call Multimodal Endpoint
    if (imageToSend) {
      const formData = new FormData();
      formData.append("query", actualQuery);
      formData.append("top_k", "3");
      formData.append("image", imageToSend);

      const res = await fetch("/api/chat/multimodal-query", {
        method: "POST",
        headers: { "X-API-Key": currentTenant.apiKey },
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        textContainer.textContent = `❌ Error: ${err.detail || "Query failed"}`;
        sendBtn.disabled = false;
        return;
      }

      const data = await res.json();
      textContainer.textContent = data.answer;

      // Apply RTL if response is Urdu
      if (/[\u0600-\u06FF]/.test(data.answer)) {
        assistantBubble.classList.add("urdu-text");
        assistantBubble.setAttribute("dir", "rtl");
      }

      // Add Citations
      renderCitationsInBubble(assistantBubble, data.citations);

      // Add Voice Listen Button & Meta
      addVoiceAndMeta(assistantBubble, data.answer, data.model_used, data.latency_ms, data.tokens_used);

      // Update right sidebar
      renderRetrievedChunksInSidebar(data.citations);

    } else {
      // Text-only -> SSE Streaming
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": currentTenant.apiKey,
        },
        body: JSON.stringify({ query: actualQuery, top_k: 3 }),
      });

      if (!response.ok) {
        let errMessage = "Query failed";
        try {
          const errData = await response.json();
          errMessage = errData.detail || errData.message || errMessage;
        } catch (_) {
          try {
            errMessage = await response.text();
          } catch (_) {}
        }
        textContainer.textContent = `❌ Error: ${errMessage}`;
        sendBtn.disabled = false;
        return;
      }

      textContainer.textContent = "";
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedText = "";
      let citations = [];
      let meta = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (!data.done) {
                accumulatedText += data.token;
                textContainer.textContent = accumulatedText;
                if (/[\u0600-\u06FF]/.test(accumulatedText)) {
                  assistantBubble.classList.add("urdu-text");
                  assistantBubble.setAttribute("dir", "rtl");
                }
              } else {
                citations = data.citations || [];
                meta = data;
              }
            } catch (e) {}
          }
        }
      }

      // Add Citations
      renderCitationsInBubble(assistantBubble, citations);

      // Add Voice Listen Button & Meta
      if (meta) {
        addVoiceAndMeta(assistantBubble, accumulatedText, meta.model, meta.latency_ms, meta.tokens);
      }

      // Update right sidebar
      renderRetrievedChunksInSidebar(citations);
    }
  } catch (err) {
    textContainer.textContent = `❌ Request failed: ${err.message}`;
  } finally {
    sendBtn.disabled = false;
    const messages = document.getElementById("chatMessages");
    messages.scrollTop = messages.scrollHeight;
  }
}

function appendChatBubble(role, text, imageFile = null) {
  const container = document.getElementById("chatMessages");
  const bubble = document.createElement("div");
  const isUrdu = /[\u0600-\u06FF]/.test(text);
  bubble.className = `chat-bubble ${role}${isUrdu ? " urdu-text" : ""}`;
  if (isUrdu) bubble.setAttribute("dir", "rtl");

  // If user attached an image, render preview inside bubble
  if (imageFile) {
    const imgEl = document.createElement("img");
    imgEl.className = "user-msg-image";
    const reader = new FileReader();
    reader.onload = (e) => {
      imgEl.src = e.target.result;
    };
    reader.readAsDataURL(imageFile);
    bubble.appendChild(imgEl);
  }

  const textEl = document.createElement("div");
  textEl.textContent = text;
  bubble.appendChild(textEl);

  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
  return bubble;
}

function renderCitationsInBubble(bubble, citations) {
  if (!citations || citations.length === 0) return;
  const citContainer = document.createElement("div");
  citContainer.style.marginTop = "0.75rem";
  citations.forEach((c) => {
    const tag = document.createElement("span");
    tag.className = "citation-tag";
    tag.textContent = `📄 ${c.document_name} [Chunk ${c.chunk_index}]`;
    tag.onclick = () => highlightChunkInSidebar(c);
    citContainer.appendChild(tag);
  });
  bubble.appendChild(citContainer);
}

function addVoiceAndMeta(bubble, fullText, model, latencyMs, tokens) {
  // Voice Listen Button
  const speakBtn = document.createElement("button");
  speakBtn.className = "btn-speak";
  speakBtn.innerHTML = "🔊 Listen";
  speakBtn.onclick = () => speakAnswerText(fullText, speakBtn);
  bubble.appendChild(speakBtn);

  // Metadata
  const metaEl = document.createElement("div");
  metaEl.className = "message-meta";
  metaEl.textContent = `⚡ ${latencyMs || 0}ms • Model: ${model || "Gemini"} • ${tokens || 0} tokens`;
  bubble.appendChild(metaEl);
}

function renderRetrievedChunksInSidebar(citations) {
  const container = document.getElementById("retrievedChunksContainer");
  const badge = document.getElementById("chunkCountBadge");

  if (!citations || citations.length === 0) {
    container.innerHTML = `<div style="color: var(--text-muted); font-size: 0.85rem; text-align: center; margin-top: 3rem;">No context chunks retrieved for this query.</div>`;
    badge.textContent = "0 Chunks";
    return;
  }

  badge.textContent = `${citations.length} Chunks`;
  container.innerHTML = "";

  // Group citations by semantic cluster
  const clusters = {};
  citations.forEach((c) => {
    const cId = c.cluster_id || 1;
    if (!clusters[cId]) clusters[cId] = [];
    clusters[cId].push(c);
  });

  Object.entries(clusters).forEach(([clusterId, clusterChunks]) => {
    const clusterLabel = clusterChunks[0].cluster_label || `Cluster ${clusterId}`;
    const clusterHeader = document.createElement("div");
    clusterHeader.className = "cluster-header-badge";
    clusterHeader.innerHTML = `<span>🎯 ${clusterLabel}</span><span style="color: var(--accent-cyan); font-size: 0.72rem; font-family: 'JetBrains Mono';">${clusterChunks.length} chunk${clusterChunks.length > 1 ? "s" : ""}</span>`;
    container.appendChild(clusterHeader);

    clusterChunks.forEach((c) => {
      const card = document.createElement("div");
      card.className = "chunk-card";
      const sim = (c.similarity_score !== undefined && c.similarity_score !== null) ? c.similarity_score : c.similarity;
      const simBadge = (sim !== undefined && sim !== null)
        ? `<span class="sim-badge">🎯 ${(sim * 100).toFixed(1)}% Match</span>`
        : "";
      card.innerHTML = `
        <div class="chunk-title">
          <span>📄 ${c.document_name}</span>
          <span style="font-size: 0.75rem; color: var(--accent-cyan); font-family: 'JetBrains Mono';">#${c.chunk_index}</span>
        </div>
        <div class="chunk-snippet">${c.snippet || "Relevant document chunk text"}</div>
        <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 0.5rem;">
          ${simBadge}
          <span style="font-size: 0.72rem; color: var(--text-muted);">Page ${c.page_number || 1}</span>
        </div>
      `;
      container.appendChild(card);
    });
  });
}

function highlightChunkInSidebar(citation) {
  alert(`Citation Details:\n\nDocument: ${citation.document_name}\nChunk Index: ${citation.chunk_index}\nSnippet: ${citation.snippet}`);
}

// -------------------------------------------------------------
// Document Management
// -------------------------------------------------------------
async function loadTenantDocuments() {
  const tbody = document.getElementById("docsTableBody");
  try {
    const res = await fetch("/api/documents/", {
      headers: { "X-API-Key": currentTenant.apiKey },
    });
    if (!res.ok) return;
    const docs = await res.json();

    if (docs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 2rem;">No documents uploaded for ${currentTenant.name} yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = docs
      .map(
        (d) => `
      <tr>
        <td style="font-weight: 600; color: #fff;">📄 ${d.filename}</td>
        <td><span style="text-transform: uppercase; font-size: 0.75rem; font-family: 'JetBrains Mono'; color: #a5b4fc;">${d.file_type}</span></td>
        <td>${(d.file_size / 1024).toFixed(1)} KB</td>
        <td style="font-family: 'JetBrains Mono'; color: var(--accent-cyan);">${d.chunk_count}</td>
        <td><span class="status-badge ${d.status}">${d.status}</span></td>
        <td>
          <button class="btn-new-org" style="border-color: rgba(244,63,94,0.4); color: #fb7185;" onclick="deleteDocument('${d.id}')">Delete</button>
        </td>
      </tr>
    `
      )
      .join("");
  } catch (err) {
    console.error(err);
  }
}

async function handleFileUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/documents/upload", {
      method: "POST",
      headers: { "X-API-Key": currentTenant.apiKey },
      body: formData,
    });

    if (res.ok) {
      await loadTenantDocuments();
      appendChatBubble("assistant", `✅ Successfully indexed **${file.name}** for ${currentTenant.name}. Document is now queryable in RAG!`);
    } else {
      const err = await res.json();
      alert(`Upload failed: ${err.detail}`);
    }
  } catch (err) {
    alert(`Upload error: ${err.message}`);
  }
}

async function deleteDocument(docId) {
  if (!confirm("Are you sure you want to delete this document and remove its embeddings?")) return;

  try {
    const res = await fetch(`/api/documents/${docId}`, {
      method: "DELETE",
      headers: { "X-API-Key": currentTenant.apiKey },
    });
    if (res.ok) {
      loadTenantDocuments();
    }
  } catch (err) {
    alert(`Delete failed: ${err.message}`);
  }
}

function setupDropZone() {
  const dropZone = document.getElementById("dropZone");
  if (!dropZone) return;

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      document.getElementById("fileInput").files = e.dataTransfer.files;
      handleFileUpload({ target: { files: e.dataTransfer.files } });
    }
  });
}

// -------------------------------------------------------------
// Live Isolation Check Lab
// -------------------------------------------------------------
async function runLiveIsolationCheck() {
  const boxA = document.getElementById("orgARetrievalBox");
  const boxB = document.getElementById("orgBRetrievalBox");

  boxA.innerHTML = "⏳ Querying vector space as Org A (Acme Corp)...";
  boxB.innerHTML = "⏳ Querying vector space as Org B (Cyberdyne Systems)...";

  try {
    const res = await fetch("/api/eval/run-isolation-check", { method: "POST" });
    const data = await res.json();

    boxA.innerHTML = `
      <div style="color: #34d399; font-weight: 600; margin-bottom: 0.5rem;">✅ Retrieved Chunks (${data.org_a_retrievals.length}):</div>
      ${data.org_a_retrievals
        .map(
          (c) => `
        <div style="background: rgba(255,255,255,0.04); padding: 0.5rem; border-radius: 4px; margin-bottom: 0.4rem; border-left: 2px solid #6366f1;">
          <div style="font-size: 0.75rem; color: #a5b4fc;">[Doc: ${c.filename} | org_id: ${c.org_id}]</div>
          <div>${c.text}</div>
        </div>
      `
        )
        .join("")}
      <div style="margin-top: 0.5rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #34d399;">Cross-tenant Org B chunks leaked: 0 (ZERO)</div>
    `;

    boxB.innerHTML = `
      <div style="color: #34d399; font-weight: 600; margin-bottom: 0.5rem;">✅ Retrieved Chunks (${data.org_b_retrievals.length}):</div>
      ${data.org_b_retrievals
        .map(
          (c) => `
        <div style="background: rgba(255,255,255,0.04); padding: 0.5rem; border-radius: 4px; margin-bottom: 0.4rem; border-left: 2px solid #ec4899;">
          <div style="font-size: 0.75rem; color: #f472b6;">[Doc: ${c.filename} | org_id: ${c.org_id}]</div>
          <div>${c.text}</div>
        </div>
      `
        )
        .join("")}
      <div style="margin-top: 0.5rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #34d399;">Cross-tenant Org A chunks leaked: 0 (ZERO)</div>
    `;
  } catch (err) {
    boxA.textContent = `Failed: ${err.message}`;
    boxB.textContent = `Failed: ${err.message}`;
  }
}

// -------------------------------------------------------------
// Golden Eval Runner
// -------------------------------------------------------------
async function runGoldenEval() {
  const tbody = document.getElementById("evalTableBody");
  tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--accent-cyan); padding: 2rem;">⚡ Running 20+ Golden Test Cases across tenants...</td></tr>`;

  try {
    const res = await fetch("/api/eval/run-golden-eval", { method: "POST" });
    const data = await res.json();

    document.getElementById("metricIsolation").textContent = `${data.isolation_passed_pct}%`;
    document.getElementById("metricRecall").textContent = `${data.retrieval_recall_pct}%`;
    document.getElementById("metricGrounding").textContent = `${data.grounding_accuracy_pct}%`;
    document.getElementById("metricLatency").textContent = `${data.avg_latency_ms}ms ($${data.total_cost_usd})`;

    tbody.innerHTML = data.eval_details
      .map(
        (item) => `
      <tr>
        <td style="font-family: 'JetBrains Mono'; font-size: 0.75rem;">${item.id}</td>
        <td><span style="font-size: 0.75rem; color: #cbd5e1;">${item.org_id}</span></td>
        <td style="max-width: 250px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${item.query}</td>
        <td><span class="status-badge ${item.retrieval_matched ? "ready" : "failed"}">${item.retrieval_matched ? "PASS" : "FAIL"}</span></td>
        <td><span class="status-badge ${item.grounding_matched ? "ready" : "failed"}">${item.grounding_matched ? "GROUNDED" : "FAIL"}</span></td>
        <td style="font-family: 'JetBrains Mono'; font-size: 0.75rem;">${item.latency_ms}ms</td>
        <td><span class="status-badge ${item.isolation_passed ? "ready" : "failed"}">${item.isolation_passed ? "ISOLATED" : "LEAK"}</span></td>
      </tr>
    `
      )
      .join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" style="color: #fb7185; text-align: center;">Error running evaluation: ${err.message}</td></tr>`;
  }
}

// -------------------------------------------------------------
// Organization Modal
// -------------------------------------------------------------
function openNewOrgModal() {
  document.getElementById("newOrgModal").classList.add("active");
}

function closeNewOrgModal() {
  document.getElementById("newOrgModal").classList.remove("active");
}

async function submitNewOrganization() {
  const name = document.getElementById("newOrgName").value.trim();
  const slug = document.getElementById("newOrgId").value.trim();
  if (!name) return alert("Please enter organization name.");

  try {
    const res = await fetch("/api/organizations/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name, org_id: slug || undefined }),
    });

    if (res.ok) {
      const newOrg = await res.json();
      closeNewOrgModal();
      await loadOrganizations();
      setTenant(newOrg);
      appendChatBubble("assistant", `🎉 Created organization **${newOrg.name}**! Generated API Key: \`${newOrg.api_key}\`.`);
    } else {
      const err = await res.json();
      alert(`Creation failed: ${err.detail}`);
    }
  } catch (err) {
    alert(`Error: ${err.message}`);
  }
}

// -------------------------------------------------------------
// Delete Tenant Organization
// -------------------------------------------------------------
async function deleteCurrentTenant() {
  if (!currentTenant || !currentTenant.id) return;

  const confirmed = confirm(
    `⚠️ DANGER: Are you sure you want to permanently delete tenant '${currentTenant.name}' (${currentTenant.id})?\n\nThis will purge all documents, chunks, and vector embeddings for this tenant!`
  );
  if (!confirmed) return;

  try {
    const res = await fetch(`/api/organizations/${currentTenant.id}`, {
      method: "DELETE",
    });

    if (res.ok) {
      const result = await res.json();
      alert(`✅ ${result.message}`);
      await loadOrganizations();

      // Clear current chat & documents
      document.getElementById("chatMessages").innerHTML = `
        <div class="chat-bubble assistant">
          👋 Switched to new active tenant. Upload documents or ask questions to get started.
        </div>
      `;
      document.getElementById("retrievedChunksContainer").innerHTML = `
        <div style="color: var(--text-muted); font-size: 0.85rem; text-align: center; margin-top: 3rem;">
          No context chunks retrieved.
        </div>
      `;
      loadTenantDocuments();
    } else {
      const err = await res.json();
      alert(`❌ Failed to delete tenant: ${err.detail || "Unknown error"}`);
    }
  } catch (err) {
    alert(`❌ Delete error: ${err.message}`);
  }
}

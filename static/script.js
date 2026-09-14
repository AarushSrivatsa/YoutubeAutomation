const API = "/api/v1";

let currentJobId = null;
let detailPollTimer = null;
let jobsListPollTimer = null;
let progressStream = null;

// ── DOM refs ──────────────────────────────────────────────────────────────
const navJobs = document.getElementById("nav-jobs");
const navNew = document.getElementById("nav-new");

const viewJobs = document.getElementById("view-jobs");
const viewNew = document.getElementById("view-new");
const viewDetail = document.getElementById("view-detail");

const jobsList = document.getElementById("jobs-list");
const refreshJobsBtn = document.getElementById("refresh-jobs-btn");

const createForm = document.getElementById("create-form");
const createBtn = document.getElementById("create-btn");

const backBtn = document.getElementById("back-btn");
const statusBadge = document.getElementById("status-badge");
const jobIdDisplay = document.getElementById("job-id-display");
const promptDisplay = document.getElementById("prompt-display");
const progressLog = document.getElementById("progress-log");
const errorBox = document.getElementById("error-box");
const retryBtn = document.getElementById("retry-btn");
const deleteBtn = document.getElementById("delete-btn");

const scriptCard = document.getElementById("script-card");
const scriptTitle = document.getElementById("script-title");
const segmentsList = document.getElementById("segments-list");
const approveBtn = document.getElementById("approve-btn");

const resultCard = document.getElementById("result-card");
const resultVideo = document.getElementById("result-video");
const resultAudio = document.getElementById("result-audio");
const videoLink = document.getElementById("video-link");
const audioLink = document.getElementById("audio-link");

// ── View switching ───────────────────────────────────────────────────────
function showView(name) {
  viewJobs.classList.toggle("hidden", name !== "jobs");
  viewNew.classList.toggle("hidden", name !== "new");
  viewDetail.classList.toggle("hidden", name !== "detail");

  navJobs.classList.toggle("active", name === "jobs");
  navNew.classList.toggle("active", name === "new");

  clearInterval(jobsListPollTimer);
  clearInterval(detailPollTimer);
  if (progressStream) { progressStream.close(); progressStream = null; }

  if (name === "jobs") {
    loadJobsList();
    jobsListPollTimer = setInterval(loadJobsList, 5000);
  }
}

navJobs.addEventListener("click", () => showView("jobs"));
navNew.addEventListener("click", () => { createForm.reset(); showView("new"); });
backBtn.addEventListener("click", () => showView("jobs"));

// ── Jobs list ────────────────────────────────────────────────────────────
async function loadJobsList() {
  try {
    const res = await fetch(`${API}/jobs?limit=50`);
    if (!res.ok) throw new Error(res.statusText);
    const jobs = await res.json();
    renderJobsList(jobs);
  } catch (err) {
    jobsList.innerHTML = `<p class="muted">Failed to load jobs: ${err.message}</p>`;
  }
}

function renderJobsList(jobs) {
  if (!jobs.length) {
    jobsList.innerHTML = `<p class="muted">No jobs yet — click "+ New Job" to create one.</p>`;
    return;
  }

  jobsList.innerHTML = "";
  jobs.forEach((job) => {
    const div = document.createElement("div");
    div.className = "job-card";
    const created = new Date(job.created_at).toLocaleString();
    div.innerHTML = `
      <div class="job-info">
        <p class="job-prompt">${escapeHtml(job.prompt)}</p>
        <p class="job-meta">
          <span class="badge ${job.status}">${job.status}</span>
          &nbsp;·&nbsp; ${created}
        </p>
      </div>
      <div class="job-actions">
        ${job.status === "failed" ? `<button class="ghost" data-action="retry">🔁 Retry</button>` : ""}
        <button data-action="view">View</button>
        <button class="danger" data-action="delete">🗑</button>
      </div>
    `;
    div.querySelector('[data-action="view"]').addEventListener("click", () => openJob(job.id));
    const retryEl = div.querySelector('[data-action="retry"]');
    if (retryEl) retryEl.addEventListener("click", (e) => { e.stopPropagation(); retryFromList(job.id); });
    div.querySelector('[data-action="delete"]').addEventListener("click", (e) => {
      e.stopPropagation();
      deleteFromList(job.id);
    });
    jobsList.appendChild(div);
  });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str || "";
  return div.innerHTML;
}

async function retryFromList(jobId) {
  try {
    const res = await fetch(`${API}/jobs/${jobId}/retry`, { method: "POST" });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    openJob(jobId);
  } catch (err) {
    alert("Retry failed: " + err.message);
  }
}

async function deleteFromList(jobId) {
  if (!confirm("Delete this job and its media? This can't be undone.")) return;
  try {
    const res = await fetch(`${API}/jobs/${jobId}`, { method: "DELETE" });
    if (!res.ok && res.status !== 204) throw new Error(res.statusText);
    loadJobsList();
  } catch (err) {
    alert("Delete failed: " + err.message);
  }
}

refreshJobsBtn.addEventListener("click", loadJobsList);

// ── Create job ────────────────────────────────────────────────────────────
createForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  createBtn.disabled = true;
  createBtn.textContent = "Starting…";

  const formData = new FormData();
  formData.append("prompt", document.getElementById("prompt").value);
  formData.append("image", document.getElementById("image").files[0]);
  formData.append("wpm", document.getElementById("wpm").value);
  formData.append("video_length_min", document.getElementById("video_length_min").value);
  formData.append("voice_model", document.getElementById("voice_model").value);

  try {
    const res = await fetch(`${API}/jobs`, { method: "POST", body: formData });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    openJob(data.job_id);
  } catch (err) {
    alert("Failed to create job: " + err.message);
  } finally {
    createBtn.disabled = false;
    createBtn.textContent = "Generate";
  }
});

// ── Job detail ───────────────────────────────────────────────────────────
function openJob(jobId) {
  currentJobId = jobId;
  jobIdDisplay.textContent = jobId;
  progressLog.textContent = "";
  showSection(errorBox, false);
  showSection(scriptCard, false);
  showSection(resultCard, false);
  showSection(retryBtn, false);
  showView("detail");

  logLine("tracking job " + jobId);
  openProgressStream(jobId);
  clearInterval(detailPollTimer);
  detailPollTimer = setInterval(() => pollJob(jobId), 2500);
  pollJob(jobId);
}

function logLine(text) {
  const time = new Date().toLocaleTimeString();
  progressLog.textContent += `[${time}] ${text}\n`;
  progressLog.scrollTop = progressLog.scrollHeight;
}

function setStatusBadge(status) {
  statusBadge.textContent = status;
  statusBadge.className = "badge " + status;
}

function showSection(el, show = true) {
  el.classList.toggle("hidden", !show);
}

function openProgressStream(jobId) {
  progressStream = new EventSource(`${API}/jobs/${jobId}/progress`);
  progressStream.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      logLine(data.stage || JSON.stringify(data));
    } catch {
      logLine(event.data);
    }
  };
  progressStream.onerror = () => {
    // stream naturally closes once the job finishes server-side; polling
    // below is what actually drives the UI state, so just clean up quietly
    if (progressStream) { progressStream.close(); progressStream = null; }
  };
}

async function pollJob(jobId) {
  try {
    const res = await fetch(`${API}/jobs/${jobId}`);
    if (!res.ok) throw new Error("job not found");
    const job = await res.json();

    setStatusBadge(job.status);
    promptDisplay.textContent = `"${job.prompt}"`;

    if (job.error_message) {
      errorBox.textContent = job.error_message;
      showSection(errorBox, true);
    } else {
      showSection(errorBox, false);
    }

    if (job.status === "awaiting_approval") {
      renderScript(job);
      showSection(scriptCard, true);
    } else {
      showSection(scriptCard, false);
    }

    if (job.status === "completed") {
      renderResult(job);
      showSection(resultCard, true);
      showSection(retryBtn, false);
      clearInterval(detailPollTimer);
    } else if (job.status === "failed") {
      showSection(retryBtn, true);
      clearInterval(detailPollTimer);
    }
  } catch (err) {
    logLine("poll error: " + err.message);
  }
}

// ── Script review ────────────────────────────────────────────────────────
function renderScript(job) {
  scriptTitle.textContent = job.story?.title || "(untitled)";
  segmentsList.innerHTML = "";

  (job.segments || []).forEach((seg) => {
    const div = document.createElement("div");
    div.className = "segment";
    div.innerHTML = `
      <div class="segment-header">
        <span>Segment ${seg.segment_index} — ${seg.status}</span>
      </div>
      <textarea data-index="${seg.segment_index}">${escapeHtml(seg.text)}</textarea>
      <div class="segment-actions">
        <button data-action="save" data-index="${seg.segment_index}">💾 Save edit</button>
      </div>
    `;
    segmentsList.appendChild(div);
  });
}

segmentsList.addEventListener("click", async (e) => {
  if (e.target.dataset.action !== "save") return;
  const index = e.target.dataset.index;
  const textarea = segmentsList.querySelector(`textarea[data-index="${index}"]`);
  const newText = textarea.value;

  e.target.disabled = true;
  e.target.textContent = "Saving…";

  try {
    const res = await fetch(`${API}/jobs/${currentJobId}/segments/${index}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: newText }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    textarea.closest(".segment").classList.add("saved");
    e.target.textContent = "✅ Saved";
    logLine(`segment ${index} updated`);
  } catch (err) {
    alert("Failed to save segment: " + err.message);
    e.target.textContent = "💾 Save edit";
  } finally {
    e.target.disabled = false;
  }
});

approveBtn.addEventListener("click", async () => {
  approveBtn.disabled = true;
  approveBtn.textContent = "Approving…";
  try {
    const res = await fetch(`${API}/jobs/${currentJobId}/approve`, { method: "POST" });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    logLine("approved — generating voiceover + video…");
    showSection(scriptCard, false);
    clearInterval(detailPollTimer);
    detailPollTimer = setInterval(() => pollJob(currentJobId), 2500);
  } catch (err) {
    alert("Failed to approve: " + err.message);
  } finally {
    approveBtn.disabled = false;
    approveBtn.textContent = "✅ Approve & generate video";
  }
});

// ── Retry / delete from detail view ─────────────────────────────────────
retryBtn.addEventListener("click", async () => {
  retryBtn.disabled = true;
  try {
    const res = await fetch(`${API}/jobs/${currentJobId}/retry`, { method: "POST" });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    logLine("retry started");
    showSection(retryBtn, false);
    clearInterval(detailPollTimer);
    detailPollTimer = setInterval(() => pollJob(currentJobId), 2500);
  } catch (err) {
    alert("Failed to retry: " + err.message);
  } finally {
    retryBtn.disabled = false;
  }
});

deleteBtn.addEventListener("click", async () => {
  if (!confirm("Delete this job and its media? This can't be undone.")) return;
  try {
    const res = await fetch(`${API}/jobs/${currentJobId}`, { method: "DELETE" });
    if (!res.ok && res.status !== 204) throw new Error(res.statusText);
    showView("jobs");
  } catch (err) {
    alert("Delete failed: " + err.message);
  }
});

// ── Final result ─────────────────────────────────────────────────────────
function renderResult(job) {
  if (job.video_url) {
    resultVideo.src = job.video_url;
    videoLink.href = job.video_url;
  }
  if (job.audio_url) {
    resultAudio.src = job.audio_url;
    audioLink.href = job.audio_url;
  }
}

// ── Boot ─────────────────────────────────────────────────────────────────
showView("jobs");
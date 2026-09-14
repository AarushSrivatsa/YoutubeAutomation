const API = "/api/v1";

let currentJobId = null;
let detailPollTimer = null;
let jobsListPollTimer = null;
let progressStream = null;
let activeQuery = "";

// ── DOM refs ──────────────────────────────────────────────────────────────
const navJobs = document.getElementById("nav-jobs");
const navNew = document.getElementById("nav-new");

const viewJobs = document.getElementById("view-jobs");
const viewNew = document.getElementById("view-new");
const viewDetail = document.getElementById("view-detail");

const jobsList = document.getElementById("jobs-list");
const refreshJobsBtn = document.getElementById("refresh-jobs-btn");

const searchForm = document.getElementById("search-form");
const searchInput = document.getElementById("search-input");
const clearSearchBtn = document.getElementById("clear-search-btn");

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
const scriptCardHeading = document.getElementById("script-card-heading");
const scriptHint = document.getElementById("script-hint");
const scriptTitle = document.getElementById("script-title");
const segmentsList = document.getElementById("segments-list");
const approveBtn = document.getElementById("approve-btn");

const resultCard = document.getElementById("result-card");
const resultVideo = document.getElementById("result-video");
const resultAudio = document.getElementById("result-audio");
const videoLink = document.getElementById("video-link");
const audioLink = document.getElementById("audio-link");
const swapImageForm = document.getElementById("swap-image-form");
const swapImageBtn = document.getElementById("swap-image-btn");

// ── Helpers ──────────────────────────────────────────────────────────────
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str || "";
  return div.innerHTML;
}

function formatStatus(status) {
  if (!status) return "";
  const spaced = status.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function showSection(el, show = true) {
  el.classList.toggle("hidden", !show);
}

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
    const params = new URLSearchParams({ limit: "50" });
    if (activeQuery) params.set("q", activeQuery);
    const res = await fetch(`${API}/jobs?${params.toString()}`);
    if (!res.ok) throw new Error(res.statusText);
    const jobs = await res.json();
    renderJobsList(jobs);
  } catch (err) {
    jobsList.innerHTML = `<p class="muted">Failed to load jobs: ${escapeHtml(err.message)}</p>`;
  }
}

function renderJobsList(jobs) {
  if (!jobs.length) {
    jobsList.innerHTML = activeQuery
      ? `<p class="muted">No jobs match “${escapeHtml(activeQuery)}”.</p>`
      : `<p class="muted">No jobs yet — click "New job" to create one.</p>`;
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
          <span class="badge ${job.status}">${formatStatus(job.status)}</span>
          &nbsp;·&nbsp; ${created}
        </p>
      </div>
      <div class="job-actions">
        ${job.status === "failed" ? `<button class="ghost" data-action="retry">↻ Retry</button>` : ""}
        <button data-action="view">View</button>
        <button class="danger" data-action="delete">Delete</button>
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

searchForm.addEventListener("submit", (e) => {
  e.preventDefault();
  activeQuery = searchInput.value.trim();
  showSection(clearSearchBtn, !!activeQuery);
  loadJobsList();
});

clearSearchBtn.addEventListener("click", () => {
  activeQuery = "";
  searchInput.value = "";
  showSection(clearSearchBtn, false);
  loadJobsList();
});

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
  segmentsList.innerHTML = "";
  swapImageForm.reset();
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
  statusBadge.textContent = formatStatus(status);
  statusBadge.className = "badge " + status;
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

// Statuses where the script (segments) are meaningful to show.
// Editing is allowed both pre-approval (instant patch) and after
// completion/failure (the API re-runs voiceover+stitch automatically).
const SCRIPT_VISIBLE_STATUSES = ["awaiting_approval", "completed", "failed"];

async function pollJob(jobId) {
  try {
    const res = await fetch(`${API}/jobs/${jobId}`);
    if (!res.ok) throw new Error("job not found");
    const job = await res.json();

    // Ignore stale responses if the user has already navigated to another job.
    if (jobId !== currentJobId) return;

    setStatusBadge(job.status);
    promptDisplay.textContent = `"${job.prompt}"`;

    if (job.error_message) {
      errorBox.textContent = job.error_message;
      showSection(errorBox, true);
    } else {
      showSection(errorBox, false);
    }

    const hasScript = job.segments && job.segments.length > 0;
    if (hasScript && SCRIPT_VISIBLE_STATUSES.includes(job.status)) {
      renderScript(job);
      showSection(scriptCard, true);
      const isPending = job.status === "awaiting_approval";
      showSection(approveBtn, isPending);
      scriptCardHeading.textContent = isPending ? "Review the script" : "Script";
      scriptHint.textContent = isPending
        ? "Edit any segment below, then approve to generate voiceover and video."
        : "Editing a segment here regenerates the voiceover and video for this job.";
    } else {
      showSection(scriptCard, false);
    }

    if (job.status === "completed") {
      renderResult(job);
      showSection(resultCard, true);
      showSection(retryBtn, false);
    } else {
      showSection(resultCard, false);
    }

    showSection(retryBtn, job.status === "failed");
    // Polling intentionally keeps running after completion/failure — a
    // segment or image edit can send the job back to "running" and then
    // "completed" again, and we still want to see that happen live.
  } catch (err) {
    logLine("poll error: " + err.message);
  }
}

// ── Script review ────────────────────────────────────────────────────────

function buildSegmentEl(seg) {
  const div = document.createElement("div");
  div.className = "segment";
  div.dataset.index = String(seg.segment_index);
  div.innerHTML = `
    <div class="segment-header">
      <span>Segment ${seg.segment_index} — ${formatStatus(seg.status)}</span>
    </div>
    <textarea data-index="${seg.segment_index}">${escapeHtml(seg.text)}</textarea>
    <div class="segment-actions">
      <button data-action="save" data-index="${seg.segment_index}">Save edit</button>
    </div>
    <div class="segment-rewrite">
      <input type="text" class="rewrite-instruction" data-index="${seg.segment_index}"
             placeholder="Or describe a rewrite — “make the ending more hopeful”">
      <button data-action="rewrite" data-index="${seg.segment_index}" class="ghost">✨ AI rewrite</button>
    </div>
  `;
  const textarea = div.querySelector("textarea");
  textarea.addEventListener("input", () => {
    div.dataset.dirty = "1";
    div.classList.add("dirty");
    div.classList.remove("saved");
  });
  return div;
}

// Merges the server's segment data into the DOM instead of wiping and
// rebuilding it — rebuilding on every 2.5s poll used to discard whatever
// the user was mid-typing in a textarea before they hit Save.
function renderScript(job) {
  scriptTitle.textContent = job.story?.title || "(untitled)";
  const segments = job.segments || [];
  const seenIndices = new Set();

  segments.forEach((seg) => {
    const idx = String(seg.segment_index);
    seenIndices.add(idx);
    let el = segmentsList.querySelector(`.segment[data-index="${idx}"]`);

    if (!el) {
      el = buildSegmentEl(seg);
      segmentsList.appendChild(el);
      return;
    }

    const textarea = el.querySelector("textarea");
    const isDirty = el.dataset.dirty === "1";
    const isFocused = document.activeElement === textarea;
    if (!isDirty && !isFocused) {
      textarea.value = seg.text;
    }
    el.querySelector(".segment-header span").textContent =
      `Segment ${seg.segment_index} — ${formatStatus(seg.status)}`;
  });

  // Remove segments no longer present server-side (shouldn't normally happen).
  Array.from(segmentsList.children).forEach((el) => {
    if (!seenIndices.has(el.dataset.index)) el.remove();
  });
}

segmentsList.addEventListener("click", async (e) => {
  const action = e.target.dataset.action;
  if (action !== "save" && action !== "rewrite") return;

  const index = e.target.dataset.index;
  const segEl = segmentsList.querySelector(`.segment[data-index="${index}"]`);
  const textarea = segEl.querySelector("textarea");
  const saveBtn = segEl.querySelector('[data-action="save"]');
  const rewriteBtn = segEl.querySelector('[data-action="rewrite"]');
  const instructionInput = segEl.querySelector(".rewrite-instruction");

  let body;
  if (action === "save") {
    const newText = textarea.value.trim();
    if (!newText) { alert("Segment text can't be empty."); return; }
    body = { text: newText };
  } else {
    const instruction = instructionInput.value.trim();
    if (!instruction) { alert("Describe what you'd like changed first."); return; }
    body = { instruction };
  }

  saveBtn.disabled = true;
  rewriteBtn.disabled = true;
  e.target.textContent = action === "save" ? "Saving…" : "Rewriting…";

  try {
    const res = await fetch(`${API}/jobs/${currentJobId}/segments/${index}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);

    delete segEl.dataset.dirty;
    segEl.classList.remove("dirty");
    segEl.classList.add("saved");
    if (action === "rewrite") instructionInput.value = "";
    logLine(`segment ${index} ${action === "save" ? "updated" : "rewritten"}`);
    // Refresh immediately so the textarea picks up the server's version
    // (especially important for AI rewrite, which computes the new text).
    pollJob(currentJobId);
  } catch (err) {
    alert(`Failed to ${action === "save" ? "save" : "rewrite"} segment: ` + err.message);
  } finally {
    saveBtn.disabled = false;
    rewriteBtn.disabled = false;
    saveBtn.textContent = "Save edit";
    rewriteBtn.textContent = "✨ AI rewrite";
  }
});

approveBtn.addEventListener("click", async () => {
  approveBtn.disabled = true;
  approveBtn.textContent = "Approving…";
  try {
    const res = await fetch(`${API}/jobs/${currentJobId}/approve`, { method: "POST" });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    logLine("approved — generating voiceover + video…");
    pollJob(currentJobId);
  } catch (err) {
    alert("Failed to approve: " + err.message);
  } finally {
    approveBtn.disabled = false;
    approveBtn.textContent = "Approve & generate video";
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
    pollJob(currentJobId);
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

swapImageForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = document.getElementById("swap-image-input").files[0];
  if (!file) return;

  swapImageBtn.disabled = true;
  swapImageBtn.textContent = "Swapping…";

  const formData = new FormData();
  formData.append("image", file);

  try {
    const res = await fetch(`${API}/jobs/${currentJobId}/image`, {
      method: "PATCH",
      body: formData,
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    logLine("image swap started — re-stitching video…");
    swapImageForm.reset();
    pollJob(currentJobId);
  } catch (err) {
    alert("Image swap failed: " + err.message);
  } finally {
    swapImageBtn.disabled = false;
    swapImageBtn.textContent = "Swap image";
  }
});

// ── Boot ─────────────────────────────────────────────────────────────────
showView("jobs");
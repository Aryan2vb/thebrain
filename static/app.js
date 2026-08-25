const statusEl = document.getElementById("status");
const rosterEl = document.getElementById("roster");
const enrollForm = document.getElementById("enroll-form");
const identifyForm = document.getElementById("identify-form");
const result = document.getElementById("result");
const nameInput = document.getElementById("person-name");
const enrollFile = document.getElementById("enroll-file");
const identifyFile = document.getElementById("identify-file");

function setRoster(people) {
  rosterEl.textContent = people.length
    ? `On file: ${people.join(", ")}`
    : "On file: nobody yet";
}

function setStatus(text) {
  statusEl.textContent = text;
}

async function refreshStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  setRoster(data.people || []);
  if (data.error) {
    setStatus(data.error);
    return false;
  }
  if (!data.ready) {
    setStatus("Warming the model…");
    return false;
  }
  setStatus("Ready.");
  return true;
}

async function waitUntilReady() {
  while (!(await refreshStatus())) {
    await new Promise((r) => setTimeout(r, 1200));
  }
}

function bindDrop(zone, input, onFile) {
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("is-over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("is-over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("is-over");
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  });
  input.addEventListener("change", () => {
    const file = input.files[0];
    if (file) onFile(file);
  });
}

document.querySelectorAll(".mode").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mode").forEach((b) => b.classList.remove("is-on"));
    btn.classList.add("is-on");
    const identify = btn.dataset.mode === "identify";
    enrollForm.classList.toggle("is-hidden", identify);
    identifyForm.classList.toggle("is-hidden", !identify);
    if (!identify) result.hidden = true;
  });
});

enrollForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = enrollFile.files[0];
  if (!file) {
    setStatus("Choose a front-face photo.");
    return;
  }
  await enrollFilePair(nameInput.value, file);
});

async function enrollFilePair(name, file) {
  const body = new FormData();
  body.append("name", name);
  body.append("image", file);
  document.body.classList.add("is-busy");
  setStatus("Saving…");
  const res = await fetch("/api/enroll", { method: "POST", body });
  const data = await res.json();
  document.body.classList.remove("is-busy");
  if (data.people) setRoster(data.people);
  setStatus(data.message || data.error || "Done.");
  enrollFile.value = "";
}

async function identifyFilePair(file) {
  const body = new FormData();
  body.append("image", file);
  document.body.classList.add("is-busy");
  setStatus("Reading faces…");
  result.hidden = true;
  const res = await fetch("/api/identify", { method: "POST", body });
  const data = await res.json();
  document.body.classList.remove("is-busy");
  if (!data.ok) {
    setStatus(data.error || "Could not read that photo.");
    return;
  }
  result.src = data.image;
  result.hidden = false;
  const names = data.faces.map((f) => f.name);
  const took = data.seconds != null ? ` (${data.seconds}s)` : "";
  setStatus(
    names.length
      ? `Found ${names.length}: ${names.join(", ")}${took}`
      : `No faces found in that photo.${took}`
  );
  identifyFile.value = "";
}

bindDrop(document.getElementById("enroll-drop"), enrollFile, (file) => {
  enrollFile.files = fileList(file);
  setStatus(`Portrait ready: ${file.name}`);
});

bindDrop(document.getElementById("identify-drop"), identifyFile, (file) => {
  identifyFilePair(file);
});

function fileList(file) {
  const list = new DataTransfer();
  list.items.add(file);
  return list.files;
}

waitUntilReady();

const DEFAULTS = {
  backendUrl: "http://127.0.0.1:5000",
  dateOverride: "",
  enabled: true
};

function $(id) {
  return document.getElementById(id);
}

async function load() {
  const stored = await chrome.storage.sync.get(DEFAULTS);
  $("backendUrl").value = stored.backendUrl || DEFAULTS.backendUrl;
  $("dateOverride").value = stored.dateOverride || "";
  $("enable").checked = stored.enabled !== false;
}

async function save() {
  const backendUrl = $("backendUrl").value.trim();
  const dateOverride = $("dateOverride").value.trim();
  const enabled = $("enable").checked;

  await chrome.storage.sync.set({
    backendUrl: backendUrl || DEFAULTS.backendUrl,
    dateOverride,
    enabled
  });

  $("status").textContent = "Saved.";
  setTimeout(() => ($("status").textContent = ""), 1200);
}

document.addEventListener("DOMContentLoaded", load);
$("save").addEventListener("click", save);

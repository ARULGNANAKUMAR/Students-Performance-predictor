/* EduPredict – Frontend JS */

// ── Sidebar toggle ────────────────────────
const sidebar  = document.getElementById("sidebar");
const overlay  = document.getElementById("sidebarOverlay");
const burgerBtn = document.getElementById("burgerBtn");

function openSidebar()  { sidebar.classList.add("open");  overlay.classList.add("open"); }
function closeSidebar() { sidebar.classList.remove("open"); overlay.classList.remove("open"); }

burgerBtn?.addEventListener("click", openSidebar);
overlay?.addEventListener("click", closeSidebar);

// ── Model status pill ─────────────────────
async function updateModelStatus() {
  try {
    const r = await fetch("/api/model/status");
    const d = await r.json();
    const dot   = document.getElementById("statusDot");
    const label = document.getElementById("statusLabel");
    if (!dot || !label) return;
    if (d.trained) {
      dot.className   = "status-dot trained";
      label.textContent = "Model: Ready";
    } else {
      dot.className   = "status-dot untrained";
      label.textContent = "Model: Untrained";
    }
  } catch (_) {}
}
updateModelStatus();

// ── Auto-dismiss flash messages ───────────
setTimeout(() => {
  document.querySelectorAll(".flash").forEach(el => {
    el.style.transition = "opacity .5s";
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 500);
  });
}, 5000);

// ── Confirm delete buttons (data-confirm) ─
document.querySelectorAll("[data-confirm]").forEach(btn => {
  btn.addEventListener("click", e => {
    if (!confirm(btn.dataset.confirm)) e.preventDefault();
  });
});

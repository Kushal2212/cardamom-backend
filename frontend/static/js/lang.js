// ── Current language state ─────────────────────────────────────────────────
let currentLang = localStorage.getItem("cardamom_lang") || "en";

// ── Switch language ────────────────────────────────────────────────────────
function toggleLang() {
  currentLang = currentLang === "en" ? "np" : "en";
  localStorage.setItem("cardamom_lang", currentLang);
  applyLang();
}

function setLang(lang) {
  currentLang = lang;
  localStorage.setItem("cardamom_lang", lang);
  applyLang();
}

// ── Apply language to all elements ─────────────────────────────────────────
function applyLang() {
  const isNepali = currentLang === "np";

  // Update all elements with data-en and data-np
  document.querySelectorAll("[data-en]").forEach((el) => {
    const en = el.getAttribute("data-en");
    const np = el.getAttribute("data-np");
    if (en && np) {
      // Preserve child elements — only update text if no children
      if (el.children.length === 0) {
        el.textContent = isNepali ? np : en;
      } else {
        // Has children — update data-text only
        el.setAttribute("data-current", isNepali ? np : en);
      }
    }
  });

  // Update placeholders
  document.querySelectorAll("[data-en-placeholder]").forEach((el) => {
    const en = el.getAttribute("data-en-placeholder");
    const np = el.getAttribute("data-np-placeholder");
    if (en && np) el.placeholder = isNepali ? np : en;
  });

  // Update toggle button text
  const btn = document.getElementById("lang-toggle-btn");
  if (btn) {
    btn.textContent = isNepali ? "🇬🇧 English" : "🇳🇵 नेपाली";
    btn.title = isNepali ? "Switch to English" : "नेपालीमा हेर्नुहोस्";
  }

  // Update html lang attribute
  document.documentElement.lang = isNepali ? "ne" : "en";

  // Update font for Nepali (optional — helps readability)
  if (isNepali) {
    document.body.style.fontFamily =
      "'Noto Sans Devanagari', 'Outfit', sans-serif";
  } else {
    document.body.style.fontFamily = "'Outfit', sans-serif";
  }
}

// ── Run on page load ────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", function () {
  // Load Noto Sans Devanagari font for Nepali
  if (!document.getElementById("noto-font")) {
    const link = document.createElement("link");
    link.id = "noto-font";
    link.rel = "stylesheet";
    link.href =
      "https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@300;400;500;600&display=swap";
    document.head.appendChild(link);
  }
  applyLang();
});

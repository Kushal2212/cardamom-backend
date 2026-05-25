const TOKEN_KEY = "cardamom_token";
const USER_KEY = "cardamom_user";

function isLoggedIn() {
  return !!localStorage.getItem(TOKEN_KEY);
}
function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}
function getUsername() {
  return localStorage.getItem(USER_KEY) || "User";
}

function protect() {
  if (!isLoggedIn()) {
    localStorage.setItem("redirectAfterLogin", location.pathname);
    location.href = "login.html";
  }
}

function redirectAfterLogin() {
  const dest = localStorage.getItem("redirectAfterLogin") || "index.html";
  localStorage.removeItem("redirectAfterLogin");
  location.href = dest;
}

function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

function logout() {
  clearAuth();
  location.href = "login.html";
}

async function apiCall(url, method = "GET", body = null) {
  const headers = { "Content-Type": "application/json" };
  const token = getToken();
  if (token) headers["Authorization"] = "Bearer " + token;

  const opts = { method, headers };
  if (body) opts.body = JSON.stringify(body);

  try {
    const fullUrl = API_BASE.replace(/\/$/, "") + "/" + url.replace(/^\//, "");
    const res = await fetch(fullUrl, opts);

    // ✅ Return error object — do NOT redirect here
    if (res.status === 401) {
      clearAuth();
      return { error: "unauthorized" }; // ← NO location.href
    }

    return await res.json();
  } catch (err) {
    console.error("apiCall error:", err);
    return { error: "network_error" };
  }
}

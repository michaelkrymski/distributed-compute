export function teamKey() {
  return typeof window === "undefined"
    ? ""
    : (sessionStorage.getItem("gator-team-key") ?? "");
}
export async function apiFetch(path: string, options: RequestInit = {}) {
  return fetch(path, {
    ...options,
    headers: {
      ...options.headers,
      ...(teamKey() && teamKey() !== "cookie-session"
        ? { Authorization: "Bearer " + teamKey() }
        : {}),
    },
    cache: "no-store",
  });
}
export function connectTeam(key: string) {
  if (key) sessionStorage.setItem("gator-team-key", key.trim());
  else sessionStorage.removeItem("gator-team-key");
  window.dispatchEvent(new Event("glm-auth"));
}

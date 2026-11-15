"use client";
import { useEffect, useState } from "react";
import { connectTeam, teamKey } from "../lib/client-api";

export default function AdminLogin({
  connected,
  name,
}: {
  connected: boolean;
  name?: string;
}) {
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [password, setPassword] = useState(""),
    [confirmation, setConfirmation] = useState("");
  const [key, setKey] = useState(""),
    [memberKey, setMemberKey] = useState("");
  const [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  useEffect(() => {
    fetch("/api/auth", { cache: "no-store" })
      .then(async (r) => {
        if (!r.ok) throw new Error("Admin sign-in is unavailable.");
        const status = (await r.json()) as {
          configured: boolean;
          authenticated: boolean;
        };
        setConfigured(status.configured);
        if (status.authenticated) connectTeam("cookie-session");
        else if (sessionStorage.getItem("gator-team-key") === "cookie-session")
          connectTeam("");
      })
      .catch((e) => setError(e.message));
  }, []);
  async function submit() {
    setBusy(true);
    setError("");
    try {
      if (!configured && password !== confirmation)
        throw new Error("Passwords do not match.");
      const response = await fetch("/api/auth", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(!configured ? { Authorization: "Bearer " + key.trim() } : {}),
        },
        body: JSON.stringify({
          action: configured ? "login" : "setup",
          password,
        }),
      });
      const body = (await response.json()) as { error?: string };
      if (!response.ok) throw new Error(body.error ?? "Sign-in failed.");
      setPassword("");
      setConfirmation("");
      setKey("");
      setConfigured(true);
      connectTeam("cookie-session");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function logout() {
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/auth", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "logout" }),
      });
      if (!response.ok) throw new Error("Could not sign out. Try again.");
      connectTeam("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="admin-access">
      {connected ? (
        <div className="team-access">
          <span>Connected as {name}</span>
          {name === "michaelkrymski" && teamKey() !== "cookie-session" && (
            <button
              className="primary small"
              onClick={() => {
                setKey(teamKey());
                connectTeam("");
              }}
            >
              {configured === false
                ? "Set up admin password"
                : "Use admin password"}
            </button>
          )}
          <button
            className="secondary small"
            disabled={busy}
            onClick={() => void logout()}
          >
            Sign out / disconnect
          </button>
        </div>
      ) : (
        <>
          <form
            className="team-access"
            onSubmit={(e) => {
              e.preventDefault();
              void submit();
            }}
          >
            <span>
              {configured === false
                ? "Set up your admin password"
                : "Administrator sign-in"}
            </span>
            <input
              type="text"
              autoComplete="username"
              value="michaelkrymski"
              readOnly
              aria-label="Admin username"
            />
            {configured === false && (
              <input
                type="password"
                autoComplete="off"
                aria-label="Coordinator activation key"
                placeholder="Coordinator key (one-time activation)"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                required
              />
            )}
            <input
              type="password"
              aria-label="Admin password"
              autoComplete={configured ? "current-password" : "new-password"}
              minLength={16}
              maxLength={256}
              placeholder="Password (16+ characters)"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            {configured === false && (
              <input
                type="password"
                aria-label="Confirm admin password"
                autoComplete="new-password"
                minLength={16}
                maxLength={256}
                placeholder="Confirm password"
                value={confirmation}
                onChange={(e) => setConfirmation(e.target.value)}
                required
              />
            )}
            <button
              type="submit"
              className="primary small"
              disabled={busy || configured === null}
            >
              {busy
                ? "Please wait…"
                : configured === false
                  ? "Create admin account"
                  : "Sign in"}
            </button>
          </form>
          <details className="member-access">
            <summary>Team member access key</summary>
            <form
              className="team-access"
              onSubmit={(e) => {
                e.preventDefault();
                connectTeam(memberKey);
                setMemberKey("");
              }}
            >
              <input
                type="password"
                aria-label="Team access key"
                autoComplete="off"
                placeholder="Your member key"
                value={memberKey}
                onChange={(e) => setMemberKey(e.target.value)}
                required
              />
              <button className="secondary small">Connect</button>
            </form>
          </details>
        </>
      )}
      {error && (
        <p role="alert" className="notice error">
          {error}
        </p>
      )}
    </section>
  );
}

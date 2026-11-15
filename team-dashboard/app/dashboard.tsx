"use client";
import ProjectForecast from "./project-forecast";
import BenchmarkPanel from "./benchmark-panel";
import AdminLogin from "./admin-login";
import { apiFetch, teamKey } from "../lib/client-api";
import { useEffect, useState, useCallback } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Box,
  Check,
  ChevronRight,
  CircleHelp,
  Cpu,
  FlaskConical,
  GitBranch,
  Code2 as Github,
  LayoutDashboard,
  Plus,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
  Users,
  Vote,
  X,
} from "lucide-react";
import {
  initialPolls,
  phases,
  defaultConfig,
  parameterBudget,
  applyDecision,
  type ModelConfig,
  type Poll,
} from "../lib/project";

type Ballot = { pollId: string; userId: string; name: string; choice: number };
type Resolution = {
  pollId: string;
  selected: number | null;
  updatedBy: string;
  updatedAt: string;
};
type Milestone = {
  id: string;
  status: string;
  updatedBy: string;
  updatedAt: string;
};
type Run = {
  id: string;
  name: string;
  tokens: number;
  loss: number | null;
  notes: string;
  createdBy: string;
  createdAt: string;
};
type Team = {
  polls: Poll[];
  votes: Ballot[];
  resolutions: Resolution[];
  milestones: Milestone[];
  runs: Run[];
  user: { id: string; name: string } | null;
};
const empty: Team = {
  polls: initialPolls,
  votes: [],
  resolutions: [],
  milestones: [],
  runs: [],
  user: null,
};
const roster = [
  "onlyspxctre",
  "michaelkrymski",
  "jsaluja09",
  "jacklowry08",
  "carlimarquilage-ctrl",
];
const million = (n: number) =>
  `${(n / 1e6).toFixed(n >= 1e9 ? 0 : 2).replace(/\.00$/, "")}M`;
const number = (n: number) =>
  new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(n);
const nav = [
  { id: "overview", name: "Overview", icon: LayoutDashboard },
  { id: "decisions", name: "Team decisions", icon: Vote },
  { id: "model", name: "Model lab", icon: Box },
  { id: "training", name: "Training", icon: Activity },
  { id: "benchmark", name: "Benchmark", icon: Cpu },
];

export default function Dashboard() {
  const [tab, setTab] = useState("overview"),
    [team, setTeam] = useState<Team>(empty),
    [connected, setConnected] = useState(false),
    [loaded, setLoaded] = useState(false),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [message, setMessage] = useState("");
  const [config, setConfig] = useState<ModelConfig>(defaultConfig),
    [filter, setFilter] = useState("All"),
    [query, setQuery] = useState(""),
    [newPoll, setNewPoll] = useState(false),
    [newRun, setNewRun] = useState(false);
  const [speed, setSpeed] = useState(20000),
    [hourly, setHourly] = useState(0),
    [simHours, setSimHours] = useState(0),
    [playing, setPlaying] = useState(false);
  const budget = parameterBudget(config),
    closed = team.resolutions.filter((r) => r.selected != null),
    done = team.milestones.filter((m) => m.status === "done").length;
  const progress = Math.round((done / phases.length) * 100),
    hours = config.tokens / speed / 3600;
  const reload = useCallback(async () => {
    if (!teamKey()) {
      setTeam(empty);
      setConnected(false);
      setLoaded(true);
      setError("");
      return;
    }
    try {
      const res = await apiFetch("/api/team");
      const data = (await res.json()) as Team & { error?: string };
      if (!res.ok) throw new Error(data.error ?? "Could not load team data.");
      setTeam(data);
      setConnected(true);
      setError("");
    } catch (e) {
      setConnected(false);
      setError((e as Error).message);
    } finally {
      setLoaded(true);
    }
  }, []);
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [tab]);
  useEffect(() => {
    void reload();
    window.addEventListener("glm-auth", reload);
    const timer = setInterval(() => void reload(), 15000);
    return () => {
      clearInterval(timer);
      window.removeEventListener("glm-auth", reload);
    };
  }, [reload]);
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(
      () =>
        setSimHours((h) => {
          if (h >= hours) {
            setPlaying(false);
            return hours;
          }
          return Math.min(hours, h + hours / 80);
        }),
      100,
    );
    return () => clearInterval(t);
  }, [playing, hours]);
  async function mutate(body: object) {
    setBusy(true);
    setMessage("");
    try {
      const res = await apiFetch("/api/team", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = (await res.json()) as Team & { error?: string };
      if (!res.ok) throw new Error(data.error ?? "Could not save.");
      await reload();
      setMessage("Saved to the team workspace.");
      return true;
    } catch (e) {
      setError((e as Error).message);
      return false;
    } finally {
      setBusy(false);
    }
  }
  function applyApproved() {
    let next = { ...defaultConfig };
    for (const r of closed) next = applyDecision(next, r.pollId, r.selected!);
    setConfig(next);
    setMessage(
      "Approved choices applied. Undecided settings use the baseline.",
    );
  }
  function exportData() {
    const blob = new Blob(
      [
        JSON.stringify(
          {
            exportedAt: new Date().toISOString(),
            team,
            simulation: config,
            assumptions:
              "Bias-free dense attention, GELU FFN, RMSNorm, RoPE; not a measured training run.",
          },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob),
      a = document.createElement("a");
    a.href = url;
    a.download = "glm-team-2.json";
    a.click();
    URL.revokeObjectURL(url);
  }
  function impact(p: Poll, i: number) {
    const next = applyDecision(config, p.id, i),
      n = parameterBudget(next).total;
    if (["vocab", "width", "layers", "ffn", "tied"].includes(p.id))
      return `${million(n)} total · ${Math.round((n / 25e6) * 100)}% of budget`;
    if (p.id === "context")
      return "0 additional position parameters · changes attention cost";
    if (p.id === "data")
      return `${(next.tokens / budget.total).toFixed(1)} tokens / parameter`;
    if (p.id === "batch")
      return `${number(Math.ceil(config.tokens / next.batch))} optimizer steps`;
    return "Quality / workflow choice; no direct parameter change";
  }
  function DecisionCard({
    poll: p,
    compact = false,
  }: {
    poll: Poll;
    compact?: boolean;
  }) {
    const ballots = team.votes.filter((v) => v.pollId === p.id),
      mine = ballots.find((v) => v.userId === team.user?.id),
      resolution = team.resolutions.find((r) => r.pollId === p.id),
      resolved = resolution?.selected != null;
    const counts = p.options.map(
        (_, i) => ballots.filter((v) => v.choice === i).length,
      ),
      max = Math.max(...counts),
      leaders = counts
        .map((n, i) => (n === max && n > 0 ? i : -1))
        .filter((i) => i >= 0);
    return (
      <article className={`card decision-card ${compact ? "compact" : ""}`}>
        <div className="card-top">
          <span className="eyebrow">{p.category}</span>
          <span className={`badge ${resolved ? "green" : ""}`}>
            {resolved ? "Decided" : "Open vote"}
          </span>
        </div>
        <h3>{p.title}</h3>
        <p className="muted description">{p.description}</p>
        <div className="vote-options">
          {p.options.map((option, i) => (
            <button
              key={option}
              disabled={!connected || busy || resolved}
              className={`vote-option ${mine?.choice === i ? "chosen" : ""} ${resolved && resolution.selected === i ? "approved" : ""}`}
              onClick={() =>
                void mutate({ action: "vote", pollId: p.id, choice: i })
              }
            >
              <span
                className="vote-fill"
                style={{
                  width: `${ballots.length ? (counts[i] / ballots.length) * 100 : 0}%`,
                }}
              />
              <span className="option-text">
                <span className="radio">
                  {(mine?.choice === i ||
                    (resolved && resolution.selected === i)) && (
                    <Check size={11} />
                  )}
                </span>
                {option}
              </span>
              <span className="vote-count">{counts[i]}</span>
            </button>
          ))}
        </div>
        {!compact && (
          <div className="impact-note">
            <FlaskConical size={14} />
            <span>
              {impact(
                p,
                mine?.choice ?? resolution?.selected ?? leaders[0] ?? 0,
              )}
            </span>
          </div>
        )}
        <div className="vote-footer">
          <span>
            {ballots.length} vote{ballots.length !== 1 ? "s" : ""}
            {mine ? " · yours saved" : ""}
          </span>
          {mine && !resolved && (
            <button
              className="text-button"
              disabled={busy || !connected}
              onClick={() =>
                void mutate({ action: "vote", pollId: p.id, choice: null })
              }
            >
              Retract
            </button>
          )}
        </div>
        {!compact && (
          <details className="decision-detail">
            <summary>Team record & decision controls</summary>
            <p>
              {ballots.length
                ? ballots
                    .map((v) => `${v.name}: ${p.options[v.choice]}`)
                    .join(" · ")
                : "No votes yet."}
            </p>
            {resolved ? (
              <>
                <p>Selected by {resolution.updatedBy}. Votes are closed.</p>
                <button
                  className="secondary small"
                  disabled={!connected || busy}
                  onClick={() =>
                    void mutate({
                      action: "resolve",
                      pollId: p.id,
                      choice: null,
                    })
                  }
                >
                  Reopen voting
                </button>
              </>
            ) : (
              <>
                <p>
                  Any teammate can finalize an option after discussion. Ties
                  stay open until the team chooses.
                </p>
                <select
                  aria-label={`Finalize ${p.title}`}
                  defaultValue=""
                  disabled={!connected || busy}
                  onChange={(e) => {
                    if (e.target.value !== "")
                      void mutate({
                        action: "resolve",
                        pollId: p.id,
                        choice: Number(e.target.value),
                      });
                    e.target.value = "";
                  }}
                >
                  <option value="" disabled>
                    Finalize a team choice…
                  </option>
                  {p.options.map((o, i) => (
                    <option key={o} value={i}>
                      {o}
                    </option>
                  ))}
                </select>
              </>
            )}
          </details>
        )}
      </article>
    );
  }
  function BudgetChart() {
    const parts = [
      { name: "Embeddings", value: budget.embeddings, color: "#c6ec96" },
      { name: "Attention", value: budget.attention, color: "#62a58b" },
      { name: "Feed-forward", value: budget.feedForward, color: "#213c34" },
      { name: "Output head", value: budget.output, color: "#cda974" },
      { name: "Norms", value: budget.norms, color: "#a9b9b0" },
    ];
    return (
      <>
        <div className="budget-number">
          {million(budget.total)}
          <span>/ 25M parameters</span>
        </div>
        <div
          className="stacked"
          role="img"
          aria-label={parts
            .map((p) => `${p.name}: ${number(p.value)}`)
            .join(", ")}
        >
          {parts
            .filter((p) => p.value > 0)
            .map((p) => (
              <div
                key={p.name}
                style={{
                  width: `${(p.value / Math.max(25e6, budget.total)) * 100}%`,
                  background: p.color,
                }}
              />
            ))}
          {budget.total < 25e6 && (
            <div style={{ flex: 1, background: "#e8ece4" }} />
          )}
        </div>
        <div className="legend">
          {parts
            .filter((p) => p.value > 0)
            .map((p) => (
              <span key={p.name}>
                <i style={{ background: p.color }} />
                {p.name}
                <b>{million(p.value)}</b>
              </span>
            ))}
        </div>
        <div className={`budget-status ${budget.total > 25e6 ? "over" : ""}`}>
          <span>{budget.total > 25e6 ? "Over target" : "Within target"}</span>
          <b>
            {million(Math.abs(25e6 - budget.total))}{" "}
            {budget.total > 25e6 ? "over" : "remaining"}
          </b>
        </div>
      </>
    );
  }
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setTab("overview");
          }}
        >
          <span className="brand-mark">
            <GitBranch size={23} />
          </span>
          <span>
            Compute
            <small>DISTRIBUTED COMPUTE</small>
          </span>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {nav.map((n) => (
            <button
              key={n.id}
              className={tab === n.id ? "active" : ""}
              onClick={() => setTab(n.id)}
            >
              <n.icon size={18} />
              {n.name}
              {n.id === "decisions" && (
                <span className="nav-count">
                  {team.polls.length - closed.length}
                </span>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-project">
          <div className="eyebrow">THE EXPERIMENT</div>
          <h3>Small by design.</h3>
          <p>
            25 million parameters.
            <br />
            500 million tokens.
            <br />
            One shared baseline.
          </p>
          <div className="mini-grid">
            {Array.from({ length: 24 }, (_, i) => (
              <i key={i} style={{ opacity: 0.18 + (i % 7) / 8 }} />
            ))}
          </div>
          <span className="tiny">DISTRIBUTED COMPUTE</span>
        </div>
        <div className="sidebar-bottom">
          <a
            href="https://github.com/michaelkrymski/distributed-compute"
            target="_blank"
            rel="noreferrer"
          >
            <Github size={17} />
            Project repository
            <ArrowRight size={14} />
          </a>
          <div className="profile">
            <div className="avatar">
              {team.user?.name.charAt(0).toUpperCase() ?? "G"}
            </div>
            <span>
              {team.user?.name ?? "Team workspace"}
              <small>
                {connected
                  ? "Signed in · shared workspace"
                  : "Connect with your team key"}
              </small>
            </span>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <span>
            Compute <ChevronRight size={13} /> Team <ChevronRight size={13} />
            <b>{nav.find((n) => n.id === tab)?.name}</b>
          </span>
          <div>
            <span className={`connection ${connected ? "online" : ""}`}>
              <i />
              {connected
                ? "Team data synced"
                : loaded
                  ? "Not connected"
                  : "Connecting…"}
            </span>
            <button
              className="icon-button"
              aria-label="Refresh team data"
              onClick={() => void reload()}
            >
              <RefreshCw size={16} />
            </button>
            <a
              className="github-link"
              href="https://github.com/michaelkrymski/distributed-compute"
              target="_blank"
              rel="noreferrer"
            >
              <Github size={17} />
              <span>GitHub</span>
            </a>
          </div>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <div className="eyebrow">DISTRIBUTED COMPUTE / EXPERIMENT</div>
              <h1>
                {tab === "overview"
                  ? "Small model. Big decisions."
                  : tab === "decisions"
                    ? "Build it together."
                    : tab === "model"
                      ? "Every parameter counts."
                      : tab === "benchmark"
                        ? "Measure the real workload."
                        : "From pilot to baseline."}
              </h1>
              <p>
                {tab === "overview"
                  ? "A shared home for the choices, experiments, and work ahead."
                  : tab === "decisions"
                    ? "Compare the tradeoffs, cast your vote, and agree on a baseline."
                    : tab === "model"
                      ? "Explore the architecture before spending the compute."
                      : tab === "benchmark"
                        ? "Connect the whole team, warm up together, and measure the shared run."
                        : "Plan the compute. Record the evidence. Keep the team in sync."}
              </p>
            </div>
            <button className="secondary export" onClick={exportData}>
              <ArrowDownToLine size={16} />
              Export workspace
            </button>
          </div>
          <AdminLogin connected={connected} name={team.user?.name} />
          {error && (
            <div className="notice error" role="alert">
              <CircleHelp size={17} />
              <span>
                {error}
                {!team.user &&
                  " The model lab and training simulator are available below."}
              </span>
              {!team.user && <span>Sign in or connect with your member key above.</span>}
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={15} />
              </button>
            </div>
          )}
          {message && (
            <div className="notice success" role="status">
              <Check size={16} />
              {message}
              <button
                aria-label="Dismiss notification"
                onClick={() => setMessage("")}
              >
                <X size={15} />
              </button>
            </div>
          )}
          {tab === "benchmark" && <BenchmarkPanel />}
          {tab === "overview" && (
            <>
              <section className="metrics" aria-label="Project summary">
                <div>
                  <span>PARAMETER TARGET</span>
                  <strong>
                    25<span>M</span>
                  </strong>
                  <small>Dense decoder-only transformer</small>
                  <Cpu size={20} />
                </div>
                <div>
                  <span>TRAINING BUDGET</span>
                  <strong>
                    500<span>M</span>
                  </strong>
                  <small>20 tokens per target parameter</small>
                  <Activity size={20} />
                </div>
                <div>
                  <span>DECISIONS MADE</span>
                  <strong>
                    {closed.length}
                    <span> / {team.polls.length}</span>
                  </strong>
                  <small>
                    {team.polls.length - closed.length} choices open for
                    discussion
                  </small>
                  <Vote size={20} />
                </div>
                <div>
                  <span>PROJECT PROGRESS</span>
                  <strong>
                    {progress}
                    <span>%</span>
                  </strong>
                  <small>{done} of 6 milestones complete</small>
                  <GitBranch size={20} />
                </div>
              </section>
              <div className="overview-grid">
                <section className="card journey">
                  <div className="section-title">
                    <div>
                      <span className="eyebrow">THE ROAD TO A BASELINE</span>
                      <h2>One experiment, six milestones</h2>
                    </div>
                    <span className="badge">Team-reported</span>
                  </div>
                  <div className="journey-progress">
                    <div style={{ width: `${progress}%` }} />
                  </div>
                  {phases.map((p, i) => {
                    const status =
                        team.milestones.find((m) => m.id === p.id)?.status ??
                        "not_started",
                      decisions = p.tasks.filter((t) =>
                        closed.some((r) => r.pollId === t),
                      ).length;
                    return (
                      <div className={`phase ${status}`} key={p.id}>
                        <div className="phase-number">
                          {status === "done" ? (
                            <Check size={15} />
                          ) : (
                            String(i + 1).padStart(2, "0")
                          )}
                        </div>
                        <div className="phase-copy">
                          <h3>{p.title}</h3>
                          <p>{p.detail}</p>
                          <button
                            className="text-button"
                            onClick={() => {
                              setTab("decisions");
                              setFilter("All");
                              setQuery("");
                            }}
                          >
                            {decisions}/{p.tasks.length} related decisions
                            agreed <ArrowRight size={12} />
                          </button>
                        </div>
                        <select
                          aria-label={`${p.title} status`}
                          value={status}
                          disabled={!connected || busy}
                          onChange={(e) =>
                            void mutate({
                              action: "milestone",
                              id: p.id,
                              status: e.target.value,
                            })
                          }
                        >
                          <option value="not_started">Not started</option>
                          <option value="in_progress">In progress</option>
                          <option value="done">Complete</option>
                        </select>
                      </div>
                    );
                  })}
                </section>
                <div className="right-column">
                  <section className="card budget-preview">
                    <div className="section-title">
                      <div>
                        <span className="eyebrow">BASELINE SKETCH</span>
                        <h2>Where the parameters go</h2>
                      </div>
                      <Box size={19} />
                    </div>
                    <BudgetChart />
                    <p className="caption">
                      Local simulation · {config.layers} layers · d=
                      {config.width} · {number(config.vocab)} vocabulary
                    </p>
                    <button
                      className="secondary full"
                      onClick={() => setTab("model")}
                    >
                      Open model lab <ArrowRight size={15} />
                    </button>
                  </section>
                  <section className="next-step">
                    <span className="eyebrow">
                      <Sparkles size={14} />
                      START HERE
                    </span>
                    <h2>
                      Data first.
                      <br />
                      Tokenizer second.
                    </h2>
                    <p>
                      Agree on a corpus, then train a small tokenizer on the
                      training split. Freeze it before counting your 500M
                      tokens.
                    </p>
                    <button
                      onClick={() => {
                        setTab("decisions");
                        setFilter("Tokenizer");
                      }}
                    >
                      Discuss the tokenizer <ArrowRight size={15} />
                    </button>
                  </section>
                </div>
              </div>
              <ProjectForecast />
              <section className="section-title below">
                <div>
                  <span className="eyebrow">ON THE TABLE</span>
                  <h2>Start with these decisions</h2>
                </div>
                <button
                  className="text-button"
                  onClick={() => setTab("decisions")}
                >
                  All decisions <ArrowRight size={15} />
                </button>
              </section>
              <div className="decision-grid overview-decisions">
                {team.polls
                  .filter((p) =>
                    ["corpus", "tokenizer", "vocab"].includes(p.id),
                  )
                  .map((p) => (
                    <DecisionCard key={p.id} poll={p} compact />
                  ))}
              </div>
              <section className="team-strip">
                <div>
                  <Users size={18} />
                  <b>The team</b>
                  <span>GitHub team roster · Nov 12, 2026</span>
                </div>
                <div>
                  {roster.map((name, i) => (
                    <a
                      key={name}
                      title={name}
                      href={`https://github.com/${name}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      <span
                        style={{
                          background: [
                            "#dbe8bf",
                            "#dce5ef",
                            "#f0dfc4",
                            "#e4dcf0",
                            "#d0e8df",
                          ][i],
                        }}
                      >
                        {name.slice(0, 2).toUpperCase()}
                      </span>
                      {name}
                    </a>
                  ))}
                </div>
              </section>
            </>
          )}
          {tab === "decisions" && (
            <>
              <div className="decision-toolbar">
                <div className="filters">
                  {[
                    "All",
                    "Data",
                    "Tokenizer",
                    "Architecture",
                    "Training",
                    "Evaluation",
                    "Team",
                  ].map((c) => (
                    <button
                      key={c}
                      className={filter === c ? "selected" : ""}
                      onClick={() => setFilter(c)}
                    >
                      {c}
                    </button>
                  ))}
                </div>
                <button
                  className="primary"
                  disabled={!connected}
                  onClick={() => setNewPoll(true)}
                >
                  <Plus size={16} />
                  New decision
                </button>
              </div>
              <div className="decision-subbar">
                <input
                  aria-label="Search decisions"
                  placeholder="Search decisions…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <span>
                  One vote per person · change your vote while open · refreshes
                  every 15 seconds
                </span>
              </div>
              <div className="decision-grid">
                {team.polls
                  .filter(
                    (p) =>
                      (filter === "All" || p.category === filter) &&
                      (p.title + " " + p.description)
                        .toLowerCase()
                        .includes(query.toLowerCase()),
                  )
                  .map((p) => (
                    <DecisionCard key={p.id} poll={p} />
                  ))}
              </div>
              {!team.polls.some(
                (p) =>
                  (filter === "All" || p.category === filter) &&
                  (p.title + " " + p.description)
                    .toLowerCase()
                    .includes(query.toLowerCase()),
              ) && (
                <div className="empty-state">
                  No matching decisions. Try another category or search.
                </div>
              )}
              <p className="caption">
                All admitted teammates can vote, finalize decisions, and update
                milestones. Finalizing is a team action, not an automatic
                majority rule.
              </p>
            </>
          )}
          {tab === "model" && (
            <>
              <div className="lab-banner">
                <span>
                  <FlaskConical size={18} />
                  <b>Interactive simulation</b> · Changes here are local. They
                  do not change votes or train a model.
                </span>
                <button className="text-button" onClick={applyApproved}>
                  Apply approved choices <ArrowRight size={14} />
                </button>
              </div>
              <div className="lab-grid">
                <section className="card configuration">
                  <div className="section-title">
                    <h2>Architecture controls</h2>
                    <SlidersHorizontal size={18} />
                  </div>
                  {[
                    {
                      key: "vocab",
                      name: "Vocabulary",
                      values: [256, 4096, 8192, 16384, 32768],
                    },
                    {
                      key: "width",
                      name: "Hidden width",
                      values: [288, 384, 480],
                    },
                    {
                      key: "layers",
                      name: "Transformer layers",
                      values: [8, 12, 16],
                    },
                    {
                      key: "expansion",
                      name: "Feed-forward expansion",
                      values: [3, 4],
                    },
                    {
                      key: "context",
                      name: "Context length",
                      values: [256, 512, 1024, 2048],
                    },
                  ].map((field) => (
                    <label className="field" key={field.key}>
                      {field.name}
                      <select
                        value={config[field.key as keyof ModelConfig] as number}
                        onChange={(e) =>
                          setConfig((c) => ({
                            ...c,
                            [field.key]: Number(e.target.value),
                          }))
                        }
                      >
                        {field.values.map((v) => (
                          <option key={v} value={v}>
                            {number(v)}
                            {field.key === "expansion" ? "×" : ""}
                          </option>
                        ))}
                      </select>
                    </label>
                  ))}
                  <label className="check-field">
                    <input
                      type="checkbox"
                      checked={config.tied}
                      onChange={(e) =>
                        setConfig((c) => ({ ...c, tied: e.target.checked }))
                      }
                    />
                    <span>
                      Tie input & output embeddings
                      <small>Reuse the vocabulary weight matrix</small>
                    </span>
                  </label>
                  <button
                    className="secondary full"
                    onClick={() => setConfig(defaultConfig)}
                  >
                    Reset baseline
                  </button>
                </section>
                <section className="card parameter-card">
                  <div className="section-title">
                    <div>
                      <span className="eyebrow">
                        25,000,000 PARAMETER TARGET
                      </span>
                      <h2>Your parameter budget</h2>
                    </div>
                    <span
                      className={`badge ${budget.total > 25e6 ? "orange" : "green"}`}
                    >
                      {Math.round((budget.total / 25e6) * 100)}% used
                    </span>
                  </div>
                  <BudgetChart />
                  <h3 className="chart-heading">
                    Vocabulary is an architecture decision.
                  </h3>
                  <p className="muted">
                    Embedding share at the current width. Each bar is a fraction
                    of the 25M target.
                  </p>
                  <div className="vocab-bars">
                    {[4096, 8192, 16384, 32768].map((v) => {
                      const total = v * config.width * (config.tied ? 1 : 2);
                      return (
                        <button
                          key={v}
                          className={config.vocab === v ? "selected" : ""}
                          onClick={() => setConfig((c) => ({ ...c, vocab: v }))}
                        >
                          <span>{number(v)}</span>
                          <span className="bar-track">
                            <i
                              style={{
                                width: `${Math.min(100, (total / 25e6) * 100)}%`,
                              }}
                            />
                          </span>
                          <b>{((total / 25e6) * 100).toFixed(1)}%</b>
                        </button>
                      );
                    })}
                  </div>
                  <div className="mini-stats">
                    <div>
                      <span>Tokens / parameter</span>
                      <strong>
                        {(config.tokens / budget.total).toFixed(1)}:1
                      </strong>
                    </div>
                    <div>
                      <span>BF16 weights only</span>
                      <strong>
                        {((budget.total * 2) / 1e6).toFixed(1)} MB
                      </strong>
                    </div>
                    <div>
                      <span>Attention heads</span>
                      <strong>
                        6 <small>× {config.width / 6} dims</small>
                      </strong>
                    </div>
                  </div>
                </section>
              </div>
              <section className="card assumptions">
                <CircleHelp size={20} />
                <div>
                  <h3>What this estimate includes</h3>
                  <p>
                    Bias-free dense Q/K/V/output attention, a two-matrix GELU
                    feed-forward block, two RMSNorms per layer, a final RMSNorm,
                    and RoPE with no learned position table. Total = Vd × (tied
                    ? 1 : 2) + L(4d² + 2df + 2d) + d; f = expansion × d. Actual
                    implementation must verify this count.
                  </p>
                  <p>
                    Context length affects attention compute and activation
                    memory, not this parameter count. The memory figure includes
                    weights only; gradients, optimizer states, and activations
                    need additional GPU memory. Tokenizer algorithm and
                    precision choices do not automatically modify these
                    architecture controls.
                  </p>
                </div>
              </section>
            </>
          )}
          {tab === "training" && (
            <>
              <div className="training-grid">
                <section className="card training-inputs">
                  <div className="section-title">
                    <h2>Training planner</h2>
                    <Activity size={19} />
                  </div>
                  <span className="badge orange">
                    Hypothetical until benchmarked
                  </span>
                  <label className="field">
                    Training tokens
                    <select
                      value={config.tokens}
                      onChange={(e) => {
                        setConfig((c) => ({
                          ...c,
                          tokens: Number(e.target.value),
                        }));
                        setSimHours(0);
                      }}
                    >
                      {[250e6, 500e6, 1e9].map((v) => (
                        <option key={v} value={v}>
                          {million(v)} tokens
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    Effective tokens per optimizer step
                    <select
                      value={config.batch}
                      onChange={(e) =>
                        setConfig((c) => ({
                          ...c,
                          batch: Number(e.target.value),
                        }))
                      }
                    >
                      {[16384, 32768, 65536].map((v) => (
                        <option key={v} value={v}>
                          {number(v)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    Assumed throughput (tokens / second)
                    <input
                      type="number"
                      min="1"
                      max="10000000"
                      value={speed}
                      onChange={(e) => {
                        setSpeed(
                          Math.max(
                            1,
                            Math.min(1e7, Number(e.target.value) || 1),
                          ),
                        );
                        setSimHours(0);
                      }}
                    />
                  </label>
                  <label className="field">
                    Compute cost (USD / hour)
                    <input
                      type="number"
                      min="0"
                      max="10000"
                      step="0.01"
                      value={hourly}
                      onChange={(e) =>
                        setHourly(
                          Math.max(
                            0,
                            Math.min(1e4, Number(e.target.value) || 0),
                          ),
                        )
                      }
                    />
                  </label>
                  <p className="caption">
                    Measure tokens/second on your actual GPU during the pilot.
                    Estimates exclude validation, checkpointing, startup, and
                    interruptions.
                  </p>
                </section>
                <section className="card training-simulation">
                  <div className="section-title">
                    <div>
                      <span className="eyebrow">COMPUTE BEFORE COMMITMENT</span>
                      <h2>See the run ahead</h2>
                    </div>
                    <span className="badge">Simulation</span>
                  </div>
                  <div className="training-stats">
                    <div>
                      <span>Estimated runtime</span>
                      <strong>
                        {hours.toFixed(1)}
                        <small> hours</small>
                      </strong>
                    </div>
                    <div>
                      <span>Optimizer steps</span>
                      <strong>
                        {number(Math.ceil(config.tokens / config.batch))}
                      </strong>
                    </div>
                    <div>
                      <span>Compute estimate</span>
                      <strong>
                        {hourly ? `$${number(hours * hourly)}` : "—"}
                      </strong>
                      <small>
                        {hourly
                          ? "at your hourly rate"
                          : "Enter an hourly rate"}
                      </small>
                    </div>
                  </div>
                  <svg
                    className="training-chart"
                    viewBox="0 0 650 220"
                    role="img"
                    aria-label="Simulated tokens processed over time at constant throughput"
                  >
                    <defs>
                      <linearGradient id="area" x1="0" y1="0" x2="0" y2="1">
                        <stop
                          offset="0%"
                          stopColor="#95c877"
                          stopOpacity=".35"
                        />
                        <stop
                          offset="100%"
                          stopColor="#95c877"
                          stopOpacity=".03"
                        />
                      </linearGradient>
                    </defs>
                    {[30, 80, 130, 180].map((y) => (
                      <line
                        key={y}
                        x1="55"
                        x2="625"
                        y1={y}
                        y2={y}
                        stroke="#e4e8df"
                        strokeDasharray="4 5"
                      />
                    ))}
                    <path d="M55 180 L625 30 L625 180Z" fill="url(#area)" />
                    <path
                      d="M55 180 L625 30"
                      fill="none"
                      stroke="#83ac68"
                      strokeWidth="3"
                    />
                    <line
                      x1={55 + 570 * Math.min(1, simHours / hours)}
                      x2={55 + 570 * Math.min(1, simHours / hours)}
                      y1="25"
                      y2="180"
                      stroke="#253c32"
                      strokeDasharray="4 3"
                    />
                    <circle
                      cx={55 + 570 * Math.min(1, simHours / hours)}
                      cy={180 - 150 * Math.min(1, simHours / hours)}
                      r="6"
                      fill="#253c32"
                    />
                    <text x="8" y="35">
                      {million(config.tokens)}
                    </text>
                    <text x="24" y="184">
                      0
                    </text>
                    <text x="55" y="210">
                      0 hours
                    </text>
                    <text x="570" y="210">
                      {hours.toFixed(1)} hours
                    </text>
                  </svg>
                  <div className="simulation-controls">
                    <button
                      className="primary small"
                      onClick={() => {
                        if (simHours >= hours) setSimHours(0);
                        setPlaying(!playing);
                      }}
                    >
                      {playing ? "Pause" : "Play forecast"}
                    </button>
                    <input
                      aria-label="Simulated elapsed training hours"
                      type="range"
                      min="0"
                      max={hours}
                      step={hours / 1000}
                      value={Math.min(hours, simHours)}
                      onChange={(e) => {
                        setPlaying(false);
                        setSimHours(Number(e.target.value));
                      }}
                    />
                    <b>{Math.round(Math.min(1, simHours / hours) * 100)}%</b>
                  </div>
                  <div className="forecast-label">
                    {million(Math.min(config.tokens, simHours * 3600 * speed))}{" "}
                    simulated tokens · {Math.min(hours, simHours).toFixed(1)}{" "}
                    elapsed hours
                  </div>
                  <p className="caption">
                    A constant-throughput forecast, not live training telemetry
                    or a predicted loss curve. Approximate training compute:{" "}
                    {((6 * budget.total * config.tokens) / 1e15).toFixed(2)}{" "}
                    PFLOP (6ND; excludes extra attention costs).
                  </p>
                </section>
              </div>
              <section className="section-title below">
                <div>
                  <span className="eyebrow">OBSERVED, NOT SIMULATED</span>
                  <h2>Experiment notebook</h2>
                </div>
                <button
                  className="primary"
                  disabled={!connected}
                  onClick={() => setNewRun(true)}
                >
                  <Plus size={16} />
                  Log a checkpoint
                </button>
              </section>
              {team.runs.length ? (
                <div className="card run-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Experiment / checkpoint</th>
                        <th>Tokens processed</th>
                        <th>Validation loss</th>
                        <th>Recorded by</th>
                      </tr>
                    </thead>
                    <tbody>
                      {team.runs.map((r) => (
                        <tr key={r.id}>
                          <td>
                            <strong>{r.name}</strong>
                            <p>{r.notes}</p>
                            <small>
                              {new Date(r.createdAt).toLocaleString()}
                            </small>
                          </td>
                          <td>{number(r.tokens)}</td>
                          <td>{r.loss ?? "—"}</td>
                          <td>{r.createdBy}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="card empty-state">
                  <FlaskConical size={28} />
                  <h3>The first run starts here.</h3>
                  <p>
                    Log a pilot or checkpoint when you have real measurements.
                    <br />
                    No training results have been recorded yet.
                  </p>
                </div>
              )}
              <div className="training-checklist">
                <h3>Before the full run</h3>
                <span>01 · Overfit a tiny batch</span>
                <span>02 · Check held-out loss</span>
                <span>03 · Benchmark throughput</span>
                <span>04 · Test checkpoint resume</span>
              </div>
            </>
          )}
          <footer>
            <span>
              Distributed Compute <i /> Built for the first baseline.
            </span>
            <span>25M parameters / 500M tokens</span>
          </footer>
        </main>
      </div>
      {newPoll && (
        <div className="modal-overlay">
          <section
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="poll-title"
          >
            <button
              className="modal-close icon-button"
              aria-label="Close new decision"
              onClick={() => setNewPoll(false)}
            >
              <X size={20} />
            </button>
            <div className="eyebrow">MAKE ROOM FOR DISCUSSION</div>
            <h2 id="poll-title">New team decision</h2>
            <form
              onSubmit={async (e) => {
                e.preventDefault();
                const f = new FormData(e.currentTarget);
                if (
                  await mutate({
                    action: "poll",
                    title: f.get("title"),
                    description: f.get("description"),
                    options: String(f.get("options"))
                      .split("\n")
                      .map((s) => s.trim())
                      .filter(Boolean),
                  })
                )
                  setNewPoll(false);
              }}
            >
              <label className="field">
                Question
                <input
                  name="title"
                  required
                  maxLength={120}
                  autoFocus
                  placeholder="What should we decide?"
                />
              </label>
              <label className="field">
                Context
                <textarea
                  name="description"
                  maxLength={700}
                  placeholder="Explain the tradeoff…"
                />
              </label>
              <label className="field">
                Options · one per line (2–6)
                <textarea
                  name="options"
                  required
                  placeholder={"Option A\nOption B"}
                  rows={4}
                />
              </label>
              <p className="caption">
                Custom decisions store votes and the final choice. They do not
                change model-lab formulas.
              </p>
              <button className="primary full" disabled={busy}>
                Create decision
              </button>
            </form>
          </section>
        </div>
      )}
      {newRun && (
        <div className="modal-overlay">
          <section
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="run-title"
          >
            <button
              className="modal-close icon-button"
              aria-label="Close checkpoint form"
              onClick={() => setNewRun(false)}
            >
              <X size={20} />
            </button>
            <div className="eyebrow">A RECORD OF THE REAL RUN</div>
            <h2 id="run-title">Log a checkpoint</h2>
            <form
              onSubmit={async (e) => {
                e.preventDefault();
                const f = new FormData(e.currentTarget);
                if (
                  await mutate({
                    action: "run",
                    name: f.get("name"),
                    tokens: Number(f.get("tokens")),
                    loss: f.get("loss") === "" ? null : Number(f.get("loss")),
                    notes: f.get("notes"),
                  })
                )
                  setNewRun(false);
              }}
            >
              <label className="field">
                Run / checkpoint name
                <input
                  name="name"
                  required
                  maxLength={120}
                  autoFocus
                  placeholder="baseline-001 / step 1000"
                />
              </label>
              <label className="field">
                Cumulative tokens processed
                <input
                  name="tokens"
                  type="number"
                  required
                  min="0"
                  max="1000000000000"
                  step="1"
                />
              </label>
              <label className="field">
                Validation loss · optional
                <input name="loss" type="number" min="0" max="100" step="any" />
              </label>
              <label className="field">
                Notes
                <textarea
                  name="notes"
                  maxLength={1500}
                  placeholder="GPU, config, throughput, sample observations…"
                />
              </label>
              <button className="primary full" disabled={busy}>
                Save checkpoint
              </button>
            </form>
          </section>
        </div>
      )}
    </div>
  );
}

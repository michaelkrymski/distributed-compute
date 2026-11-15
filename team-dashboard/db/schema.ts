import {
  sqliteTable,
  text,
  integer,
  real,
  primaryKey,
} from "drizzle-orm/sqlite-core";
export const adminAccounts = sqliteTable("admin_accounts", {
  id: text("id").primaryKey(),
  salt: text("salt").notNull(),
  passwordHash: text("password_hash").notNull(),
  failures: integer("failures").notNull(),
  lockedUntil: integer("locked_until").notNull(),
});
export const adminSessions = sqliteTable("admin_sessions", {
  tokenHash: text("token_hash").primaryKey(),
  expiresAt: integer("expires_at").notNull(),
});
export const polls = sqliteTable("polls", {
  id: text("id").primaryKey(),
  title: text("title").notNull(),
  description: text("description").notNull(),
  category: text("category").notNull(),
  options: text("options", { mode: "json" }).$type<string[]>().notNull(),
  createdBy: text("created_by").notNull(),
});
export const votes = sqliteTable(
  "votes",
  {
    pollId: text("poll_id").notNull(),
    userId: text("user_id").notNull(),
    name: text("name").notNull(),
    choice: integer("choice").notNull(),
    updatedAt: text("updated_at").notNull(),
  },
  (t) => [primaryKey({ columns: [t.pollId, t.userId] })],
);
export const resolutions = sqliteTable("resolutions", {
  pollId: text("poll_id").primaryKey(),
  selected: integer("selected"),
  updatedBy: text("updated_by").notNull(),
  updatedAt: text("updated_at").notNull(),
});
export const milestones = sqliteTable("milestones", {
  id: text("id").primaryKey(),
  status: text("status").notNull(),
  updatedBy: text("updated_by").notNull(),
  updatedAt: text("updated_at").notNull(),
});
export const runs = sqliteTable("runs", {
  id: text("id").primaryKey(),
  name: text("name").notNull(),
  tokens: integer("tokens").notNull(),
  loss: real("loss"),
  notes: text("notes").notNull(),
  createdBy: text("created_by").notNull(),
  createdAt: text("created_at").notNull(),
});
export const benchmarkSessions = sqliteTable("benchmark_sessions", {
  id: text("id").primaryKey(),
  activeSlot: text("active_slot").unique(),
  createdAt: integer("created_at").notNull(),
  createdBy: text("created_by").notNull(),
  peers: text("peers", { mode: "json" }).$type<string[]>().notNull(),
  expected: text("expected", { mode: "json" }).$type<string[]>().notNull(),
  config: text("config", { mode: "json" })
    .$type<{
      protocol: number;
      payloadElements: number;
      matrixSize: number;
      warmupSeconds: number;
      measureSeconds: number;
      workload?: "synthetic" | "baseline-training";
      trainingTokens?: number;
      batchTokens?: number;
      trainingProfile?: "reference-fp32" | "optimized-fp32" | "optimized-bf16" | "compiled-bf16";
      microbatch?: number;
      localSteps?: number;
      model?: {
        vocab_size: number;
        width: number;
        layers: number;
        heads: number;
        expansion: number;
        context: number;
        tied: boolean;
      };
    }>()
    .notNull(),
  coordinatorSeenAt: integer("coordinator_seen_at").notNull(),
  startsAt: integer("starts_at"),
  cancelledAt: integer("cancelled_at"),
});
export const benchmarkAgents = sqliteTable(
  "benchmark_agents",
  {
    sessionId: text("session_id").notNull(),
    userId: text("user_id").notNull(),
    instanceId: text("instance_id").notNull(),
    device: text("device", { mode: "json" })
      .$type<{
        name: string;
        kind: string;
        torch: string;
        platform: string;
        workloadFingerprint?: string;
        trainingProfile?: string;
        microbatch?: number;
        localSteps?: number;
        compiledModel?: boolean;
      }>()
      .notNull(),
    lastSeenAt: integer("last_seen_at").notNull(),
    requestedReady: integer("requested_ready", { mode: "boolean" }).notNull(),
    ready: integer("ready", { mode: "boolean" }).notNull(),
    result: text("result", { mode: "json" }).$type<Record<string, unknown>>(),
  },
  (t) => [primaryKey({ columns: [t.sessionId, t.userId] })],
);

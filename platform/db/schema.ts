import { sqliteTable, text, integer, primaryKey } from "drizzle-orm/sqlite-core";
export const accounts = sqliteTable("platform_accounts", {
  id: text("id").primaryKey(), username: text("username").notNull().unique(), salt: text("salt").notNull(), passwordHash: text("password_hash").notNull(), createdAt: integer("created_at").notNull(),
});
export const sessions = sqliteTable("platform_sessions", {
  tokenHash: text("token_hash").primaryKey(), accountId: text("account_id").notNull().references(() => accounts.id, { onDelete: "cascade" }), expiresAt: integer("expires_at").notNull(),
});
export const teams = sqliteTable("compute_teams", { id: text("id").primaryKey(), name: text("name").notNull(), createdAt: integer("created_at").notNull() });
export const members = sqliteTable("team_members", {
  teamId: text("team_id").notNull().references(() => teams.id, { onDelete: "cascade" }), accountId: text("account_id").notNull().references(() => accounts.id, { onDelete: "cascade" }), role: text("role").notNull(),
}, t => [primaryKey({ columns: [t.teamId, t.accountId] })]);
export const projects = sqliteTable("compute_projects", {
  id: text("id").primaryKey(), teamId: text("team_id").notNull().references(() => teams.id, { onDelete: "cascade" }), spec: text("spec").notNull(), revision: integer("revision").notNull(), createdAt: integer("created_at").notNull(),
});
export const keys = sqliteTable("worker_keys", {
  id: text("id").primaryKey(), teamId: text("team_id").notNull().references(() => teams.id, { onDelete: "cascade" }), projectId: text("project_id").notNull().references(() => projects.id, { onDelete: "cascade" }), accountId: text("account_id").notNull().references(() => accounts.id, { onDelete: "cascade" }), tokenHash: text("token_hash").notNull().unique(), prefix: text("prefix").notNull(), label: text("label").notNull(), generation: integer("generation").notNull(), createdAt: integer("created_at").notNull(), revokedAt: integer("revoked_at"),
});
export const invitations = sqliteTable("team_invitations", {
  id: text("id").primaryKey(), teamId: text("team_id").notNull().references(() => teams.id, { onDelete: "cascade" }), tokenHash: text("token_hash").notNull().unique(), createdBy: text("created_by").notNull(), expiresAt: integer("expires_at").notNull(), claimedBy: text("claimed_by"),
});
export const states = sqliteTable("coordination_states", {
  projectId: text("project_id").notNull().references(() => projects.id, { onDelete: "cascade" }), kind: text("kind").notNull(), revision: integer("revision").notNull(), payload: text("payload").notNull(),
}, t => [primaryKey({ columns: [t.projectId, t.kind] })]);
export const rates = sqliteTable("rate_limits", { id: text("id").primaryKey(), hits: integer("hits").notNull(), expiresAt: integer("expires_at").notNull() });
export const auditEvents = sqliteTable("audit_events", { id: text("id").primaryKey(), teamId: text("team_id").notNull(), actorId: text("actor_id").notNull(), action: text("action").notNull(), target: text("target").notNull(), createdAt: integer("created_at").notNull() });

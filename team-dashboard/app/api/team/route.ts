import { getTeamUser, apiJson as json, apiOptions } from "../../../lib/access";
import { getDb } from "../../../db";
import {
  polls,
  votes,
  resolutions,
  milestones,
  runs,
} from "../../../db/schema";
import { initialPolls, phases } from "../../../lib/project";
import { eq, and, desc, sql } from "drizzle-orm";
export const dynamic = "force-dynamic";
export const OPTIONS = apiOptions;
export async function GET(request: Request) {
  const user = await getTeamUser(request);
  if (!user)
    return json(
      { error: "Enter your team access key to view shared records." },
      401,
    );
  try {
    const db = getDb();
    const [custom, ballots, decisions, progress, logs] = await Promise.all([
      db.select().from(polls),
      db.select().from(votes),
      db.select().from(resolutions),
      db.select().from(milestones),
      db.select().from(runs).orderBy(desc(runs.createdAt)).limit(100),
    ]);
    return json({
      polls: [...initialPolls, ...custom],
      votes: ballots,
      resolutions: decisions,
      milestones: progress,
      runs: logs,
      user: { id: user.userId, name: user.displayName },
    });
  } catch (error) {
    console.error("Team data read failed", error);
    return json(
      { error: "Team database unavailable. Your changes have not been saved." },
      503,
    );
  }
}
export async function POST(request: Request) {
  const user = await getTeamUser(request);
  if (!user)
    return json(
      { error: "Enter your team access key before making changes." },
      401,
    );
  if (!request.headers.get("content-type")?.includes("application/json"))
    return json({ error: "JSON required." }, 415);
  const raw = await request.text();
  if (raw.length > 12000) return json({ error: "Request too large." }, 413);
  let body;
  try {
    body = JSON.parse(raw);
  } catch {
    return json({ error: "Invalid JSON." }, 400);
  }
  if (!body || typeof body !== "object")
    return json({ error: "Invalid request." }, 400);
  const now = new Date().toISOString();
  try {
    const db = getDb();
    if (body.action === "vote" || body.action === "resolve") {
      if (typeof body.pollId !== "string")
        return json({ error: "Choose a decision." }, 400);
      const poll =
        initialPolls.find((p) => p.id === body.pollId) ??
        (await db.select().from(polls).where(eq(polls.id, body.pollId)))[0];
      if (!poll) return json({ error: "Decision not found." }, 404);
      if (
        body.choice !== null &&
        (!Number.isInteger(body.choice) ||
          body.choice < 0 ||
          body.choice >= poll.options.length)
      )
        return json({ error: "Invalid option." }, 400);
      if (body.action === "resolve") {
        await db
          .insert(resolutions)
          .values({
            pollId: poll.id,
            selected: body.choice,
            updatedBy: user.displayName,
            updatedAt: now,
          })
          .onConflictDoUpdate({
            target: resolutions.pollId,
            set: {
              selected: body.choice,
              updatedBy: user.displayName,
              updatedAt: now,
            },
          });
      } else {
        const closed = (
          await db
            .select()
            .from(resolutions)
            .where(eq(resolutions.pollId, poll.id))
        )[0];
        if (closed?.selected != null)
          return json(
            { error: "Voting is closed. Reopen this decision first." },
            409,
          );
        if (body.choice === null)
          await db
            .delete(votes)
            .where(
              and(eq(votes.pollId, poll.id), eq(votes.userId, user.userId)),
            );
        else
          await db.run(
            sql`INSERT INTO votes (poll_id,user_id,name,choice,updated_at) SELECT ${poll.id},${user.userId},${user.displayName},${body.choice},${now} WHERE NOT EXISTS (SELECT 1 FROM resolutions WHERE poll_id=${poll.id} AND selected IS NOT NULL) ON CONFLICT(poll_id,user_id) DO UPDATE SET name=excluded.name, choice=excluded.choice, updated_at=excluded.updated_at`,
          );
      }
    } else if (body.action === "milestone") {
      if (
        !phases.some((p) => p.id === body.id) ||
        !["not_started", "in_progress", "done"].includes(body.status)
      )
        return json({ error: "Invalid milestone." }, 400);
      await db
        .insert(milestones)
        .values({
          id: body.id,
          status: body.status,
          updatedBy: user.displayName,
          updatedAt: now,
        })
        .onConflictDoUpdate({
          target: milestones.id,
          set: {
            status: body.status,
            updatedBy: user.displayName,
            updatedAt: now,
          },
        });
    } else if (body.action === "poll") {
      if (
        typeof body.title !== "string" ||
        !body.title.trim() ||
        body.title.length > 120 ||
        typeof body.description !== "string" ||
        body.description.length > 700 ||
        !Array.isArray(body.options) ||
        body.options.length < 2 ||
        body.options.length > 6 ||
        body.options.some(
          (s: unknown) => typeof s !== "string" || !s.trim() || s.length > 120,
        ) ||
        new Set(body.options.map((s: string) => s.trim().toLowerCase()))
          .size !== body.options.length
      )
        return json(
          { error: "Use a title and 2–6 distinct, nonempty options." },
          400,
        );
      await db.insert(polls).values({
        id: crypto.randomUUID(),
        title: body.title.trim(),
        description: body.description.trim(),
        category: "Team",
        options: body.options.map((s: string) => s.trim()),
        createdBy: user.userId,
      });
    } else if (body.action === "run") {
      if (
        typeof body.name !== "string" ||
        !body.name.trim() ||
        body.name.length > 120 ||
        !Number.isSafeInteger(body.tokens) ||
        body.tokens < 0 ||
        body.tokens > 1e12 ||
        (body.loss !== null &&
          (typeof body.loss !== "number" ||
            !Number.isFinite(body.loss) ||
            body.loss < 0 ||
            body.loss > 100)) ||
        typeof body.notes !== "string" ||
        body.notes.length > 1500
      )
        return json(
          {
            error:
              "Use a name, valid token count, and optional nonnegative loss.",
          },
          400,
        );
      await db.insert(runs).values({
        id: crypto.randomUUID(),
        name: body.name.trim(),
        tokens: body.tokens,
        loss: body.loss,
        notes: body.notes,
        createdBy: user.displayName,
        createdAt: now,
      });
    } else return json({ error: "Unknown action." }, 400);
    return json({ ok: true });
  } catch (error) {
    console.error("Team data write failed", error);
    return json({ error: "Could not save. Please retry." }, 503);
  }
}

CREATE TABLE desktop_sessions (token_hash text PRIMARY KEY NOT NULL, user_id text NOT NULL, expires_at integer NOT NULL);
--> statement-breakpoint
CREATE TABLE training_rooms (id text PRIMARY KEY NOT NULL, active_slot text UNIQUE, name text NOT NULL, fingerprint text NOT NULL, peer text NOT NULL, target_tokens integer NOT NULL, shards integer NOT NULL, phase text NOT NULL, expected text NOT NULL DEFAULT '[]', created_by text NOT NULL, created_at integer NOT NULL);
--> statement-breakpoint
CREATE TABLE training_members (room_id text NOT NULL, instance_id text NOT NULL, user_id text NOT NULL, name text NOT NULL, device text NOT NULL, fingerprint text NOT NULL, last_seen integer NOT NULL, ready integer NOT NULL DEFAULT 0, shard integer, leader integer NOT NULL DEFAULT 0, progress text, PRIMARY KEY(room_id,instance_id));

CREATE TABLE server_directory (id text PRIMARY KEY NOT NULL, name text NOT NULL, description text NOT NULL, origin text NOT NULL UNIQUE, is_public integer NOT NULL DEFAULT 0, updated_at integer NOT NULL);
--> statement-breakpoint
-- Owner explicitly approved publishing this name, description and public URL.

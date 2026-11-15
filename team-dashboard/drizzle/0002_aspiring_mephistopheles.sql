CREATE TABLE `admin_accounts` (
	`id` text PRIMARY KEY NOT NULL,
	`salt` text NOT NULL,
	`password_hash` text NOT NULL,
	`failures` integer NOT NULL,
	`locked_until` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `admin_sessions` (
	`token_hash` text PRIMARY KEY NOT NULL,
	`expires_at` integer NOT NULL
);

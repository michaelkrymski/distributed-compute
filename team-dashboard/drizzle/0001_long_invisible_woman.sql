CREATE TABLE `benchmark_agents` (
	`session_id` text NOT NULL,
	`user_id` text NOT NULL,
	`instance_id` text NOT NULL,
	`device` text NOT NULL,
	`last_seen_at` integer NOT NULL,
	`requested_ready` integer NOT NULL,
	`ready` integer NOT NULL,
	`result` text,
	PRIMARY KEY(`session_id`, `user_id`)
);
--> statement-breakpoint
CREATE TABLE `benchmark_sessions` (
	`id` text PRIMARY KEY NOT NULL,
	`active_slot` text,
	`created_at` integer NOT NULL,
	`created_by` text NOT NULL,
	`peers` text NOT NULL,
	`expected` text NOT NULL,
	`config` text NOT NULL,
	`coordinator_seen_at` integer NOT NULL,
	`starts_at` integer,
	`cancelled_at` integer
);
--> statement-breakpoint
CREATE UNIQUE INDEX `benchmark_sessions_active_slot_unique` ON `benchmark_sessions` (`active_slot`);

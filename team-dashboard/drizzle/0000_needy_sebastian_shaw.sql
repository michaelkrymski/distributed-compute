CREATE TABLE `milestones` (
	`id` text PRIMARY KEY NOT NULL,
	`status` text NOT NULL,
	`updated_by` text NOT NULL,
	`updated_at` text NOT NULL
);
--> statement-breakpoint
CREATE TABLE `polls` (
	`id` text PRIMARY KEY NOT NULL,
	`title` text NOT NULL,
	`description` text NOT NULL,
	`category` text NOT NULL,
	`options` text NOT NULL,
	`created_by` text NOT NULL
);
--> statement-breakpoint
CREATE TABLE `resolutions` (
	`poll_id` text PRIMARY KEY NOT NULL,
	`selected` integer,
	`updated_by` text NOT NULL,
	`updated_at` text NOT NULL
);
--> statement-breakpoint
CREATE TABLE `runs` (
	`id` text PRIMARY KEY NOT NULL,
	`name` text NOT NULL,
	`tokens` integer NOT NULL,
	`loss` real,
	`notes` text NOT NULL,
	`created_by` text NOT NULL,
	`created_at` text NOT NULL
);
--> statement-breakpoint
CREATE TABLE `votes` (
	`poll_id` text NOT NULL,
	`user_id` text NOT NULL,
	`name` text NOT NULL,
	`choice` integer NOT NULL,
	`updated_at` text NOT NULL,
	PRIMARY KEY(`poll_id`, `user_id`)
);

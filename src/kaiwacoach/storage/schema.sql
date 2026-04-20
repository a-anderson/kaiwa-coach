-- KaiwaCoach SQLite schema (v3)
-- Forward compatibility notes:
-- - Schema changes should be handled with migrations; avoid in-place edits without a migration plan.
-- - Prefer nullable columns or defaults for additive changes, then backfill if needed.
-- - During MVP, local DB reset on mismatch is acceptable for the sole tester.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  version INTEGER NOT NULL,
  applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO schema_version (id, version)
VALUES (1, 3);

CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  title TEXT,
  language TEXT NOT NULL,
  asr_model_id TEXT NOT NULL,
  llm_model_id TEXT NOT NULL,
  tts_model_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  model_metadata_json TEXT,
  conversation_type TEXT NOT NULL DEFAULT 'chat'
);

CREATE TABLE IF NOT EXISTS user_turns (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  input_text TEXT,
  asr_text TEXT,
  asr_meta_json TEXT,
  FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS assistant_turns (
  id TEXT PRIMARY KEY,
  user_turn_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  reply_text TEXT NOT NULL,
  llm_meta_json TEXT,
  FOREIGN KEY (user_turn_id) REFERENCES user_turns(id) ON DELETE CASCADE,
  FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS corrections (
  id TEXT PRIMARY KEY,
  user_turn_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  errors_json TEXT,
  corrected_text TEXT,
  native_text TEXT,
  explanation_text TEXT,
  prompt_hash TEXT,
  FOREIGN KEY (user_turn_id) REFERENCES user_turns(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  meta_json TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_turns_conversation_id
  ON user_turns(conversation_id);

CREATE INDEX IF NOT EXISTS idx_assistant_turns_user_turn_id
  ON assistant_turns(user_turn_id);

CREATE INDEX IF NOT EXISTS idx_assistant_turns_conversation_id
  ON assistant_turns(conversation_id);

CREATE INDEX IF NOT EXISTS idx_corrections_user_turn_id
  ON corrections(user_turn_id);

CREATE INDEX IF NOT EXISTS idx_artifacts_conversation_id
  ON artifacts(conversation_id);

-- Singleton user profile row (id must be 1).
-- language_proficiency_json stores a JSON object keyed by language code,
-- e.g. {"ja": "N4", "ja_kanji": "N2", "fr": "B1"}.
CREATE TABLE IF NOT EXISTS user_profile (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  user_name TEXT,
  language_proficiency_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO user_profile (id) VALUES (1);

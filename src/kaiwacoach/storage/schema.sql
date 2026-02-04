-- KaiwaCoach SQLite schema (v1)
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  version INTEGER NOT NULL,
  applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO schema_version (id, version)
VALUES (1, 1);

CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  title TEXT,
  language TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  model_metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS user_turns (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  input_text TEXT,
  input_audio_path TEXT,
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
  reply_audio_path TEXT,
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

CREATE TRIGGER IF NOT EXISTS trg_conversations_updated_at
AFTER UPDATE ON conversations
FOR EACH ROW
BEGIN
  UPDATE conversations
  SET updated_at = datetime('now')
  WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_user_turns_updated_at
AFTER UPDATE ON user_turns
FOR EACH ROW
BEGIN
  UPDATE user_turns
  SET updated_at = datetime('now')
  WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_assistant_turns_updated_at
AFTER UPDATE ON assistant_turns
FOR EACH ROW
BEGIN
  UPDATE assistant_turns
  SET updated_at = datetime('now')
  WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_corrections_updated_at
AFTER UPDATE ON corrections
FOR EACH ROW
BEGIN
  UPDATE corrections
  SET updated_at = datetime('now')
  WHERE id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_artifacts_updated_at
AFTER UPDATE ON artifacts
FOR EACH ROW
BEGIN
  UPDATE artifacts
  SET updated_at = datetime('now')
  WHERE id = OLD.id;
END;

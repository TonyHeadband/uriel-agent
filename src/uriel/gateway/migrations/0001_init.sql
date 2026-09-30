CREATE TABLE decisions (
  id               uuid PRIMARY KEY,
  created_at       timestamptz NOT NULL DEFAULT now(),
  thread_id        text NOT NULL,
  user_id          text NOT NULL,
  point            text NOT NULL,
  context          text NOT NULL,
  options          jsonb NOT NULL,
  choice           text NOT NULL,
  confidence       real,
  adapter          text NOT NULL,
  model            text,
  latency_ms       integer,
  corrected_choice text
);
CREATE INDEX decisions_created_at_idx ON decisions (created_at);

CREATE TABLE conversations (
  id         uuid PRIMARY KEY,
  user_id    text NOT NULL,
  title      text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX conversations_user_updated_idx ON conversations (user_id, updated_at DESC);

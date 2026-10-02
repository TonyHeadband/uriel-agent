-- The last Talk message the gateway handled (answered or deliberately ignored), per room, so a restart
-- neither replays nor skips.
CREATE TABLE talk_cursors (
    room_token      text PRIMARY KEY,
    last_message_id bigint NOT NULL,
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- uriel-tools keeps its document index in its own database. The extension needs a superuser, which
-- uriel-tools' role is not, so it is created here (the image runs this once, on an empty volume).
CREATE ROLE uriel_tools LOGIN PASSWORD 'uriel_tools';
CREATE DATABASE uriel_tools OWNER uriel_tools;
\connect uriel_tools
CREATE EXTENSION IF NOT EXISTS vector;

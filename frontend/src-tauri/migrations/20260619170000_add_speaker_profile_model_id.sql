-- Keep persistent voice profiles isolated by speaker embedding model.
-- Older profiles were produced before model selection and used the original WeSpeaker model.
ALTER TABLE speaker_profiles ADD COLUMN model_id TEXT NOT NULL DEFAULT 'wespeaker-campp';

CREATE INDEX IF NOT EXISTS idx_speaker_profiles_model_id ON speaker_profiles(model_id);

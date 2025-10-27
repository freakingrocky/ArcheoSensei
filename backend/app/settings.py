from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SUPABASE_URL: str = "https://dlvzvrjsfmizztpgjown.supabase.co"
    SUPABASE_SERVICE_ROLE_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRsdnp2cmpzZm1penp0cGdqb3duIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1ODY2Mzk0NiwiZXhwIjoyMDc0MjM5OTQ2fQ.Vr4g_JhEX8fwnHKG7Jl7fRDyiGGa4trYPeelcQZqvaw"
    DATABASE_URL: str = "postgresql://postgres:Rocky%40123@db.dlvzvrjsfmizztpgjown.supabase.co:5432/postgres"  # postgres://...sslmode=require
    GROQ_API_KEY: str = "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"
    BACKEND_PORT: int = 8000
    EMBEDDING_DIM: int = 384
    SIG_NAME: str = ""
    SIG_API_KEY: str = ""
    SIG_GPT5_BASE: str = ""   # e.g., https://api-iw.azure-api.net/sig-shared-jpeast
    SIG_API_VERSION: str = "2025-01-01-preview"
    SIG_GPT5_DEPLOYMENT: str = "gpt-5-mini"
    HUGGINGFACE_TOKEN: str = ""


settings = Settings()

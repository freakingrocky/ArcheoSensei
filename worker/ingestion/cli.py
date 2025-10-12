import os
import sys
import time

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # later add DATABASE_URL, SUPABASE_URL, etc.
    pass

def main():
    _ = Settings()
    print("Worker ready. (Add ingestion logic here.)")
    # placeholder loop for docker dev
    if os.environ.get("WORKER_MODE") == "loop":
        while True:
            print("tick")
            time.sleep(10)

if __name__ == "__main__":
    main()

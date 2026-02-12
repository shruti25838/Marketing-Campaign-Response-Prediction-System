from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    db_url: str = os.getenv("DATABASE_URL", "")
    default_table: str = os.getenv("CAMPAIGN_TABLE", "campaign_data")
    target_column: str = os.getenv("TARGET_COLUMN", "response")
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))


settings = Settings()

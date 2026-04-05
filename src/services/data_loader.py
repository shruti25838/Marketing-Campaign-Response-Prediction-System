from typing import Optional

import pandas as pd
from sqlalchemy import create_engine


def load_from_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def load_from_postgres(db_url: str, table: str, query: Optional[str] = None) -> pd.DataFrame:
    engine = create_engine(db_url)
    sql = query if query else f"SELECT * FROM {table}"
    with engine.connect() as connection:
        return pd.read_sql(sql, connection)

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.services.storage.settings import load_persistence_settings

DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "db" / "migrations" / "001_tip_postgres_foundation.sql"


def apply_schema(*, database_url: str, schema_path: Path) -> None:
    sql = schema_path.read_text(encoding="utf-8")
    with psycopg.connect(database_url, autocommit=False) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql)
        connection.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplica el schema base de TIP sobre PostgreSQL/Neon."
    )
    parser.add_argument(
        "--database-url",
        help="Connection string PostgreSQL. Si se omite, usa DATABASE_URL.",
    )
    parser.add_argument(
        "--schema",
        default=str(DEFAULT_SCHEMA_PATH),
        help="Ruta al archivo SQL que se debe aplicar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_persistence_settings()
    database_url = str(args.database_url or settings.database_url).strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL es obligatorio para aplicar el schema PostgreSQL.")

    schema_path = Path(args.schema).resolve()
    if not schema_path.exists():
        raise FileNotFoundError(f"No se encontró el schema SQL: {schema_path}")

    apply_schema(database_url=database_url, schema_path=schema_path)
    print(f"Schema aplicado correctamente desde {schema_path}.")


if __name__ == "__main__":
    main()

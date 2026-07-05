from __future__ import annotations

from dataclasses import dataclass

import mysql.connector

from app.config import config


@dataclass(frozen=True)
class Tweet:
    id: int
    text: str
    positive: int
    negative: int


def get_connection():
    return mysql.connector.connect(
        host=config.mysql_host,
        port=config.mysql_port,
        user=config.mysql_user,
        password=config.mysql_password,
        database=config.mysql_database,
        connection_timeout=5,
    )


def fetch_annotated_tweets() -> list[Tweet]:
    query = """
        SELECT id, text, positive, negative
        FROM tweets
        WHERE text IS NOT NULL
          AND positive IN (0, 1)
          AND negative IN (0, 1)
    """
    with get_connection() as connection:
        with connection.cursor(dictionary=True) as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

    return [
        Tweet(
            id=int(row["id"]),
            text=str(row["text"]),
            positive=int(row["positive"]),
            negative=int(row["negative"]),
        )
        for row in rows
    ]

import json
import os
from typing import Any


FOLDER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "external",
    "tau_bench_airline",
    "data",
)


def load_data() -> dict[str, Any]:
    with open(os.path.join(FOLDER_PATH, "flights.json")) as f:
        flight_data = json.load(f)
    with open(os.path.join(FOLDER_PATH, "reservations.json")) as f:
        reservation_data = json.load(f)
    with open(os.path.join(FOLDER_PATH, "users.json")) as f:
        user_data = json.load(f)
    return {
        "flights": flight_data,
        "reservations": reservation_data,
        "users": user_data,
    }

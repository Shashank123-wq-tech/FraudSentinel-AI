import json
from datetime import datetime, timezone

from src.mlops.config import (
    PROJECT_NAME,
    PIPELINE_VERSION,
    COMPONENT_VERSIONS,
    RANDOM_SEED,
    REGISTRY_DIR,
)


VERSION_FILE = REGISTRY_DIR / "versions.json"


def build_version_record():

    return {
        "project": PROJECT_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "random_seed": RANDOM_SEED,
        "components": COMPONENT_VERSIONS,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def save_versions():

    record = build_version_record()

    with open(
        VERSION_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            record,
            file,
            indent=2,
        )

    return record


def load_versions():

    if not VERSION_FILE.exists():
        return None

    with open(
        VERSION_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


if __name__ == "__main__":

    record = save_versions()

    print(
        json.dumps(
            record,
            indent=2,
        )
    )
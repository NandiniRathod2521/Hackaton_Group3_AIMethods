import os
from functools import lru_cache
from huggingface_hub import snapshot_download


@lru_cache(maxsize=1)
def get_model_directory():
    repo_id = os.getenv("HF_MODEL_REPO")

    if not repo_id:
        raise RuntimeError(
            "HF_MODEL_REPO environment variable is missing"
        )

    token = os.getenv("HF_TOKEN") or None

    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        local_files_only=False
    )


if __name__ == "__main__":
    print("Model files downloaded to:")
    print(get_model_directory())
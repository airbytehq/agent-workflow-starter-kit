import os
import time

from sentence_transformers import SentenceTransformer


def main() -> None:
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    download_dir = os.path.normpath(os.path.join(current_dir, "..", ".models"))
    download_path = os.path.join(download_dir, model_name)
    os.makedirs(download_dir, exist_ok=True)

    if os.path.exists(download_path):
        size = dir_size(download_path)
        print(
            f"🤖✅ Embedding model `{model_name}` already downloaded to `{download_path}` ({size / 1024 / 1024:.2f} MB)"
        )
        return

    print(f"🤖 Downloading embedding model `{model_name}` to `{download_path}`")

    model = SentenceTransformer(model_name)

    start_time = time.time()
    model.save(download_path)

    size = dir_size(download_path)
    print(
        f"🤖✅ Downloaded embedding model `{model_name}` to `{download_path}` in {time.time() - start_time:.2f} seconds ({size / 1024 / 1024:.2f} MB)"
    )


def dir_size(path: str) -> int:
    return sum(
        os.path.getsize(os.path.join(root, file))
        for root, _, files in os.walk(path)
        for file in files
    )


if __name__ == "__main__":
    main()

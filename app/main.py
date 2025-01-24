import pathlib

from chainlit.utils import mount_chainlit
from fastapi import FastAPI

application = FastAPI()


@application.get("/")
def read_main() -> dict[str, str]:
    return {"message": "Hello World from main app. Try /chat."}


chainlit_app_path = pathlib.Path(__file__).parent / "chat.py"
mount_chainlit(app=application, target=str(chainlit_app_path), path="/chat")

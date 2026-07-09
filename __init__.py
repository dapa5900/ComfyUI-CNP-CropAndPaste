from .nodes.crop_and_paste_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

from aiohttp import web
from server import PromptServer
from pathlib import Path

if hasattr(PromptServer, "instance"):
    try:
        PromptServer.instance.app.add_routes(
            [
                web.static(
                    "/cpweb", (Path(__file__).parent.absolute() / "web").as_posix()
                )
            ]
        )
    except Exception as e:
        import logging

        logging.warning(f"Crop And Paste Nodes: Could not initialize web routes: {e}")

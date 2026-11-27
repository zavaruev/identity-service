"""HTTP clients for the two existing identity engines.

This module never writes a model file: enrollment goes through each engine's
own API, which is the only way the running face server sees a new mtime.
"""
import os
from pathlib import Path

import aiohttp


class EngineError(Exception):
    def __init__(self, status: int, detail: str = ""):
        self.status = int(status)
        self.detail = str(detail)[:300]
        super().__init__(f"engine HTTP {self.status}: {self.detail}")


class Engines:
    def __init__(self, voice_url: str, face_url: str, api_key: str,
                 timeout: float = 20.0, session=None, voice_gallery_dir: str = ""):
        self.voice_url = str(voice_url).rstrip("/")
        self.face_url = str(face_url).rstrip("/")
        self.api_key = str(api_key or "")
        self.timeout = float(timeout)
        self.http = session
        self.voice_gallery_dir = str(voice_gallery_dir or "")

    def _form(self) -> aiohttp.FormData:
        return aiohttp.FormData()

    async def _json(self, method: str, url: str, form=None, headers=None) -> dict:
        owner = self.http is None
        sess = self.http or aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        try:
            kwargs = {"timeout": self.timeout}
            if form is not None:
                kwargs["data"] = form
            if headers is not None:
                kwargs["headers"] = headers
            call = {
                "POST": lambda: sess.post(url, **kwargs),
                "GET": lambda: sess.get(url, **kwargs),
                "DELETE": lambda: sess.delete(url, **kwargs),
            }[method]()
            async with call as resp:
                if resp.status != 200:
                    try:
                        detail = await resp.json()
                    except Exception:
                        detail = ""
                    raise EngineError(resp.status, detail)
                return await resp.json()
        finally:
            if owner:
                await sess.close()

    async def identify(self, audio: bytes) -> tuple:
        form = self._form()
        form.add_field("file", audio, filename="audio.ogg", content_type="audio/ogg")
        data = await self._json("POST", f"{self.voice_url}/identify", form)
        return str(data.get("user_id", "unknown")), float(data.get("confidence", 0.0))

    async def recognize(self, image: bytes, threshold: float) -> list:
        form = self._form()
        form.add_field("file", image, filename="frame.jpg", content_type="image/jpeg")
        form.add_field("threshold", str(threshold))
        data = await self._json("POST", f"{self.face_url}/recognize", form)
        return data.get("faces", []) or []

    async def enroll_voice(self, key: str, files: list) -> dict:
        form = self._form()
        form.add_field("user_id", str(key))
        for name, content in files:
            form.add_field("files", content, filename=name)
        return await self._json("POST", f"{self.voice_url}/enroll", form,
                                {"X-API-Key": self.api_key})

    async def enroll_face(self, key: str, files: list, replace: bool = False) -> dict:
        form = self._form()
        form.add_field("name", str(key))
        form.add_field("replace", "true" if replace else "false")
        for name, content in files:
            form.add_field("files", content, filename=name)
        return await self._json("POST", f"{self.face_url}/enroll", form)

    async def delete_face(self, key: str) -> dict:
        return await self._json("DELETE", f"{self.face_url}/enroll/{key}")

    async def rosters(self) -> tuple:
        voice = []
        if self.voice_gallery_dir and Path(self.voice_gallery_dir).is_dir():
            voice = sorted(p.stem for p in Path(self.voice_gallery_dir).glob("*.npy"))
        data = await self._json("GET", f"{self.face_url}/health")
        return voice, sorted(data.get("known_faces", []) or [])

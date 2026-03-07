import base64
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests


@dataclass(frozen=True)
class GitHubConfig:
    owner: str
    repo: str
    token: str
    branch: str = "main"


class GitHubRepoStore:
    def __init__(self, cfg: GitHubConfig):
        self.cfg = cfg
        self.api = "https://api.github.com"
        self.headers = {
            "Authorization": f"Bearer {cfg.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _url(self, path: str) -> str:
        return f"{self.api}{path}"

    def _repo_path(self) -> str:
        return f"/repos/{self.cfg.owner}/{self.cfg.repo}"

    def get_ref_sha(self, branch: Optional[str] = None) -> str:
        br = branch or self.cfg.branch
        r = requests.get(
            self._url(self._repo_path() + f"/git/ref/heads/{br}"),
            headers=self.headers,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["object"]["sha"]

    def get_tree(self, branch: Optional[str] = None) -> List[dict]:
        sha = self.get_ref_sha(branch)
        r = requests.get(
            self._url(self._repo_path() + f"/git/trees/{sha}?recursive=1"),
            headers=self.headers,
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("tree", [])

    def list_files(
        self,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> List[str]:
        tree = self.get_tree(branch)
        out = []
        for item in tree:
            if item.get("type") != "blob":
                continue
            path = item.get("path", "")
            if prefix and not path.startswith(prefix):
                continue
            if suffix and not path.endswith(suffix):
                continue
            out.append(path)
        return sorted(out)

    def get_file_metadata(self, path: str, ref: Optional[str] = None) -> dict:
        params = {}
        if ref:
            params["ref"] = ref

        r = requests.get(
            self._url(self._repo_path() + f"/contents/{path}"),
            headers=self.headers,
            params=params,
            timeout=60,
        )
        r.raise_for_status()
        j = r.json()
        if j.get("type") != "file":
            raise ValueError(f"Not a file: {path}")
        return j

    def get_file_sha(self, path: str, ref: Optional[str] = None) -> str:
        return self.get_file_metadata(path, ref=ref)["sha"]

    def exists(self, path: str, ref: Optional[str] = None) -> bool:
        try:
            self.get_file_metadata(path, ref=ref)
            return True
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return False
            raise

    def get_raw_content(self, path: str, ref: Optional[str] = None) -> bytes:
        meta = self.get_file_metadata(path, ref=ref)
        download_url = meta.get("download_url")
        if not download_url:
            raise ValueError(f"No download_url for file: {path}")
        r = requests.get(download_url, timeout=300)
        r.raise_for_status()
        return r.content

    def put_content(
        self,
        path: str,
        content_bytes: bytes,
        message: str,
        branch: Optional[str] = None,
        sha: Optional[str] = None,
    ):
        payload = {
            "message": message,
            "content": base64.b64encode(content_bytes).decode("utf-8"),
            "branch": branch or self.cfg.branch,
        }
        if sha:
            payload["sha"] = sha

        r = requests.put(
            self._url(self._repo_path() + f"/contents/{path}"),
            headers=self.headers,
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    def write_json(self, path: str, obj: dict, message: str, branch: Optional[str] = None):
        branch = branch or self.cfg.branch
        sha = self.get_file_sha(path, ref=branch) if self.exists(path, ref=branch) else None
        payload = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        return self.put_content(path=path, content_bytes=payload, message=message, branch=branch, sha=sha)

    def read_json(self, path: str, ref: Optional[str] = None) -> dict:
        raw = self.get_raw_content(path, ref=ref)
        return json.loads(raw.decode("utf-8"))

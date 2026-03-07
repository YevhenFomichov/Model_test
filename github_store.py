import base64
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests


@dataclass(frozen=True)
class GitHubConfig:
    owner: str
    repo: str
    token: str
    branch: str = "main"


class GitHubRepoStore:
    """
    GitHub API wrapper for:
    - reading file content
    - creating/updating files
    - deleting files
    - moving file by copy+delete
    - listing files from repo tree
    """

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

    def list_files(self, prefix: Optional[str] = None, suffix: Optional[str] = None, branch: Optional[str] = None) -> List[str]:
        """
        List files from repository tree recursively.
        Returns repo-relative paths like:
          selected_models/basic_raw_None_mse.keras
        """
        sha = self.get_ref_sha(branch)
        r = requests.get(
            self._url(self._repo_path() + f"/git/trees/{sha}?recursive=1"),
            headers=self.headers,
            timeout=60,
        )
        r.raise_for_status()
        tree = r.json().get("tree", [])

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

    def get_content(self, path: str, ref: Optional[str] = None) -> Tuple[bytes, str]:
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

        content_b64 = j["content"]
        sha = j["sha"]
        raw = base64.b64decode(content_b64)
        return raw, sha

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
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def delete_content(
        self,
        path: str,
        sha: str,
        message: str,
        branch: Optional[str] = None,
    ):
        payload = {
            "message": message,
            "sha": sha,
            "branch": branch or self.cfg.branch,
        }
        r = requests.delete(
            self._url(self._repo_path() + f"/contents/{path}"),
            headers=self.headers,
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def exists(self, path: str, ref: Optional[str] = None) -> bool:
        try:
            self.get_content(path, ref=ref)
            return True
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return False
            raise

    def move_file(
        self,
        src_path: str,
        dst_path: str,
        message_prefix: str,
        branch: Optional[str] = None,
        overwrite: bool = False,
    ):
        branch = branch or self.cfg.branch

        src_bytes, src_sha = self.get_content(src_path, ref=branch)

        dst_sha = None
        if self.exists(dst_path, ref=branch):
            if not overwrite:
                raise FileExistsError(f"Destination exists: {dst_path}")
            _, dst_sha = self.get_content(dst_path, ref=branch)

        self.put_content(
            dst_path,
            src_bytes,
            message=f"{message_prefix}: add {dst_path}",
            branch=branch,
            sha=dst_sha,
        )

        self.delete_content(
            src_path,
            sha=src_sha,
            message=f"{message_prefix}: delete {src_path}",
            branch=branch,
        )

import base64
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
    """
    GitHub API wrapper for:
    - listing files from repo tree
    - downloading exact blob bytes through Git Blobs API
    - checking file existence
    - moving file by copy+delete (direct commit to main)
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

    def get_tree(self, branch: Optional[str] = None) -> List[dict]:
        sha = self.get_ref_sha(branch)
        r = requests.get(
            self._url(self._repo_path() + f"/git/trees/{sha}?recursive=1"),
            headers=self.headers,
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("tree", [])

    def list_files(self, prefix: Optional[str] = None, suffix: Optional[str] = None, branch: Optional[str] = None) -> List[str]:
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

    def path_to_blob_sha_map(self, branch: Optional[str] = None) -> Dict[str, str]:
        tree = self.get_tree(branch)
        out: Dict[str, str] = {}
        for item in tree:
            if item.get("type") != "blob":
                continue
            path = item.get("path", "")
            sha = item.get("sha", "")
            if path and sha:
                out[path] = sha
        return out

    def get_blob_sha(self, path: str, branch: Optional[str] = None) -> str:
        mapping = self.path_to_blob_sha_map(branch)
        if path not in mapping:
            raise FileNotFoundError(f"Path not found in repo tree: {path}")
        return mapping[path]

    def get_blob_bytes(self, blob_sha: str) -> bytes:
        """
        Downloads exact blob content using Git Blobs API.
        This is reliable for binary .keras files.
        """
        r = requests.get(
            self._url(self._repo_path() + f"/git/blobs/{blob_sha}"),
            headers=self.headers,
            timeout=120,
        )
        r.raise_for_status()
        j = r.json()

        if j.get("encoding") != "base64":
            raise ValueError(f"Unexpected blob encoding for sha={blob_sha}: {j.get('encoding')}")

        content_b64 = j.get("content", "").replace("\n", "")
        return base64.b64decode(content_b64)

    def get_raw_content(self, path: str, ref: Optional[str] = None) -> bytes:
        """
        Resolve path -> blob sha -> exact bytes.
        """
        blob_sha = self.get_blob_sha(path, branch=ref)
        return self.get_blob_bytes(blob_sha)

    def get_file_sha(self, path: str, ref: Optional[str] = None) -> str:
        """
        This is the contents/file SHA used by GitHub Contents API for delete/update.
        """
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
        return j["sha"]

    def exists(self, path: str, ref: Optional[str] = None) -> bool:
        try:
            self.get_blob_sha(path, branch=ref)
            return True
        except FileNotFoundError:
            return False

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
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    def move_file(
        self,
        src_path: str,
        dst_path: str,
        message_prefix: str,
        branch: Optional[str] = None,
        overwrite: bool = False,
    ):
        branch = branch or self.cfg.branch

        src_bytes = self.get_raw_content(src_path, ref=branch)
        src_sha = self.get_file_sha(src_path, ref=branch)

        dst_sha = None
        if self.exists(dst_path, ref=branch):
            if not overwrite:
                raise FileExistsError(f"Destination exists: {dst_path}")
            dst_sha = self.get_file_sha(dst_path, ref=branch)

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

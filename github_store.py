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
    - downloading file bytes via Contents API metadata/download_url
    - falling back to git blob download when appropriate
    - moving file by copy+delete (direct commit to main)
    """

    LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

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

    def is_lfs_pointer_bytes(self, content: bytes) -> bool:
        return content.startswith(self.LFS_POINTER_PREFIX)

    def looks_like_html_or_json_error(self, content: bytes) -> bool:
        head = content[:200].lstrip()
        return (
            head.startswith(b"<!DOCTYPE html")
            or head.startswith(b"<html")
            or head.startswith(b"{")
            or head.startswith(b"[")
        )

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

    def _download_via_download_url(self, download_url: str) -> bytes:
        # download_url is intended for direct file download
        r = requests.get(download_url, timeout=300)
        r.raise_for_status()
        return r.content

    def get_raw_content(self, path: str, ref: Optional[str] = None) -> bytes:
        """
        Preferred order:
        1) Contents API metadata -> download_url -> bytes
        2) If that yields invalid/pointer-ish content, fallback to git blob bytes
        """
        meta = self.get_file_metadata(path, ref=ref)
        download_url = meta.get("download_url")

        if download_url:
            data = self._download_via_download_url(download_url)
            if data and (not self.is_lfs_pointer_bytes(data)) and (not self.looks_like_html_or_json_error(data)):
                return data

        # Fallback: direct git blob
        blob_sha = self.get_blob_sha(path, branch=ref)
        blob_bytes = self.get_blob_bytes(blob_sha)

        # If blob is LFS pointer, then this file is stored via LFS and download_url did not resolve actual bytes
        if self.is_lfs_pointer_bytes(blob_bytes):
            raise ValueError(
                f"GitHub returned a Git LFS pointer instead of the actual file for: {path}. "
                f"This usually means the file is stored in LFS and direct API download did not resolve the object."
            )

        return blob_bytes

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

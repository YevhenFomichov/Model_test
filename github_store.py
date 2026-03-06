import base64
from dataclasses import dataclass
from typing import Optional, Tuple

import requests


@dataclass(frozen=True)
class GitHubConfig:
    owner: str
    repo: str
    token: str
    branch: str = "main"


class GitHubRepoStore:
    """
    Minimal GitHub API wrapper for:
    - reading file content
    - creating/updating files
    - deleting files
    - creating branch
    - creating pull request
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

    def get_ref_sha(self, branch: str) -> str:
        r = requests.get(
            self._url(self._repo_path() + f"/git/ref/heads/{branch}"),
            headers=self.headers,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["object"]["sha"]

    def create_branch_from(self, base_branch: str, new_branch: str) -> str:
        base_sha = self.get_ref_sha(base_branch)
        payload = {"ref": f"refs/heads/{new_branch}", "sha": base_sha}
        r = requests.post(
            self._url(self._repo_path() + "/git/refs"),
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        if r.status_code == 422:
            # branch already exists
            return new_branch
        r.raise_for_status()
        return new_branch

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
        branch: str,
        sha: Optional[str] = None,
    ):
        payload = {
            "message": message,
            "content": base64.b64encode(content_bytes).decode("utf-8"),
            "branch": branch,
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

    def delete_content(self, path: str, sha: str, message: str, branch: str):
        payload = {
            "message": message,
            "sha": sha,
            "branch": branch,
        }
        r = requests.delete(
            self._url(self._repo_path() + f"/contents/{path}"),
            headers=self.headers,
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    def create_pull_request(
        self,
        head_branch: str,
        title: str,
        body: str,
        base_branch: Optional[str] = None,
    ) -> str:
        base = base_branch or self.cfg.branch
        payload = {
            "title": title,
            "head": head_branch,
            "base": base,
            "body": body,
        }
        r = requests.post(
            self._url(self._repo_path() + "/pulls"),
            headers=self.headers,
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["html_url"]

    def move_file_via_branch(
        self,
        src_path: str,
        dst_path: str,
        branch: str,
        message_prefix: str,
        overwrite: bool = False,
    ):
        src_bytes, src_sha = self.get_content(src_path, ref=branch)

        dst_sha = None
        try:
            _, dst_sha = self.get_content(dst_path, ref=branch)
            if not overwrite:
                raise FileExistsError(f"Destination exists: {dst_path}")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                dst_sha = None
            else:
                raise

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

"""
utils/git_push.py
Git integration — push project to GitHub automatically.
Reads all credentials from .env — no terminal commands needed.
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime


def _run(cmd: list, cwd: str) -> tuple:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def _read_env(cwd: str) -> dict:
    """Read .env file and return dict of key=value pairs."""
    env_vars = {}
    env_path = Path(cwd) / ".env"
    if not env_path.exists():
        return env_vars
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env_vars[k.strip()] = v.strip()
    return env_vars


def setup_git(repo_url="", username="", email="", token="",
              branch="main", project_dir=None) -> dict:
    """Setup git — reads from .env if params are empty."""
    cwd  = project_dir or str(Path(__file__).parent.parent)
    logs = []
    env  = _read_env(cwd)

    repo_url = repo_url or env.get("GIT_REPO_URL", "")
    username = username or env.get("GIT_USERNAME", "")
    email    = email    or env.get("GIT_EMAIL",    "")
    token    = token    or env.get("GIT_TOKEN",    "")
    branch   = branch   or env.get("GIT_BRANCH",  "main")

    # git init
    _run(["git", "init"], cwd)
    logs.append("git init done")

    # identity
    if username: _run(["git", "config", "user.name",  username], cwd)
    if email:    _run(["git", "config", "user.email", email],    cwd)
    logs.append(f"Identity: {username} <{email}>")

    # remote
    if repo_url and token:
        auth_url = repo_url.replace("https://github.com",
                                    f"https://{token}@github.com")
        _run(["git", "remote", "remove", "origin"], cwd)
        _run(["git", "remote", "add", "origin", auth_url], cwd)
        logs.append(f"Remote set: {repo_url}")

    return {"ok": True, "logs": logs}


def push_to_git(commit_message=None, branch="main",
                project_dir=None, include_reports=False) -> dict:
    """
    Auto-read .env → stage → commit → push.
    No terminal commands needed.
    """
    cwd  = project_dir or str(Path(__file__).parent.parent)
    logs = []
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg  = commit_message or f"DQ Platform update — {ts}"

    # Read credentials from .env
    env      = _read_env(cwd)
    repo_url = env.get("GIT_REPO_URL", "").strip()
    username = env.get("GIT_USERNAME", "").strip()
    email    = env.get("GIT_EMAIL",    "").strip()
    token    = env.get("GIT_TOKEN",    "").strip()
    branch   = env.get("GIT_BRANCH",  branch).strip() or branch

    if not repo_url:
        return {"ok": False, "error": "GIT_REPO_URL not set in .env"}
    if not token:
        return {"ok": False, "error": "GIT_TOKEN not set in .env"}

    # Init if needed
    if not (Path(cwd) / ".git").exists():
        _run(["git", "init"], cwd)
        logs.append("git init done")

    # Set identity
    if username: _run(["git", "config", "user.name",  username], cwd)
    if email:    _run(["git", "config", "user.email", email],    cwd)

    # Set remote with token
    auth_url = repo_url.replace("https://github.com",
                                f"https://{token}@github.com") \
               if "github.com" in repo_url else repo_url
    _run(["git", "remote", "remove", "origin"], cwd)
    _run(["git", "remote", "add", "origin", auth_url], cwd)
    logs.append(f"Remote: {repo_url}")

    # Ensure .gitignore
    gi = Path(cwd) / ".gitignore"
    if not gi.exists():
        gi.write_text(".env\n*.db\ndata/uploads/\n__pycache__/\n*.pyc\n.venv/\nvenv/\n")
        logs.append("Created .gitignore")

    # Stage files
    paths = ["agents/","utils/","ui/","data/sample/",
             "requirements.txt","run_ui.py","CLAUDE.md",".gitignore"]
    if include_reports:
        paths.append("reports/")
    for p in paths:
        out, err, rc = _run(["git", "add", p], cwd)
        if err and "pathspec" not in err and "did not match" not in err:
            logs.append(f"add {p}: {err}")

    # Check anything to commit
    out, _, _ = _run(["git", "status", "--porcelain"], cwd)
    if not out.strip():
        return {"ok": True, "logs": logs,
                "message": "Nothing to commit — already up to date"}

    # Commit
    out, err, rc = _run(["git", "commit", "-m", msg], cwd)
    logs.append(f"Commit: {out or err}")
    if rc != 0 and "nothing to commit" not in (out + err):
        return {"ok": False, "logs": logs, "error": err}

    hash_out, _, _ = _run(["git", "rev-parse", "--short", "HEAD"], cwd)

    # Push
    out, err, rc = _run(["git", "push", "-u", "origin", branch, "--force"], cwd)
    logs.append(f"Push: {out or err}")

    if rc != 0:
        return {"ok": False, "logs": logs, "error": err,
                "hint": "Check GIT_TOKEN has 'repo' scope and repo exists on GitHub"}

    return {"ok": True, "logs": logs, "commit_hash": hash_out,
            "branch": branch,
            "message": f"Pushed to {branch} — commit {hash_out}"}


def get_git_status(project_dir=None) -> dict:
    """Return current git status."""
    cwd = project_dir or str(Path(__file__).parent.parent)
    if not (Path(cwd) / ".git").exists():
        return {"initialized": False, "message": "Not a git repository"}

    out,   _, _ = _run(["git", "status", "--short"],         cwd)
    branch,_, _ = _run(["git", "branch", "--show-current"],  cwd)
    log,   _, _ = _run(["git", "log", "--oneline", "-5"],    cwd)
    remote,_, _ = _run(["git", "remote", "get-url", "origin"], cwd)

    safe = ("https://github.com" + remote.split("@github.com")[1]
            if "@github.com" in remote else remote)

    return {
        "initialized":    True,
        "branch":         branch,
        "remote":         safe,
        "status":         out,
        "recent_commits": [l for l in log.split("\n") if l],
        "has_changes":    bool(out.strip()),
    }

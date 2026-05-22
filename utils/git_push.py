"""
utils/git_push.py
Git integration — push project to GitHub.
Called from the UI via /api/git/push endpoint.
"""

import os
import subprocess
import json
from pathlib import Path
from datetime import datetime


def _run(cmd: list, cwd: str) -> tuple:
    """Run a shell command, return (stdout, stderr, returncode)"""
    result = subprocess.run(
        cmd, cwd=cwd,
        capture_output=True, text=True
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def setup_git(
    repo_url:   str,
    username:   str,
    email:      str,
    token:      str,
    branch:     str = "main",
    project_dir:str = None,
) -> dict:
    """
    One-time git setup — init repo, add remote, set identity.
    Call this before the first push.
    """
    cwd = project_dir or str(Path(__file__).parent.parent)
    logs = []

    # git init
    out, err, rc = _run(["git", "init"], cwd)
    logs.append(f"git init: {out or err}")

    # Set user identity
    _run(["git", "config", "user.email", email], cwd)
    _run(["git", "config", "user.name",  username], cwd)
    logs.append(f"Identity set: {username} <{email}>")

    # Set remote with token embedded in URL for auth
    # Format: https://TOKEN@github.com/username/repo.git
    if "github.com" in repo_url:
        auth_url = repo_url.replace(
            "https://github.com",
            f"https://{token}@github.com"
        )
    else:
        auth_url = repo_url

    # Remove existing remote if any
    _run(["git", "remote", "remove", "origin"], cwd)
    out, err, rc = _run(["git", "remote", "add", "origin", auth_url], cwd)
    logs.append(f"Remote set: {repo_url}")  # show URL without token

    return {"ok": True, "logs": logs, "cwd": cwd}


def push_to_git(
    commit_message: str = None,
    branch:         str = "main",
    project_dir:    str = None,
    include_reports:bool = False,
) -> dict:
    """
    Stage, commit and push to GitHub.
    Returns dict with ok, logs, commit_hash.
    """
    cwd  = project_dir or str(Path(__file__).parent.parent)
    logs = []
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
    msg  = commit_message or f"DQ Platform update — {ts}"

    # Ensure .gitignore exists with sensible defaults
    gi_path = Path(cwd) / ".gitignore"
    if not gi_path.exists():
        gi_path.write_text(
            ".env\n*.db\ndata/uploads/\n__pycache__/\n*.pyc\n.venv/\nvenv/\n"
        )
        logs.append("Created .gitignore")

    # git add all (respects .gitignore)
    patterns = ["."]
    if not include_reports:
        # Exclude large report CSVs but include code
        patterns = [
            "agents/", "utils/", "ui/", "core/",
            "data/sample/",  # include sample data
            "requirements.txt", "run_ui.py", "CLAUDE.md",
            ".gitignore",
        ]

    for pattern in patterns:
        out, err, rc = _run(["git", "add", pattern], cwd)
        if err and "did not match" not in err:
            logs.append(f"git add {pattern}: {err}")

    # Check if there's anything to commit
    out, err, rc = _run(["git", "status", "--porcelain"], cwd)
    if not out.strip():
        return {"ok": True, "logs": logs, "message": "Nothing to commit — already up to date"}

    # Commit
    out, err, rc = _run(["git", "commit", "-m", msg], cwd)
    logs.append(f"Committed: {out or err}")
    if rc != 0:
        return {"ok": False, "logs": logs, "error": err}

    # Get commit hash
    hash_out, _, _ = _run(["git", "rev-parse", "--short", "HEAD"], cwd)

    # Push
    out, err, rc = _run(["git", "push", "-u", "origin", branch, "--force"], cwd)
    logs.append(f"Push: {out or err}")

    if rc != 0:
        return {
            "ok":    False,
            "logs":  logs,
            "error": err,
            "hint":  "Check your token has 'repo' scope and the repo exists on GitHub"
        }

    return {
        "ok":          True,
        "logs":        logs,
        "commit_hash": hash_out,
        "branch":      branch,
        "message":     f"Pushed to {branch} — commit {hash_out}",
    }


def get_git_status(project_dir: str = None) -> dict:
    """Return current git status"""
    cwd = project_dir or str(Path(__file__).parent.parent)

    # Check if git repo exists
    git_dir = Path(cwd) / ".git"
    if not git_dir.exists():
        return {"initialized": False, "message": "Not a git repository"}

    out, _, _ = _run(["git", "status", "--short"], cwd)
    branch_out, _, _ = _run(["git", "branch", "--show-current"], cwd)
    log_out, _, _ = _run(
        ["git", "log", "--oneline", "-5"], cwd)
    remote_out, _, _ = _run(["git", "remote", "get-url", "origin"], cwd)

    # Hide token from remote URL display
    safe_remote = remote_out
    if "@github.com" in safe_remote:
        safe_remote = "https://github.com" + safe_remote.split("@github.com")[1]

    return {
        "initialized": True,
        "branch":      branch_out,
        "remote":      safe_remote,
        "status":      out,
        "recent_commits": [l for l in log_out.split("\n") if l],
        "has_changes": bool(out.strip()),
    }

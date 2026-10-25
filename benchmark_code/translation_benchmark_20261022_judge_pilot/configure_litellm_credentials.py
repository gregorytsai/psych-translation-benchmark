#!/usr/bin/env python3
"""Store and use a LiteLLM token without putting the token on a command line.

The default action prompts on the controlling terminal with :mod:`getpass` and
writes one token to ``workspace.secrets/litellm_judge.key`` with mode 0600.
``--status`` emits only non-secret JSON status.  ``--exec-env`` loads the token
into a child process environment and then execs the supplied command without
using a shell, so the token does not appear in argv or in this helper's output.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECRET_PATH = ROOT / "workspace.secrets" / "litellm_judge.key"
ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
TOKEN_PREFIX = "LiteLLM token"


class CredentialError(ValueError):
    """A safe, user-actionable credential-file error."""


def _secret_path(value: str | None) -> Path:
    path = Path(value).expanduser() if value else DEFAULT_SECRET_PATH
    return path.absolute()


def _validate_token(token: str) -> str:
    if not token or not token.strip():
        raise CredentialError("token is empty")
    if token != token.strip():
        raise CredentialError("token must not have leading or trailing whitespace")
    if any(char in token for char in ("\r", "\n", "\x00")):
        raise CredentialError("token must be one line without a newline or NUL")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in token):
        raise CredentialError("token contains a control character")
    return token


def _ensure_secret_dir(path: Path) -> None:
    directory = path.parent
    old_umask = os.umask(0o077)
    try:
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    finally:
        os.umask(old_umask)
    mode = stat.S_IMODE(directory.stat().st_mode)
    if mode & 0o077:
        raise CredentialError(f"secret directory is not private (mode {mode:04o}): {directory}")


def _existing_regular_file(path: Path) -> os.stat_result | None:
    """Return lstat data, rejecting symlinks and non-regular files."""

    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode):
        raise CredentialError(f"refusing symlink secret path: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise CredentialError(f"secret path is not a regular file: {path}")
    return info


def _write_token(path: Path, token: str, *, replace: bool) -> None:
    _ensure_secret_dir(path)
    existing = _existing_regular_file(path)
    if existing is not None and not replace:
        raise CredentialError(f"secret already exists; pass --force only to replace it: {path}")

    old_umask = os.umask(0o077)
    temporary_path: Path | None = None
    fd: int | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(prefix=".litellm_judge.key.", dir=str(path.parent))
        temporary_path = Path(temporary_name)
        os.fchmod(fd, 0o600)
        payload = token.encode("utf-8")
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(temporary_path, path)
        temporary_path = None
        os.chmod(path, 0o600)
    finally:
        os.umask(old_umask)
        if fd is not None:
            os.close(fd)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _read_token(path: Path) -> str:
    """Read a private regular file, without exposing its contents in errors."""

    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise CredentialError("secret file is missing; run the helper interactively first") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CredentialError("secret path must be a regular file, not a symlink or special file")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise CredentialError("secret file must have mode 0600")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise CredentialError("secret file is not owned by the current user")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        current = os.fstat(fd)
        if not stat.S_ISREG(current.st_mode) or stat.S_IMODE(current.st_mode) != 0o600:
            raise CredentialError("secret file changed or is not mode 0600")
        data = b""
        while True:
            block = os.read(fd, 65536)
            if not block:
                break
            data += block
            if len(data) > 16384:
                raise CredentialError("secret file is unexpectedly large")
    finally:
        os.close(fd)
    try:
        token = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CredentialError("secret file is not valid UTF-8") from exc
    return _validate_token(token)


def _status(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"status": "missing", "path": str(path)}
    try:
        info = path.lstat()
    except FileNotFoundError:
        return result
    result["exists"] = True
    result["mode"] = f"{stat.S_IMODE(info.st_mode):04o}"
    result["regular_file"] = stat.S_ISREG(info.st_mode)
    result["symlink"] = stat.S_ISLNK(info.st_mode)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        result["status"] = "invalid_path"
        return result
    if stat.S_IMODE(info.st_mode) != 0o600:
        result["status"] = "invalid_permissions"
        return result
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        result["status"] = "wrong_owner"
        return result
    try:
        _read_token(path)
    except CredentialError:
        result["status"] = "invalid_contents"
    else:
        result["status"] = "ready"
    return result


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secret-path", default=None, help="local secret file (default: workspace.secrets/litellm_judge.key)")
    parser.add_argument("--status", action="store_true", help="print only non-secret JSON status and do not prompt")
    parser.add_argument("--force", action="store_true", help="explicitly replace an existing token file")
    parser.add_argument("--exec-env", metavar="ENV_NAME", help="load the token into this child environment variable")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run after --exec-env; put -- before it")
    args = parser.parse_args(argv)
    if args.exec_env and not ENV_NAME_RE.fullmatch(args.exec_env):
        parser.error("--exec-env must be an uppercase environment variable name")
    if args.status and (args.force or args.exec_env or args.command):
        parser.error("--status cannot be combined with --force, --exec-env, or a command")
    if args.command and not args.exec_env:
        parser.error("a command requires --exec-env")
    if args.exec_env:
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        if not args.command:
            parser.error("--exec-env requires a command after --")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    path = _secret_path(args.secret_path)
    try:
        if args.status:
            print(json.dumps(_status(path), ensure_ascii=False, sort_keys=True))
            return 0

        if args.exec_env:
            token = _read_token(path)
            child_env = os.environ.copy()
            child_env[args.exec_env] = token
            try:
                return subprocess.run(args.command, env=child_env, check=False).returncode
            finally:
                del token

        existing = _existing_regular_file(path)
        if existing is not None and not args.force:
            raise CredentialError(f"secret already exists; pass --force only to replace it: {path}")
        token = _validate_token(getpass.getpass(f"{TOKEN_PREFIX} (input hidden): "))
        _write_token(path, token, replace=args.force)
        print(json.dumps({"status": "stored", "path": str(path), "mode": "0600"}, ensure_ascii=False, sort_keys=True))
        del token
        return 0
    except (CredentialError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

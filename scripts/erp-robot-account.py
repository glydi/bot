#!/usr/bin/env python3
"""Create the robot's account on the school ERP (the Cloudflare Worker at
erp.xulo.in) through its admin API, and print the .env lines for the bot.

    python scripts/erp-robot-account.py --admin principal@school.example
    python scripts/erp-robot-account.py --admin ... --email glydi@school.example --name "GLYDI"

Needs an administrator login (the identifier you sign in with, and its
password, asked for without echo). Standard library only; no ERP code is
touched.

What it does, in order:
 1. Signs in as the administrator (GET /login for the CSRF cookie and
    token, then the form POST), keeping the erp_session cookie.
 2. POST /api/v1/admin/users: a user called NAME with EMAIL, the built-in
    "faculty" role (a default role, so no install step) and set_password,
    which issues a temporary password shown once. The robot's password is
    that one; "faculty" also lists it as a staff member, which is harmless.
 3. PUT /api/v1/admin/users/{id}/permissions: the direct grants the role
    lacks -- every section's attendance, school-wide reads, staff
    attendance and the staff list.
 4. Signs in as the robot and reads GET /api/v1/session to confirm the
    permissions took (they apply from the next sign-in, which this is).
 5. Prints the lines for the bot's .env.
"""
from __future__ import annotations

import argparse
import getpass
import http.cookiejar
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

NEEDED = [
    "students.read", "students.read.all", "academics.read", "academics.timetable.read",
    "academics.attendance.read", "academics.attendance.read.all", "academics.attendance.write",
    "academics.attendance.write.any", "academics.exams.read", "hr.employees.read", "hr.attendance.write",
]


class Erp:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar), NoRedirect()
        )

    def _open(self, req: urllib.request.Request):
        try:
            return self.opener.open(req, timeout=30)
        except urllib.error.HTTPError as e:
            return e

    def login(self, identifier: str, password: str) -> None:
        page = self._open(urllib.request.Request(self.base + "/login"))
        html = page.read().decode("utf-8", "replace")
        m = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', html) or re.search(
            r'value="([^"]+)"[^>]*name="csrf_token"', html
        )
        if not m:
            sys.exit("no csrf_token on the login page; is this the ERP?")
        data = urllib.parse.urlencode(
            {"identifier": identifier, "password": password, "csrf_token": m.group(1), "next": "/"}
        ).encode()
        resp = self._open(urllib.request.Request(self.base + "/login", data=data, method="POST"))
        names = {c.name for c in self.jar}
        if "erp_session" not in names:
            sys.exit(f"login as {identifier} failed (status {resp.status})")

    def json(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        resp = self._open(req)
        text = resp.read().decode("utf-8", "replace")
        try:
            return resp.status, json.loads(text) if text else {}
        except json.JSONDecodeError:
            return resp.status, {"raw": text[:300]}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="https://erp.xulo.in")
    ap.add_argument("--admin", required=True, help="administrator identifier (email, phone or username)")
    ap.add_argument("--email", default="glydi@robot.local", help="the robot's email, its sign-in identifier")
    ap.add_argument("--name", default="GLYDI robot")
    args = ap.parse_args()
    admin_pw = getpass.getpass(f"password for {args.admin}: ")

    erp = Erp(args.base)
    erp.login(args.admin, admin_pw)
    status, me = erp.json("GET", "/api/v1/session")
    if not me.get("authenticated"):
        sys.exit("admin session not authenticated")
    perms = set(me.get("permissions", []))
    for need in ("access.users.write",):
        if need not in perms:
            sys.exit(f"{args.admin} lacks {need}; use an account with it")
    print(f"signed in as {me['user']['full_name']} at {me['institution']['name']}")

    status, created = erp.json(
        "POST",
        "/api/v1/admin/users",
        {"full_name": args.name, "email": args.email, "role_keys": ["faculty"], "set_password": True},
    )
    if status not in (200, 201):
        sys.exit(f"create user: {status} {created}")
    user_id = created["id"]
    temp = created.get("temporary_password")
    print(f"user {user_id} created ({created.get('status')})")
    if not temp:
        sys.exit("no temporary password came back; the email may already exist -- reset its password in the console")

    status, granted = erp.json("PUT", f"/api/v1/admin/users/{user_id}/permissions", {"permission_keys": NEEDED})
    if status != 200:
        sys.exit(f"grant permissions: {status} {granted}")
    print(f"direct grants: {len(granted.get('direct_keys', []))}")

    robot = Erp(args.base)
    robot.login(args.email, temp)
    status, session = robot.json("GET", "/api/v1/session")
    have = set(session.get("permissions", []))
    missing = [p for p in NEEDED if p not in have]
    if session.get("user", {}).get("must_change_password"):
        print("note: the account is asked to change its password; do that once in the ERP, then update .env")
    if missing:
        print("missing after sign-in:", ", ".join(missing))
    else:
        print("the robot holds every permission it needs")

    print("\nadd to the bot's .env (the password is shown once):")
    print(f"GLYDI_ERP_URL={args.base}")
    print(f"GLYDI_ERP_USER={args.email}")
    print(f"GLYDI_ERP_PASSWORD={temp}")


if __name__ == "__main__":
    main()

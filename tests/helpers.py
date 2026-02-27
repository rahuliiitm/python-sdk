"""HTTP utility and test setup/teardown for SDK integration tests."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.request import Request, urlopen

API_BASE = os.environ.get("API_URL", "http://localhost:3001")


def api_call(
    path: str,
    *,
    method: str = "GET",
    body: Optional[dict] = None,
    token: Optional[str] = None,
) -> Any:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = json.dumps(body).encode() if body else None
    req = Request(f"{API_BASE}{path}", data=data, headers=headers, method=method)
    resp = urlopen(req, timeout=15)

    if resp.status == 204 or resp.headers.get("content-length") == "0":
        return None
    return json.loads(resp.read().decode())


@dataclass
class TestContext:
    jwt: str
    user_id: str
    project_id: str
    prompt_id: str
    prompt_slug: str
    version_id: str
    environment_id: str
    sdk_api_key: str  # raw key (lp_live_...)
    api_key_id: str


def setup() -> TestContext:
    """Bootstrap fresh test data: user, project, prompt, version, deployment, API key."""
    email = "rahul.iiitm06@gmail.com"
    password = "12345678"

    # 1. Login
    auth = api_call("/auth/login", method="POST", body={"email": email, "password": password})
    jwt = auth["accessToken"]

    # 2. Get profile → projectId
    me = api_call("/auth/me", token=jwt)
    project_id = me["projectId"]

    # 3. Get environments
    envs = api_call(f"/environment/{project_id}", token=jwt)
    env = envs[0]

    # 4. Create a managed prompt
    suffix = hex(int(time.time() * 1000))[-4:]
    prompt_slug = f"sdk-test-prompt-{suffix}"
    prompt = api_call(
        f"/prompt/{project_id}",
        method="POST",
        body={
            "slug": prompt_slug,
            "name": f"SDK Test Prompt ({suffix})",
            "description": "Integration test prompt with template variables",
            "initialContent": "Hello {{name}}, you are a {{role}}. Welcome to {{company}}!",
        },
        token=jwt,
    )
    version_id = prompt["versions"][0]["id"]

    # 5. Deploy version to environment
    api_call(
        f"/prompt/{project_id}/{prompt['id']}/versions/{version_id}/deploy-to/{env['id']}",
        method="POST",
        token=jwt,
    )

    # 6. Generate SDK API key
    key_res = api_call(
        f"/project/{project_id}/api-keys",
        method="POST",
        body={"name": "SDK Integration Test Key", "environmentId": env["id"]},
        token=jwt,
    )

    return TestContext(
        jwt=jwt,
        user_id=me["id"],
        project_id=project_id,
        prompt_id=prompt["id"],
        prompt_slug=prompt_slug,
        version_id=version_id,
        environment_id=env["id"],
        sdk_api_key=key_res["rawKey"],
        api_key_id=key_res["apiKey"]["id"],
    )


def teardown(ctx: TestContext) -> None:
    """Clean up test data."""
    try:
        api_call(f"/prompt/{ctx.project_id}/{ctx.prompt_id}", method="DELETE", token=ctx.jwt)
    except Exception:
        pass
    try:
        api_call(f"/project/{ctx.project_id}/api-keys/{ctx.api_key_id}", method="DELETE", token=ctx.jwt)
    except Exception:
        pass

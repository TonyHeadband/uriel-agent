import html
from urllib.parse import quote

from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from uriel.config import GatewaySettings
from uriel.gateway.auth import AuthError, principal_from_claims

_SIGNIN_FAILED = "The sign-in attempt failed or expired. Please try again."


def register_oauth(settings: GatewaySettings) -> OAuth:
    oauth = OAuth()
    oauth.register(
        "authelia",
        client_id=settings.oidc_client_id,
        client_secret=settings.oidc_client_secret,
        server_metadata_url=f"{settings.oidc_issuer}/.well-known/openid-configuration",
        client_kwargs={"scope": "openid profile email groups", "code_challenge_method": "S256"},
    )
    return oauth


def build_router(settings: GatewaySettings) -> APIRouter:
    router = APIRouter(prefix="/auth")

    @router.get("/login")
    async def login(request: Request):
        # Build redirect_uri from public_url: behind Traefik, request.url_for would say http://.
        return await request.app.state.services.oauth.authelia.authorize_redirect(
            request, f"{settings.public_url}/auth/callback"
        )

    @router.get("/callback")
    async def callback(request: Request):
        try:
            token = await request.app.state.services.oauth.authelia.authorize_access_token(request)
        except OAuthError:
            # e.g. a replayed or expired callback (MismatchingStateError and friends); never
            # echo the provider's own error text back to the browser.
            request.session.clear()
            return HTMLResponse(
                f"<h1>Can't sign you in</h1><p>{html.escape(_SIGNIN_FAILED)}</p>", status_code=400
            )
        try:
            principal = principal_from_claims(token.get("userinfo") or {})
        except AuthError as exc:
            request.session.clear()
            return HTMLResponse(f"<h1>Can't sign you in</h1><p>{html.escape(str(exc))}</p>", status_code=403)
        request.session.clear()
        request.session["principal"] = principal.to_session()
        return RedirectResponse("/", status_code=303)

    @router.post("/logout")
    async def logout(request: Request):
        # Redirecting to "/" would go to /auth/login, where Authelia's live SSO session signs
        # the same user straight back in - a shared family tablet would never actually sign out.
        request.session.clear()
        portal_link = ""
        if settings.oidc_logout_url:
            rd = quote(settings.public_url, safe="")
            portal_link = (
                f'<p><a href="{html.escape(f"{settings.oidc_logout_url}?rd={rd}")}">'
                "Sign out of all family services</a></p>"
            )
        return HTMLResponse(f'<h1>Signed out</h1><p><a href="/auth/login">Sign in again</a></p>{portal_link}')

    return router

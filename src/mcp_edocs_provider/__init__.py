"""Reusable AAuth-protected eDocs MCP provider."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from aauth_edocs import (
    AGENT_TYP,
    AUTH_TYP,
    AAuthError,
    Dataflow,
    FunctionDescriptor,
    SigningKey,
    VerifiedRequest,
    hash_function_args,
    peek_jwt,
)
from aauth_edocs.errors import DENIED, INVALID_TOKEN
from aauth_edocs.httpsig import KeyResolver
from aauth_edocs.resource import (
    ResourceConfig,
    mint_resource_token,
    resource_auth_challenge,
    resource_jwks_document,
    token_resource_metadata,
)
from mcp.server import MCPServer
from mcp_aauth import aauth_agent_authentication, aauth_authorization
from mcp_types import Resource as MCPResource
from starlette.types import ASGIApp, Receive, Scope, Send


@dataclass
class CatalogEntry:
    edoc_id: str
    resource_uri: str
    title: str
    description: str
    enabled: bool
    storage: Any
    controllers: tuple[str, ...] | None = None

    def public_dict(self, *, include_enabled: bool = False) -> dict[str, Any]:
        value = {
            "edoc_id": self.edoc_id,
            "resource_uri": self.resource_uri,
            "title": self.title,
            "description": self.description,
        }
        if include_enabled:
            value["enabled"] = self.enabled
        if self.controllers is not None:
            value["controllers"] = list(self.controllers)
        return value


class ProviderCatalog:
    def __init__(self, entries: tuple[CatalogEntry, ...] = ()) -> None:
        self._entries = {entry.edoc_id: entry for entry in entries}

    def add(self, entry: CatalogEntry) -> CatalogEntry:
        if entry.edoc_id in self._entries:
            raise ValueError(f"eDoc already exists: {entry.edoc_id}")
        self._entries[entry.edoc_id] = entry
        return entry

    def get(
        self,
        edoc_id: str,
        *,
        include_disabled: bool = False,
    ) -> CatalogEntry | None:
        entry = self._entries.get(edoc_id)
        if entry is None or (not include_disabled and not entry.enabled):
            return None
        return entry

    def list(self, *, include_disabled: bool = False) -> tuple[CatalogEntry, ...]:
        return tuple(
            entry
            for entry in self._entries.values()
            if include_disabled or entry.enabled
        )

    def update_metadata(
        self,
        edoc_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
    ) -> CatalogEntry:
        entry = self._required(edoc_id)
        if title is not None:
            if not isinstance(title, str) or not title.strip():
                raise ValueError("title must be a non-empty string")
            entry.title = title.strip()
        if description is not None:
            if not isinstance(description, str):
                raise ValueError("description must be a string")
            entry.description = description
        return entry

    def set_enabled(self, edoc_id: str, enabled: bool) -> CatalogEntry:
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        entry = self._required(edoc_id)
        entry.enabled = enabled
        return entry

    def _required(self, edoc_id: str) -> CatalogEntry:
        try:
            return self._entries[edoc_id]
        except KeyError as error:
            raise LookupError("unknown eDoc") from error


@dataclass(frozen=True)
class LoadedFunction:
    descriptor: FunctionDescriptor
    implementation: Callable[[Any, Mapping[str, Any]], dict[str, Any]]


class LocalFunctionLoader:
    def __init__(self, functions: Mapping[str, LoadedFunction]) -> None:
        self._functions = dict(functions)

    def load(
        self,
        descriptor: FunctionDescriptor,
    ) -> Callable[[Any, Mapping[str, Any]], dict[str, Any]]:
        registration = self._functions.get(descriptor.id)
        if registration is None:
            raise LookupError(f"function is not loaded: {descriptor.id}")
        if registration.descriptor != descriptor:
            raise ValueError(
                f"loaded descriptor does not match: {descriptor.id}"
            )
        return registration.implementation


class MutableFunctionRegistry:
    def __init__(self) -> None:
        self._functions: dict[str, LoadedFunction] = {}
        self._artifacts: dict[str, dict[str, Any]] = {}

    def register(
        self,
        registration: LoadedFunction,
        *,
        artifact: dict[str, Any] | None = None,
    ) -> LoadedFunction:
        function_id = registration.descriptor.id
        if function_id in self._functions:
            raise ValueError(f"function already exists: {function_id}")
        self._functions[function_id] = registration
        if artifact is not None:
            self._artifacts[function_id] = artifact
        return registration

    def get(self, function_id: str) -> LoadedFunction | None:
        return self._functions.get(function_id)

    def load(
        self,
        descriptor: FunctionDescriptor,
    ) -> Callable[[Any, Mapping[str, Any]], dict[str, Any]]:
        registration = self._functions.get(descriptor.id)
        if registration is None or registration.descriptor.digest != descriptor.digest:
            raise LookupError(f"function is not loaded: {descriptor.id}")
        return registration.implementation

    def artifact(self, function_id: str) -> dict[str, Any] | None:
        return self._artifacts.get(function_id)

    def items(self) -> Iterator[tuple[str, LoadedFunction]]:
        return iter(self._functions.items())


@dataclass(frozen=True)
class ProviderServerConfig:
    provider_id: str
    display_name: str
    resource_issuer: str
    sentinel_url: str
    source_agent: str
    signing_key: SigningKey
    authoritative_controllers: tuple[str, ...]


@dataclass
class ProviderResource:
    config: ProviderServerConfig
    catalog: ProviderCatalog
    functions: MutableFunctionRegistry
    loader: LocalFunctionLoader
    on_materialized: Callable[
        [Dataflow, dict[str, Any], tuple[str, ...]], Any
    ] | None = None

    def authorize(
        self,
        verified_agent: VerifiedRequest,
        *,
        provider_id: str,
        edoc_id: str,
        function_id: str,
        function_args: dict[str, Any],
    ) -> str:
        self._check_provider(provider_id)
        if self.catalog.get(edoc_id) is None or self.functions.get(function_id) is None:
            raise AAuthError(DENIED, 403, "eDoc or function is unavailable")
        return mint_resource_token(
            _aauth_resource(self.config),
            verified_agent,
            function_id,
            source_agent=self.config.source_agent,
            edoc_id=edoc_id,
            controllers=self._controllers_for(edoc_id),
            function_args=function_args,
        )

    def execute(
        self,
        authorization: VerifiedRequest,
        *,
        provider_id: str,
        edoc_id: str,
        function_id: str,
        function_args: dict[str, Any],
    ) -> dict[str, Any]:
        self._check_provider(provider_id)
        destination_agent = authorization.claims.get("agent")
        if not isinstance(destination_agent, str):
            raise AAuthError(
                INVALID_TOKEN,
                401,
                "authorization agent is incomplete",
            )
        controllers = self._controllers_for(edoc_id)
        expected = {
            "iss": self.config.sentinel_url,
            "aud": self.config.resource_issuer,
            "source_agent": self.config.source_agent,
            "scope": function_id,
            "edoc_id": edoc_id,
            "controllers": list(controllers),
            "function_args_hash": hash_function_args(function_args),
        }
        for name, value in expected.items():
            if authorization.claims.get(name) != value:
                raise AAuthError(
                    INVALID_TOKEN,
                    401,
                    f"authorization {name} does not match the invocation",
                )
        document = self.catalog.get(edoc_id)
        registration = self.functions.get(function_id)
        if document is None or registration is None:
            raise AAuthError(DENIED, 403, "eDoc or function is unavailable")
        implementation = self.loader.load(registration.descriptor)
        result = implementation(document.storage, function_args)
        if self.on_materialized is not None:
            derived = self.on_materialized(
                Dataflow.from_arguments(
                    self.config.source_agent,
                    function_id,
                    edoc_id,
                    destination_agent,
                    function_args,
                ),
                result,
                controllers,
            )
            edoc_id_value = getattr(derived, "edoc_id", None)
            if isinstance(edoc_id_value, str) and edoc_id_value:
                result = {**result, "derived_edoc_id": edoc_id_value}
        return result

    def _controllers_for(self, edoc_id: str) -> tuple[str, ...]:
        entry = self.catalog.get(edoc_id, include_disabled=True)
        if entry is not None and entry.controllers is not None:
            return entry.controllers
        return self.config.authoritative_controllers

    def _check_provider(self, provider_id: str) -> None:
        if provider_id != self.config.provider_id:
            raise AAuthError(
                DENIED,
                403,
                f"resource belongs to provider {self.config.provider_id}",
            )


ProviderApplication = ASGIApp


class _ProviderApplication:
    def __init__(
        self,
        resource: ProviderResource,
        downstream: ASGIApp,
        *,
        key_resolver: KeyResolver,
    ) -> None:
        self.resource = resource
        self.downstream = downstream
        self.challenge_app = aauth_agent_authentication(
            key_resolver=key_resolver
        )(self._challenge)
        self.authorize_app = aauth_agent_authentication(
            key_resolver=key_resolver
        )(self._authorize)
        self.authorized_app = aauth_authorization(
            key_resolver=key_resolver,
            issuer=resource.config.sentinel_url,
            audience=resource.config.resource_issuer,
        )(downstream)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path")
        if scope["type"] == "http" and path in {
            "/.well-known/aauth-resource.json",
            "/jwks.json",
        }:
            value = (
                token_resource_metadata(self.resource.config.resource_issuer)
                if path == "/.well-known/aauth-resource.json"
                else resource_jwks_document(self.resource.config.signing_key)
            )
            await _send_json(send, 200, value)
            return
        if scope["type"] == "http" and path == "/authorize":
            await self.authorize_app(scope, receive, send)
            return
        if scope["type"] == "http" and path.startswith("/admin/documents"):
            await self._admin(scope, receive, send)
            return
        if (
            scope["type"] == "http"
            and path == "/mcp"
            and _presented_token_type(scope) == AGENT_TYP
        ):
            await self.challenge_app(scope, receive, send)
            return
        if (
            scope["type"] == "http"
            and path == "/mcp"
            and _presented_token_type(scope) == AUTH_TYP
        ):
            await self.authorized_app(scope, receive, send)
            return
        await self.downstream(scope, receive, send)

    async def _authorize(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope.get("method") != "POST":
            await _send_json(send, 405, {"error": "method_not_allowed"})
            return
        try:
            body = await _read_json(receive)
            if set(body) != {"edoc_id", "function_id", "function_args"}:
                raise ValueError("authorization request has the wrong fields")
            arguments = body["function_args"]
            if not isinstance(arguments, dict):
                raise ValueError("function_args must be a JSON object")
            provider_id = _header(scope, b"edocs-provider")
            token = self.resource.authorize(
                scope["aauth"],
                provider_id=provider_id,
                edoc_id=body["edoc_id"],
                function_id=body["function_id"],
                function_args=arguments,
            )
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError, ValueError) as error:
            await _send_json(
                send,
                400,
                {"error": "invalid_request", "detail": str(error)},
            )
            return
        except AAuthError as error:
            await _send_json(send, error.status, error.body())
            return
        await _send_json(send, 200, {"resource_token": token})

    async def _challenge(
        self,
        scope: Scope,
        _receive: Receive,
        send: Send,
    ) -> None:
        body, headers = resource_auth_challenge(
            _aauth_resource(self.resource.config),
            scope["aauth"],
            "identity@1",
            source_agent=self.resource.config.source_agent,
            edoc_id="catalog",
            controllers=self.resource.config.authoritative_controllers,
        )
        await _send_json(
            send,
            401,
            body,
            headers=[
                (name.lower().encode(), value.encode())
                for name, value in headers.items()
            ],
        )

    async def _admin(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        path = scope["path"].removeprefix("/admin/documents").strip("/")
        method = scope.get("method")
        status = 200
        try:
            if not path and method == "GET":
                value = {
                    "documents": [
                        entry.public_dict(include_enabled=True)
                        for entry in self.resource.catalog.list(
                            include_disabled=True
                        )
                    ]
                }
            elif not path and method == "POST":
                body = await _read_json(receive)
                required = {
                    "edoc_id",
                    "title",
                    "description",
                    "storage",
                    "controllers",
                }
                if set(body) != required:
                    raise ValueError(
                        "document requires edoc_id, title, description, "
                        "storage, and controllers"
                    )
                controllers = body["controllers"]
                if (
                    not isinstance(controllers, list)
                    or not controllers
                    or any(not isinstance(item, str) or not item for item in controllers)
                ):
                    raise ValueError("controllers must be a non-empty string list")
                if not isinstance(body["storage"], dict):
                    raise ValueError("storage must be a JSON object")
                edoc_id = body["edoc_id"]
                if not isinstance(edoc_id, str) or not edoc_id:
                    raise ValueError("edoc_id must be a non-empty string")
                entry = self.resource.catalog.add(
                    CatalogEntry(
                        edoc_id=edoc_id,
                        resource_uri=(
                            f"edoc://{self.resource.config.provider_id}/{edoc_id}"
                        ),
                        title=body["title"],
                        description=body["description"],
                        enabled=True,
                        storage=body["storage"],
                        controllers=tuple(controllers),
                    )
                )
                value = {"document": entry.public_dict(include_enabled=True)}
                status = 201
            elif path and "/" not in path and method == "PATCH":
                body = await _read_json(receive)
                if not body or not set(body).issubset({"title", "description"}):
                    raise ValueError("only title and description may be changed")
                entry = self.resource.catalog.update_metadata(
                    path,
                    title=body.get("title"),
                    description=body.get("description"),
                )
                value = {"document": entry.public_dict(include_enabled=True)}
            elif path.endswith("/enabled") and method == "PUT":
                edoc_id = path.removesuffix("/enabled").rstrip("/")
                body = await _read_json(receive)
                if set(body) != {"enabled"}:
                    raise ValueError("enabled update requires one boolean field")
                entry = self.resource.catalog.set_enabled(
                    edoc_id, body["enabled"]
                )
                value = {"document": entry.public_dict(include_enabled=True)}
            else:
                await _send_json(send, 405, {"error": "method_not_allowed"})
                return
        except LookupError as error:
            await _send_json(send, 404, {"error": "not_found", "detail": str(error)})
            return
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            await _send_json(
                send, 400, {"error": "invalid_request", "detail": str(error)}
            )
            return
        await _send_json(send, status, value)


def _aauth_resource(config: ProviderServerConfig) -> ResourceConfig:
    """The provider is an AAuth resource whose audience is the Sentinel."""
    return ResourceConfig(
        issuer=config.resource_issuer,
        key=config.signing_key,
        as_url=config.sentinel_url,
    )


def build_provider_server(
    config: ProviderServerConfig,
    *,
    catalog: ProviderCatalog,
    functions: MutableFunctionRegistry,
    loader: LocalFunctionLoader,
    key_resolver: KeyResolver,
    register_tools: Callable[[MCPServer, ProviderResource], None],
    on_materialized: Callable[
        [Dataflow, dict[str, Any], tuple[str, ...]], Any
    ] | None = None,
) -> ProviderApplication:
    resource = ProviderResource(
        config,
        catalog,
        functions,
        loader,
        on_materialized,
    )
    mcp = MCPServer(f"{config.display_name} eDocs provider")
    register_tools(mcp, resource)

    async def list_resources() -> list[MCPResource]:
        return [
            MCPResource(
                uri=entry.resource_uri,
                name=entry.edoc_id,
                title=entry.title,
                description=entry.description,
            )
            for entry in catalog.list()
        ]

    mcp.list_resources = list_resources
    downstream = mcp.streamable_http_app(
        host=config.resource_issuer.removeprefix("http://").removeprefix(
            "https://"
        ),
    )
    return _ProviderApplication(resource, downstream, key_resolver=key_resolver)


async def _send_json(
    send: Send,
    status: int,
    value: dict[str, Any],
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = json.dumps(value, separators=(",", ":")).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                *(headers or []),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _read_json(receive: Receive) -> dict[str, Any]:
    chunks = []
    while True:
        message = await receive()
        chunks.append(message.get("body", b""))
        if not message.get("more_body"):
            break
    value = json.loads(b"".join(chunks).decode())
    if not isinstance(value, dict):
        raise TypeError("JSON object required")
    return value


def _header(scope: Scope, name: bytes) -> str:
    for header_name, value in scope.get("headers", []):
        if header_name.lower() == name:
            return value.decode("latin-1")
    raise ValueError(f"missing {name.decode()} header")


def _presented_token_type(scope: Scope) -> str | None:
    try:
        signature_key = _header(scope, b"signature-key")
        token = signature_key.split('jwt="', 1)[1].split('"', 1)[0]
        return peek_jwt(token)[0].get("typ")
    except (AAuthError, IndexError, ValueError):
        return None


__all__ = [
    "CatalogEntry",
    "LoadedFunction",
    "LocalFunctionLoader",
    "MutableFunctionRegistry",
    "ProviderApplication",
    "ProviderCatalog",
    "ProviderResource",
    "ProviderServerConfig",
    "build_provider_server",
]

import hmac
import json
import secrets
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol
from urllib.parse import urlsplit


TOKEN_PARAM = "CompanionApiToken"
MAX_REQUEST_BYTES = 16 * 1024


class ParamStore(Protocol):
  def get(self, key: str, block: bool = False, return_default: bool = False): ...
  def get_bool(self, key: str, block: bool = False) -> bool: ...
  def put(self, key: str, dat, block: bool = False) -> None: ...
  def put_bool(self, key: str, val: bool, block: bool = False) -> None: ...


class StateProvider(Protocol):
  def snapshot(self) -> dict: ...


@dataclass(frozen=True)
class ApiResponse:
  status: HTTPStatus
  body: dict


@dataclass(frozen=True)
class ApiError(Exception):
  status: HTTPStatus
  message: str


@dataclass(frozen=True)
class SettingSpec:
  param: str
  kind: str

  def read(self, params: ParamStore):
    value = params.get_bool(self.param) if self.kind == "bool" else params.get(self.param, return_default=True)
    if self.kind == "personality":
      return {0: "aggressive", 1: "standard", 2: "relaxed"}.get(value, "standard")
    return value

  def validate(self, value):
    if self.kind == "bool":
      if type(value) is not bool:
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid value")
      return value
    if self.kind == "personality" and value in {"aggressive", "standard", "relaxed"}:
      return {"aggressive": 0, "standard": 1, "relaxed": 2}[value]
    raise ApiError(HTTPStatus.BAD_REQUEST, "invalid value")

  def write(self, params: ParamStore, value) -> None:
    if self.kind == "bool":
      params.put_bool(self.param, value, block=True)
    else:
      params.put(self.param, value, block=True)


SETTINGS = {
  "experimentalMode": SettingSpec("ExperimentalMode", "bool"),
  "longitudinalPersonality": SettingSpec("LongitudinalPersonality", "personality"),
  "laneDepartureWarning": SettingSpec("IsLdwEnabled", "bool"),
  "disengageOnAccelerator": SettingSpec("DisengageOnAccelerator", "bool"),
}


def get_or_create_token(params: ParamStore) -> str:
  token = params.get(TOKEN_PARAM)
  if token:
    return token

  token = secrets.token_urlsafe(32)
  params.put(TOKEN_PARAM, token, block=True)
  return token


def rotate_token(params: ParamStore) -> str:
  token = secrets.token_urlsafe(32)
  params.put(TOKEN_PARAM, token, block=True)
  return token


class CerealStateProvider:
  """Read the small live-state subset directly from openpilot msgq/cereal."""

  def __init__(self):
    import openpilot.cereal.messaging as messaging

    self._sm = messaging.SubMaster(["deviceState", "carState", "selfdriveState", "radarState"])
    self._lock = threading.Lock()

  @staticmethod
  def _personality_name(personality) -> str:
    return {0: "aggressive", 1: "standard", 2: "relaxed"}.get(int(personality), "unknown")

  def snapshot(self) -> dict:
    with self._lock:
      self._sm.update(0)
      car_state = self._sm["carState"]
      selfdrive_state = self._sm["selfdriveState"]
      lead = self._sm["radarState"].leadOne
      lead_present = bool(lead.present)
      return {
        "started": bool(self._sm["deviceState"].started),
        "vehicleSpeedMps": float(car_state.vEgo),
        "engaged": bool(selfdrive_state.enabled),
        "setSpeedMps": float(car_state.cruiseState.speed),
        "lead": {
          "present": lead_present,
          "distanceM": float(lead.dRel) if lead_present else None,
          "relativeSpeedMps": float(lead.vRel) if lead_present else None,
        },
        "drivingPersonality": self._personality_name(selfdrive_state.personality),
      }


class CompanionApi:
  def __init__(self, params: ParamStore, state_provider: StateProvider, device_name: str, token: str):
    self._params = params
    self._state_provider = state_provider
    self._device_name = device_name
    self._token = token

  def handle(self, method: str, target: str, headers, body: bytes) -> ApiResponse:
    path = urlsplit(target).path
    if method == "GET" and path == "/v1/health":
      return self._health()

    if not self._authorized(headers.get("Authorization")):
      return self._error(HTTPStatus.UNAUTHORIZED, "unauthorized")

    try:
      if method == "GET" and path == "/v1/settings":
        return self._settings()
      if method == "PUT" and path.startswith("/v1/settings/"):
        return self._put_setting(path.removeprefix("/v1/settings/"), body)
      if method == "GET" and path == "/v1/state":
        return ApiResponse(HTTPStatus.OK, self._state_provider.snapshot())
      return self._error(HTTPStatus.NOT_FOUND, "not found")
    except ApiError as error:
      return self._error(error.status, error.message)
    except Exception:
      return self._error(HTTPStatus.SERVICE_UNAVAILABLE, "service unavailable")

  def _health(self) -> ApiResponse:
    try:
      started = self._state_provider.snapshot()["started"]
    except Exception:
      started = False
    return ApiResponse(
      HTTPStatus.OK,
      {
        "ok": True,
        "device": self._device_name,
        "openpilotVersion": self._params.get("Version") or "unknown",
        "started": started,
      },
    )

  def _settings(self) -> ApiResponse:
    return ApiResponse(HTTPStatus.OK, {"settings": {name: self._setting_value(spec) for name, spec in SETTINGS.items()}})

  def _put_setting(self, name: str, body: bytes) -> ApiResponse:
    spec = SETTINGS.get(name)
    if spec is None or "/" in name:
      raise ApiError(HTTPStatus.NOT_FOUND, "unknown setting")

    try:
      data = json.loads(body)
    except (TypeError, json.JSONDecodeError):
      raise ApiError(HTTPStatus.BAD_REQUEST, "invalid json") from None
    if type(data) is not dict or set(data) != {"value"}:
      raise ApiError(HTTPStatus.BAD_REQUEST, "invalid request")

    value = spec.validate(data["value"])
    if name == "experimentalMode" and value and not self._params.get_bool("ExperimentalModeConfirmed"):
      raise ApiError(HTTPStatus.CONFLICT, "experimental mode must first be confirmed on the comma device")
    spec.write(self._params, value)
    return ApiResponse(HTTPStatus.OK, {"setting": self._setting_value(spec)})

  def _setting_value(self, spec: SettingSpec) -> dict:
    return {"value": spec.read(self._params), "apply": "polled", "pollIntervalMs": 100}

  def _authorized(self, authorization: str | None) -> bool:
    expected = f"Bearer {self._token}"
    return authorization is not None and hmac.compare_digest(authorization, expected)

  @staticmethod
  def _error(status: HTTPStatus, message: str) -> ApiResponse:
    return ApiResponse(status, {"error": message})


class CompanionHttpServer(ThreadingHTTPServer):
  daemon_threads = True
  allow_reuse_address = True


def create_http_server(host: str, port: int, api: CompanionApi) -> CompanionHttpServer:
  class Handler(BaseHTTPRequestHandler):
    server_version = "companiond"
    sys_version = ""

    def do_GET(self):
      self._respond(b"")

    def do_PUT(self):
      try:
        content_length = int(self.headers.get("Content-Length", "0"))
      except ValueError:
        self._write_response(CompanionApi._error(HTTPStatus.BAD_REQUEST, "invalid content length"))
        return
      if content_length < 0 or content_length > MAX_REQUEST_BYTES:
        self._write_response(CompanionApi._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request too large"))
        return
      self._respond(self.rfile.read(content_length))

    def do_POST(self):
      self._write_response(CompanionApi._error(HTTPStatus.METHOD_NOT_ALLOWED, "method not allowed"))

    def log_message(self, format: str, *args) -> None:  # noqa: A002
      pass

    def _respond(self, body: bytes):
      self._write_response(api.handle(self.command, self.path, self.headers, body))

    def _write_response(self, response: ApiResponse):
      body = json.dumps(response.body, separators=(",", ":")).encode()
      self.send_response(response.status)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      self.wfile.write(body)

  return CompanionHttpServer((host, port), Handler)

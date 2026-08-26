import http.client
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from openpilot.system.companiond.api import CerealStateProvider, CompanionApi, create_http_server


class FakeParams:
  def __init__(self):
    self.values = {
      "Version": "0.10.0-test",
      "ExperimentalMode": False,
      "ExperimentalModeConfirmed": False,
      "LongitudinalPersonality": 1,
      "IsLdwEnabled": True,
      "DisengageOnAccelerator": False,
    }

  def get(self, key, block=False, return_default=False):
    del block
    value = self.values.get(key)
    return 1 if key == "LongitudinalPersonality" and return_default and value is None else value

  def get_bool(self, key, block=False):
    del block
    return bool(self.values.get(key, False))

  def put(self, key, dat, block=False):
    del block
    self.values[key] = dat

  def put_bool(self, key, val, block=False):
    del block
    self.values[key] = val


class FakeStateProvider:
  def snapshot(self):
    return {
      "started": True,
      "vehicleSpeedMps": 25.0,
      "engaged": True,
      "setSpeedMps": 27.8,
      "lead": {"present": True, "distanceM": 31.5, "relativeSpeedMps": -1.2},
      "drivingPersonality": "standard",
    }


@contextmanager
def running_server(api: CompanionApi) -> Iterator[tuple[str, int]]:
  server = create_http_server("127.0.0.1", 0, api)
  thread = threading.Thread(target=server.serve_forever)
  thread.start()
  try:
    host, port = server.server_address[:2]
    yield str(host), int(port)
  finally:
    server.shutdown()
    thread.join()
    server.server_close()


@pytest.fixture
def params():
  return FakeParams()


@pytest.fixture
def api(params):
  return CompanionApi(params, FakeStateProvider(), device_name="comma four", token="test-token")


def request(address, method, path, body=None, token=None):
  headers = {}
  if body is not None:
    headers["Content-Type"] = "application/json"
  if token is not None:
    headers["Authorization"] = f"Bearer {token}"
  connection = http.client.HTTPConnection(*address)
  connection.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers)
  response = connection.getresponse()
  data = json.loads(response.read())
  connection.close()
  return response.status, data


def test_health_is_public_and_does_not_return_the_token(api):
  with running_server(api) as address:
    status, data = request(address, "GET", "/v1/health")

  assert status == 200
  assert data == {"ok": True, "device": "comma four", "openpilotVersion": "0.10.0-test", "started": True}
  assert "token" not in data


def test_settings_require_a_bearer_token(api):
  with running_server(api) as address:
    status, data = request(address, "GET", "/v1/settings")

  assert status == 401
  assert data == {"error": "unauthorized"}


def test_settings_are_an_explicit_friendly_name_allowlist(api):
  with running_server(api) as address:
    status, data = request(address, "GET", "/v1/settings", token="test-token")

  assert status == 200
  assert data["settings"] == {
    "experimentalMode": {"value": False, "apply": "polled", "pollIntervalMs": 100},
    "longitudinalPersonality": {"value": "standard", "apply": "polled", "pollIntervalMs": 100},
    "laneDepartureWarning": {"value": True, "apply": "polled", "pollIntervalMs": 100},
    "disengageOnAccelerator": {"value": False, "apply": "polled", "pollIntervalMs": 100},
  }


def test_safe_setting_update_validates_and_persists_to_its_one_allowlisted_param(api, params):
  with running_server(api) as address:
    status, data = request(address, "PUT", "/v1/settings/disengageOnAccelerator", {"value": True}, token="test-token")

  assert status == 200
  assert data == {"setting": {"value": True, "apply": "polled", "pollIntervalMs": 100}}
  assert params.values["DisengageOnAccelerator"] is True


def test_unknown_or_invalid_setting_writes_are_rejected(api, params):
  with running_server(api) as address:
    unknown_status, unknown = request(address, "PUT", "/v1/settings/DoReboot", {"value": True}, token="test-token")
    invalid_status, invalid = request(address, "PUT", "/v1/settings/longitudinalPersonality", {"value": "turbo"}, token="test-token")

  assert unknown_status == 404
  assert unknown == {"error": "unknown setting"}
  assert invalid_status == 400
  assert invalid == {"error": "invalid value"}
  assert "DoReboot" not in params.values


def test_experimental_mode_cannot_bypass_the_on_device_confirmation(api, params):
  with running_server(api) as address:
    status, data = request(address, "PUT", "/v1/settings/experimentalMode", {"value": True}, token="test-token")

  assert status == 409
  assert data == {"error": "experimental mode must first be confirmed on the comma device"}
  assert params.values["ExperimentalMode"] is False


def test_state_uses_the_injected_cereal_provider_and_requires_authentication(api):
  with running_server(api) as address:
    status, data = request(address, "GET", "/v1/state", token="test-token")

  assert status == 200
  assert data["engaged"] is True
  assert data["lead"] == {"present": True, "distanceM": 31.5, "relativeSpeedMps": -1.2}


def test_cereal_personality_values_are_serialized_as_friendly_names():
  assert CerealStateProvider._personality_name(0) == "aggressive"
  assert CerealStateProvider._personality_name(1) == "standard"
  assert CerealStateProvider._personality_name(2) == "relaxed"

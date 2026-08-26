# Local companion API spike

## Recommendation

**GO WITH CAVEATS.** The source supports a small authenticated local service that reads an explicit setting allowlist and consumes existing local cereal/msgq messages. It does not require comma Prime or comma cloud services. The implementation is intentionally small and dependency-free.

The missing decision gate is physical networking: this checkout configures a shared Wi-Fi hotspot for both target codenames, but it does not contain the device OS firewall, network-namespace, or actual radio-driver configuration. Inbound reachability, AP operation, and multicast discovery on both devices are therefore **NEEDS HARDWARE VERIFICATION**.

## What was implemented

`companiond` is a manager-owned Python process on comma hardware. It binds to `0.0.0.0:8757` by default; manual runs can override `--host` and `--port`.

```text
Phone or laptop on the local LAN
        |
        | HTTP + Authorization: Bearer <pairing token>
        v
  companiond (standard-library HTTP server)
        |                         |
        | explicit allowlist      | local msgq subscriptions
        v                         v
     Params                 cereal: deviceState, carState,
                              selfdriveState, radarState
```

The service files are [api.py](openpilot/system/companiond/api.py), [main.py](openpilot/system/companiond/main.py), and [its focused tests](openpilot/system/companiond/tests/test_api.py). The manager starts it only where `COMMA_HARDWARE` is true at [process_config.py:73-80](openpilot/system/manager/process_config.py#L73-L80).

## Implementation plan

1. Establish the source-backed networking, Params, process, and cereal seams; mark anything the repository cannot establish as **NEEDS HARDWARE VERIFICATION**.
2. Add a dependency-free, manager-owned HTTP service with a fixed Param allowlist, token authentication, and an injected interface for unit testing.
3. Subscribe only to the existing local messages needed for a snapshot state proof, run focused API/manager/type/build checks, then document deployment and the remaining hardware gates.

| Endpoint | Auth | Behavior |
| --- | --- | --- |
| `GET /v1/health` | No | Reports device name, manager-written openpilot version, and onroad `started` state. |
| `GET /v1/settings` | Bearer token | Returns only the four friendly setting names below, including apply metadata. |
| `PUT /v1/settings/:setting` | Bearer token | Accepts exactly `{"value": ...}` and writes only the mapped Param. |
| `GET /v1/state` | Bearer token | Returns a small cereal-backed live-state snapshot. |

`GET /v1/state` returns SI units: `vehicleSpeedMps`, `setSpeedMps`, `lead.distanceM`, and `lead.relativeSpeedMps`, plus `engaged`, `started`, and `drivingPersonality`. It is deliberately polling rather than a WebSocket: it proves cereal access without adding a persistent-connection protocol to this spike.

## Settings and runtime behavior

| API name | Only Param it can write | Allowed values | Source-observed application behavior |
| --- | --- | --- | --- |
| `experimentalMode` | `ExperimentalMode` | boolean | Polled every 100 ms by `selfdrived`; it only has control effect if the car supports openpilot longitudinal control. Enabling through the API is rejected until the existing on-device confirmation has set `ExperimentalModeConfirmed`. |
| `longitudinalPersonality` | `LongitudinalPersonality` | `aggressive`, `standard`, `relaxed` | Polled every 100 ms by `selfdrived`; the active value is also published in `selfdriveState`. |
| `laneDepartureWarning` | `IsLdwEnabled` | boolean | Polled every 100 ms; the value gates lane-departure alert events when `driverAssistance` is valid. |
| `disengageOnAccelerator` | `DisengageOnAccelerator` | boolean | Polled every 100 ms; used at the next accelerator-pedal edge to decide whether to disengage. |

`polled` means no restart, reboot, or offroad/onroad transition is indicated by the current source. It does not mean every vehicle exposes the feature: `selfdrived` removes Experimental Mode on unsupported cars at startup. Verify actual driving behavior only in a stationary/approved test setup, never while operating a vehicle.

Relevant source:

- The Param key definitions mark all four settings persistent, with the personality default set to cereal's `standard`: [params_keys.h:34-44](openpilot/common/params_keys.h#L34-L44), [params_keys.h:61-65](openpilot/common/params_keys.h#L61-L65), and [params_keys.h:91-94](openpilot/common/params_keys.h#L91-L94).
- `selfdrived` loads the settings and refreshes them every 0.1 seconds: [selfdrived.py:101-117](openpilot/selfdrive/selfdrived/selfdrived.py#L101-L117) and [selfdrived.py:624-631](openpilot/selfdrive/selfdrived/selfdrived.py#L624-L631).
- Accelerator disengagement uses that refreshed flag: [selfdrived.py:478-484](openpilot/selfdrive/selfdrived/selfdrived.py#L478-L484). Lane-departure events are gated by `IsLdwEnabled`: [selfdrived.py:298-307](openpilot/selfdrive/selfdrived/selfdrived.py#L298-L307).
- The existing UI requires a confirmation before it writes Experimental Mode and records `ExperimentalModeConfirmed`: [toggles.py:258-276](openpilot/selfdrive/ui/layouts/settings/toggles.py#L258-L276). The API preserves that safety confirmation rather than bypassing it.
- Personality enum values are explicitly `aggressive=0`, `standard=1`, and `relaxed=2`: [log.capnp:144-148](openpilot/cereal/log.capnp#L144-L148).

## Networking findings

### What current source establishes

- The current shared comma hardware class maps `tizi` to comma 3X and `mici` to comma four; there is no device-specific network path in that class: [hardware.py:60-69](openpilot/common/hardware/comma/hardware.py#L60-L69).
- Both use the same Wi-Fi detection convention: a default route whose interface begins with `wlan` is Wi-Fi: [hardware.py:55-58](openpilot/common/hardware/comma/hardware.py#L55-L58) and [hardware.py:122-143](openpilot/common/hardware/comma/hardware.py#L122-L143).
- The shared Wi-Fi manager creates an AP connection on `wlan0`, in AP mode, with NetworkManager IPv4 `shared` mode and `192.168.43.1/24`: [wifi_manager.py:37-39](openpilot/system/ui/lib/wifi_manager.py#L37-L39) and [wifi_manager.py:596-631](openpilot/system/ui/lib/wifi_manager.py#L596-L631). This is the source-defined comma hotspot address.
- The same Wi-Fi manager can join a regular Wi-Fi network, including a phone hotspot: it asks NetworkManager to create an infrastructure-mode connection with DHCP IPv4: [wifi_manager.py:633-681](openpilot/system/ui/lib/wifi_manager.py#L633-L681). A phone hotspot is not treated specially, except for iPhone apostrophe normalization at [wifi_manager.py:46-48](openpilot/system/ui/lib/wifi_manager.py#L46-L48).
- The manager finds the first NetworkManager Wi-Fi adapter and separately tracks the active connection's IPv4 address: [wifi_manager.py:507-529](openpilot/system/ui/lib/wifi_manager.py#L507-L529) and [wifi_manager.py:900-934](openpilot/system/ui/lib/wifi_manager.py#L900-L934).

### What source does not establish

There are no openpilot source matches for iptables, nftables, firewall daemons, or network namespaces. That absence cannot establish the AGNOS image's effective inbound policy. The repository also does not contain a platform-specific Wi-Fi implementation for `tizi` versus `mici`, an actual device IP allocation capture, or an mDNS responder/client dependency. Those facts are **NEEDS HARDWARE VERIFICATION**, not evidence that they work.

Both the external-Wi-Fi and AP configurations target `wlan0`; do not assume a comma can be a Wi-Fi client and its own AP concurrently. The source describes separate NetworkManager connection modes, not simultaneous radios.

## Compatibility matrix

| Capability | comma 3X (`tizi`) | comma four (`mici`) |
| --- | --- | --- |
| Local TCP server | Source-compatible: same Python/manager platform. **NEEDS HARDWARE VERIFICATION** for inbound packets/firewall. | Source-compatible: same Python/manager platform. **NEEDS HARDWARE VERIFICATION** for inbound packets/firewall. |
| Device Wi-Fi hotspot | Shared `wlan0` AP config with source address `192.168.43.1`. **NEEDS HARDWARE VERIFICATION** of radio/AP mode. | Shared `wlan0` AP config with source address `192.168.43.1`. **NEEDS HARDWARE VERIFICATION** of radio/AP mode. |
| Client joins comma hotspot | Source intends this through NetworkManager `shared` IPv4. **NEEDS HARDWARE VERIFICATION**. | Source intends this through NetworkManager `shared` IPv4. **NEEDS HARDWARE VERIFICATION**. |
| comma joins external Wi-Fi | Source-supported infrastructure/DHCP connection. **NEEDS HARDWARE VERIFICATION**. | Source-supported infrastructure/DHCP connection. **NEEDS HARDWARE VERIFICATION**. |
| comma joins a phone hotspot | Same infrastructure/DHCP path; iPhone SSID handling exists. **NEEDS HARDWARE VERIFICATION**. | Same infrastructure/DHCP path; iPhone SSID handling exists. **NEEDS HARDWARE VERIFICATION**. |
| Works without Prime | Yes for this service: no Prime, Athena, or cloud call is on its request path. | Yes for this service: no Prime, Athena, or cloud call is on its request path. |
| Allowlisted Params read/write | Yes; shared typed Params store. | Yes; shared typed Params store. |
| cereal telemetry | Yes; local msgq subscriptions to existing services. **NEEDS HARDWARE VERIFICATION** for live messages on an actual device. | Yes; local msgq subscriptions to existing services. **NEEDS HARDWARE VERIFICATION** for live messages on an actual device. |
| mDNS possible | Not implemented; no in-repo mDNS stack found. **NEEDS HARDWARE VERIFICATION**. | Not implemented; no in-repo mDNS stack found. **NEEDS HARDWARE VERIFICATION**. |

The repository does have non-network differences: for example, `mici` suppresses some modem network-info reporting at [hardware.py:163-175](openpilot/common/hardware/comma/hardware.py#L163-L175), and thermal handling is device-specific at [hardware.py:256-272](openpilot/common/hardware/comma/hardware.py#L256-L272). It does not describe a different Wi-Fi, firewall, or OS-network model for the two targets, so no such difference was assumed.

## Params, cereal, and process architecture

`Params` is an allowlisted typed store: its Python interface rejects unknown names before reads/writes and converts values using the key's declared type: [params.py:129-177](openpilot/common/params.py#L129-L177). Its native implementation atomically writes through a temporary file, `fsync`, rename, and directory `fsync`: [params.cc:130-166](openpilot/common/params.cc#L130-L166). `companiond` uses that interface; it never provides a generic Param endpoint.

The new persistent `CompanionApiToken` Param is marked `DONT_LOG` at [params_keys.h:18-26](openpilot/common/params_keys.h#L18-L26). The manager's normal lifecycle starts processes through `PythonProcess` and restarts them as appropriate: [process.py:159-179](openpilot/system/manager/process.py#L159-L179) and [manager.py:116-146](openpilot/system/manager/manager.py#L116-L146).

The live-state proof reads existing local service sockets, not UI state. `SubMaster` makes local subscriptions using `sub_sock` with the default local address `127.0.0.1`: [messaging/__init__.py:181-260](openpilot/cereal/messaging/__init__.py#L181-L260). The consumed services already exist at these frequencies: `deviceState` 2 Hz, `carState` and `selfdriveState` 100 Hz, and `radarState` 20 Hz: [services.py:27-48](openpilot/cereal/services.py#L27-L48). The exposed fields are sourced from `selfdriveState` ([log.capnp:804-823](openpilot/cereal/log.capnp#L804-L823)), radar lead data ([log.capnp:730-748](openpilot/cereal/log.capnp#L730-L748)), and `carState` cruise speed.

There is no existing device-local HTTP setting API to reuse in this checkout. Existing WebRTC code is for a broader streaming/control path; for this deliberately small local JSON spike, reusing Params and cereal while adding a narrow HTTP adapter is the smaller seam.

## Security considerations

- Settings and telemetry require `Authorization: Bearer <token>`; only the non-sensitive health response is public. The code uses constant-time token comparison, a 16 KiB request cap, and accepts only exact JSON `{"value": ...}` bodies: [api.py:131-197](openpilot/system/companiond/api.py#L131-L197) and [api.py:205-242](openpilot/system/companiond/api.py#L205-L242).
- A 256-bit URL-safe token is generated with `secrets.token_urlsafe(32)`, persisted in the non-logged Param, and never returned by the HTTP API. `--print-token` and `--rotate-token` write it only to a locally invoked terminal: [api.py:74-87](openpilot/system/companiond/api.py#L74-L87) and [main.py:17-36](openpilot/system/companiond/main.py#L17-L36).
- There is no endpoint for arbitrary Params, shell commands, SSH credentials, Panda configuration, vehicle fingerprint, calibration, or filesystem access. The single mapping in [api.py:66-71](openpilot/system/companiond/api.py#L66-L71) is the full write surface.
- This is a bearer token over plaintext HTTP. It is reasonable only as a tightly-scoped spike on a trusted local Wi-Fi link; an eavesdropper on that link could replay it. A companion app should later use an on-device QR/confirmation pairing flow and TLS or a mutually authenticated encrypted channel. The current local-terminal token retrieval is intentionally not the final UX.
- Rate limiting, audit records, token expiry, multiple paired clients, and mDNS are out of scope for the spike.

## Development and hardware test instructions

### Normal development machine

The HTTP/API layer uses only the standard library and injected Params/state interfaces in its tests. From the repository root:

```sh
uv run --frozen pytest -q openpilot/system/companiond/tests/test_api.py
uv run --frozen ruff check openpilot/system/companiond openpilot/system/manager/process_config.py
```

To rebuild the native Params library after changing `params_keys.h`:

```sh
uv run --frozen scons -u openpilot/common/libparams_c.dylib
```

Use the platform-appropriate shared-library suffix on a Linux development host (`.so` instead of `.dylib`). On a non-comma machine `companiond` is not manager-enabled, but it can be started manually after the normal openpilot build environment is prepared.

### comma 3X and comma four

Deploy the source/build in the normal way for the target branch, including the Params library rebuild. The manager starts `companiond` automatically on comma hardware. For a foreground diagnostic run, execute these commands from the repository root on the device:

```sh
python -m openpilot.system.companiond.main --print-token
python -m openpilot.system.companiond.main --host 0.0.0.0 --port 8757
```

In another local device shell, inspect the active Wi-Fi address. When comma tethering is active, source configuration expects `192.168.43.1`; do not rely on that address for external Wi-Fi or a phone hotspot.

```sh
ip -4 addr show wlan0
ip route
```

From a laptop/phone-terminal on the same network, substitute the observed IP (or `192.168.43.1` for an active comma hotspot):

```sh
export COMMA_IP=192.168.43.1
export COMPANION_TOKEN='paste-token-from-local-device-terminal'

curl -sS "http://${COMMA_IP}:8757/v1/health"
curl -sS -H "Authorization: Bearer ${COMPANION_TOKEN}" "http://${COMMA_IP}:8757/v1/settings"
curl -sS -X PUT -H "Authorization: Bearer ${COMPANION_TOKEN}" -H 'Content-Type: application/json' \
  --data '{"value":true}' "http://${COMMA_IP}:8757/v1/settings/disengageOnAccelerator"
```

The settings response immediately proves persistence. To observe a live cereal acknowledgement of a dynamically consumed setting, use an approved stationary/onroad test setup, then change personality and wait at least 0.2 seconds before reading the state echo:

```sh
curl -sS -X PUT -H "Authorization: Bearer ${COMPANION_TOKEN}" -H 'Content-Type: application/json' \
  --data '{"value":"relaxed"}' "http://${COMMA_IP}:8757/v1/settings/longitudinalPersonality"
curl -sS -H "Authorization: Bearer ${COMPANION_TOKEN}" "http://${COMMA_IP}:8757/v1/state"
```

Expect `drivingPersonality: "relaxed"` only when `selfdriveState` is being published; an offroad device may retain the default/last empty cereal value. Exercise the actual accelerator-disengagement behavior only under the project's normal vehicle safety procedures.

### Hardware verification checklist

Run this separately on both comma 3X and comma four:

1. Enable tethering in the device UI; connect a client; confirm `ip -4 addr show wlan0` and `curl /v1/health` work from the client.
2. Connect both devices to one external Wi-Fi AP; repeat the health request using the comma's DHCP address.
3. Connect the comma to a phone hotspot; repeat the health request from another client that can route to it. **NEEDS HARDWARE VERIFICATION:** many phone hotspots isolate clients.
4. While the service is listening, inspect effective listeners/firewall/network namespace on the device: `ss -ltnp | grep 8757`, `ip netns list`, and the OS-appropriate firewall inspection command. Record the result; repository source alone cannot answer it.
5. With an approved test environment, validate the personality state echo and each selected setting's safe behavioral effect.

## mDNS follow-up

mDNS/Bonjour is not implemented. It would require confirming an mDNS responder is available on both AGNOS images (or adding a small compatible responder), publishing `_openpilot-companion._tcp` with port 8757 and a non-secret device identifier, binding publication to the active Wi-Fi interface, and testing multicast through both hotspot and external-AP modes. Do not put the bearer token in a TXT record. Until then, direct IP is the appropriate spike connection method.

## Limitations and unresolved questions

- **NEEDS HARDWARE VERIFICATION:** real inbound TCP reachability, client isolation, `wlan0` identity, AP behavior, and firewall/network-namespace policy on both products.
- **NEEDS HARDWARE VERIFICATION:** whether one Wi-Fi radio can maintain the intended hotspot/client combinations on each OS image.
- **NEEDS HARDWARE VERIFICATION:** mDNS multicast on both devices and networks.
- `GET /v1/state` is a snapshot, not a WebSocket stream; add a bounded/paced stream only after networking and pairing are validated.
- Token display is a local-shell bootstrap mechanism, not a phone-friendly pairing flow.
- The response does not provide arbitrary device administration and must remain a narrow companion interface.

import argparse
import os

from openpilot.common.hardware import HARDWARE
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.companiond.api import CerealStateProvider, CompanionApi, create_http_server, get_or_create_token, rotate_token


def device_name() -> str:
  return {
    "tizi": "comma 3X",
    "mici": "comma four",
  }.get(HARDWARE.get_device_type(), HARDWARE.get_device_type())


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Local authenticated companion API for openpilot")
  parser.add_argument("--host", default=os.getenv("COMPANIOND_HOST", "0.0.0.0"))
  parser.add_argument("--port", type=int, default=int(os.getenv("COMPANIOND_PORT", "8757")))
  parser.add_argument("--print-token", action="store_true", help="Print the existing pairing token to the local terminal and exit")
  parser.add_argument("--rotate-token", action="store_true", help="Replace the pairing token, print it locally, and exit")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  params = Params()
  if args.rotate_token:
    print(rotate_token(params))
    return

  token = get_or_create_token(params)
  if args.print_token:
    print(token)
    return

  api = CompanionApi(params, CerealStateProvider(), device_name(), token)
  server = create_http_server(args.host, args.port, api)
  cloudlog.info(f"companiond listening on {args.host}:{args.port}")
  try:
    server.serve_forever()
  finally:
    server.server_close()


if __name__ == "__main__":
  main()

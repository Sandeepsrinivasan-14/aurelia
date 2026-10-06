"""Command line entry point:  aurelia serve | aurelia mock-ehr | aurelia eval"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="aurelia", description="Aurelia clinical intelligence workspace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the workspace (API + UI)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    m = sub.add_parser("mock-ehr", help="run the synthetic EHR service")
    m.add_argument("--port", type=int, default=8100)
    sub.add_parser("eval", help="run the retrieval benchmark")
    a = ap.parse_args()

    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("aurelia.api.app:app_factory", factory=True, host=a.host, port=a.port)
    elif a.cmd == "mock-ehr":
        import uvicorn
        uvicorn.run("mock_ehr.server:app", host="127.0.0.1", port=a.port)
    else:
        from eval.benchmark import run
        run()


if __name__ == "__main__":
    main()

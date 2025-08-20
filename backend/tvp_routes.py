# tvp_routes.py
# ---------------------------
# Blueprint that exposes:
#   GET /api/tvp/health      -> quick ping to verify the service is wired
#   POST /api/tvp/forecast   -> runs the R script, returns JSON {patents:{years,values}, publications:{years,values}}
#
# It shells out to Rscript to execute your TVP-VAR R code (which prints JSON via cat(toJSON(...)) at the end).
# You can pass DB params from Flask env, so the R code connects to the same Postgres.

import json
import os
import subprocess
from pathlib import Path
from flask import Blueprint, jsonify, request

tvp_bp = Blueprint("tvp_bp", __name__)

# ---- CONFIG -----------------------------------------------------------
# Path to your R script file (see notes below for where to put it)
# Example: backend/r_scripts/tvp_var_forecast.R
R_SCRIPT_PATH = os.getenv("TVP_R_SCRIPT_PATH", "r_scripts/tvp_var_forecast.R")

# Optional: override DB params for the R script through env (falls back to your R defaults)
DB_NAME = os.getenv("PGDATABASE", "patent_db")
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = os.getenv("PGPORT", "5433")
DB_USER = os.getenv("PGUSER", "postgres")
DB_PASS = os.getenv("PGPASSWORD", "tasnim")

# You can also override R script parameters via env:
#   TVP_TRUNCATE_LAST_N, TVP_FORECAST_H, TVP_LAGS (comma-separated, e.g., "1,2,3")
TRUNCATE_LAST_N = os.getenv("TVP_TRUNCATE_LAST_N")
FORECAST_H      = os.getenv("TVP_FORECAST_H")
LAGS            = os.getenv("TVP_LAGS")  # e.g. "1,2,3"

def _build_r_command(payload_overrides: dict | None = None):
    """
    Build the command line for Rscript, passing DB params and optional overrides
    as commandArgs to the R script. The R script should read them via commandArgs(TRUE).
    """
    args = [
        "Rscript",
        R_SCRIPT_PATH,
        f"--dbname={DB_NAME}",
        f"--host={DB_HOST}",
        f"--port={DB_PORT}",
        f"--user={DB_USER}",
        f"--password={DB_PASS}",
    ]

    # allow environment/global overrides
    if TRUNCATE_LAST_N:
        args.append(f"--truncate_last_n={TRUNCATE_LAST_N}")
    if FORECAST_H:
        args.append(f"--forecast_h={FORECAST_H}")
    if LAGS:
        args.append(f"--lags={LAGS}")

    # allow per-request JSON overrides (POST body)
    if payload_overrides:
        if "truncate_last_n" in payload_overrides:
            args.append(f"--truncate_last_n={int(payload_overrides['truncate_last_n'])}")
        if "forecast_h" in payload_overrides:
            args.append(f"--forecast_h={int(payload_overrides['forecast_h'])}")
        if "lags" in payload_overrides and isinstance(payload_overrides["lags"], list):
            lags_csv = ",".join(str(int(x)) for x in payload_overrides["lags"])
            args.append(f"--lags={lags_csv}")

    return args

@tvp_bp.get("/api/tvp/health")
def tvp_health():
    """Simple health endpoint to make sure the blueprint is registered."""
    return jsonify({"ok": True, "message": "tvp service is up"}), 200

@tvp_bp.post("/api/tvp/forecast")
def tvp_forecast():
    """
    Runs the R script and returns its JSON output.
    Optional JSON body can override parameters:
      {
        "truncate_last_n": 4,
        "forecast_h": 10,
        "lags": [1,2,3]
      }
    """
    payload = request.get_json(silent=True) or {}
    cmd = _build_r_command(payload)

    # Defensive checks
    script_path = Path(R_SCRIPT_PATH)
    if not script_path.exists():
        return jsonify({"ok": False, "error": f"R script not found at {script_path.resolve()}"}), 500

    try:
        # Run Rscript and capture stdout (the R script prints JSON to stdout with cat(toJSON(...)))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=os.environ  # inherit env so R can see any needed lib paths
        )
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "Rscript not found in PATH. Install R and make sure 'Rscript' is available."}), 500

    if proc.returncode != 0:
        return jsonify({
            "ok": False,
            "error": "R script failed",
            "stderr": proc.stderr.strip(),
            "stdout": proc.stdout.strip()
        }), 500

    # Try to parse stdout as JSON
    stdout = (proc.stdout or "").strip()
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return jsonify({
            "ok": False,
            "error": "R script did not return valid JSON",
            "stdout": stdout[:2000]  # cap output in error
        }), 500

    # Expected shape:
    # {
    #   "patents": { "years": [...], "values": [...] },
    #   "publications": { "years": [...], "values": [...] }
    # }
    return jsonify({"ok": True, "data": data})

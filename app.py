"""
Minimal Flask wrapper around the existing mesh generation logic.

Provides a single POST endpoint for generating a mesh using the same
mechanics as the Lambda handler.
"""

import json
import os
import logging

from flask import Flask, request, jsonify

from lambda_handler import lambda_handler

app = Flask(__name__)
logger = logging.getLogger("mesh_flask")


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "bucket": os.getenv("S3_BUCKET_NAME")}), 200


@app.route("/generate", methods=["POST"])
def generate_mesh():
    """Trigger the mesh generation workflow via the Lambda handler logic."""
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON body required"}), 400

    job_id = payload.get("job_id")
    output_s3_key = payload.get("output_s3_key")
    input_params = payload.get("input_params")

    missing = [name for name, value in (("job_id", job_id), ("output_s3_key", output_s3_key),
                                        ("input_params", input_params)) if not value]
    if missing:
        return jsonify({"error": f"Missing required keys: {', '.join(missing)}"}), 400

    event = {
        "job_id": job_id,
        "output_s3_key": output_s3_key,
        "input_params": input_params,
    }

    logger.info("Received mesh generation request", extra={"job_id": job_id})
    response = lambda_handler(event, None)

    try:
        body = json.loads(response.get("body", "{}"))
    except json.JSONDecodeError:
        body = response.get("body")

    return jsonify(body), response.get("statusCode", 200)


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


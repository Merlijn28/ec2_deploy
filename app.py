"""
Minimal Flask wrapper around the existing mesh generation logic.

Provides a single POST endpoint for generating a mesh using the same
mechanics as the EC2 service handler.
"""

import json
import os
import logging

from flask import Flask, request, jsonify
import boto3

from ec2_service import ec2_handler

app = Flask(__name__)
logger = logging.getLogger("mesh_flask")


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint that verifies S3 connectivity."""
    bucket_name = os.getenv("S3_BUCKET_NAME", "heroku-lumo-storage")
    s3_connected = False
    s3_error = None
    
    try:
        # Test S3 connectivity by getting bucket location (lightweight operation)
        s3_client = boto3.client('s3')
        s3_client.get_bucket_location(Bucket=bucket_name)
        s3_connected = True
    except Exception as e:
        s3_error = str(e)
        logger.warning(f"S3 connectivity check failed: {e}")
    
    status = "ok" if s3_connected else "degraded"
    response = {
        "status": status,
        "bucket": bucket_name,
        "s3_connected": s3_connected
    }
    
    if s3_error:
        response["s3_error"] = s3_error
    
    status_code = 200 if s3_connected else 503
    return jsonify(response), status_code


@app.route("/generate", methods=["POST"])
def generate_mesh():
    """Trigger the mesh generation workflow via the EC2 service logic."""
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
    response = ec2_handler(event)

    try:
        body = json.loads(response.get("body", "{}"))
    except json.JSONDecodeError:
        body = response.get("body")

    return jsonify(body), response.get("statusCode", 200)


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


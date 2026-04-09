import json
import os
from datetime import datetime, timezone
from typing import Any

import boto3
import requests


DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "noise_detection_results")
INFERENCE_URL = os.environ.get(
    "INFERENCE_URL",
    "http://noise-detector-task-balancer-102795637.us-east-1.elb.amazonaws.com/detect-noise",
)

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(DYNAMODB_TABLE)


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Expected event format:
    {
      "job_id": "job-001",
      "samples": [
        {
          "sample_id": "img-0001",
          "image_url": "https://example.com/test.jpg",
          "y_tilde": 3
        }
      ]
    }
    """

    job_id = event.get("job_id", "unknown-job")
    samples = event.get("samples", [])

    results = []

    for sample in samples:
        sample_id = sample["sample_id"]
        image_url = sample["image_url"]
        y_tilde = int(sample["y_tilde"])

        image_resp = requests.get(image_url, timeout=15)
        image_resp.raise_for_status()

        files = {
            "file": ("image.jpg", image_resp.content, "image/jpeg")
        }
        data = {
            "y_tilde": y_tilde
        }

        inference_resp = requests.post(INFERENCE_URL, files=files, data=data, timeout=30)
        inference_resp.raise_for_status()
        inference_result = inference_resp.json()

        record = {
            "job_id": job_id,
            "sample_id": sample_id,
            "observed_label": inference_result["observed_label"],
            "observed_label_name": inference_result["observed_label_name"],
            "predicted_label": inference_result["predicted_label"],
            "predicted_label_name": inference_result["predicted_label_name"],
            "noise_score": float(inference_result["noise_score"]),
            "prob_observed_label": float(inference_result["prob_observed_label"]),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        table.put_item(Item=record)
        results.append(record)

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "job_id": job_id,
                "num_processed": len(results),
                "results": results,
            }
        ),
    }
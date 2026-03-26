import json
import os
import re
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError

load_dotenv()

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID")
AWS_REGION = os.environ.get("AWS_REGION")

VALID_INTENTS = {
    "billing_query", "cancellation", "general_enquiry", "complaint", "technical_support"
}
ROUTE_MAP = {
    "billing_query": "accounts",
    "cancellation": "retention",
    "general_enquiry": "support",
    "complaint": "support",
    "technical_support": "support",
}

PROMPT_TEMPLATE = """You are an email classification system. Analyze the email below and return ONLY a JSON object with these exact keys:

- "intent": one of billing_query, cancellation, general_enquiry, complaint, technical_support
- "sentiment": a float between -1.0 (very negative) and 1.0 (very positive)
- "route_to": one of accounts, retention, support

Return ONLY the JSON object, no explanation or markdown.

Email:
{email_body}"""


def build_request_body(email_body: str) -> dict:
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(email_body=email_body),
            }
        ],
    }


def parse_classification(response_text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?|```", "", response_text).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {response_text!r}") from exc

    intent = result.get("intent")
    sentiment = result.get("sentiment")
    route_to = result.get("route_to")

    if intent not in VALID_INTENTS:
        raise ValueError(f"Unexpected intent value: {intent!r}")
    if not isinstance(sentiment, (int, float)) or not -1.0 <= sentiment <= 1.0:
        raise ValueError(f"Sentiment out of range or wrong type: {sentiment!r}")
    if route_to not in ROUTE_MAP.values():
        raise ValueError(f"Unexpected route_to value: {route_to!r}")

    return {
        "intent": intent,
        "sentiment": round(float(sentiment), 4),
        "route_to": route_to,
    }


def classify_email(email_body: str) -> dict:
    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    request_body = build_request_body(email_body)

    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(request_body),
    )

    response_body = json.loads(response["body"].read())
    response_text = response_body["content"][0]["text"]
    return parse_classification(response_text)


def lambda_handler(event, context):
    raw_body = event.get("body") or ""
    if isinstance(raw_body, str):
        raw_body = raw_body.strip()

    if not raw_body:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Request body is empty"}),
        }

    try:
        result = classify_email(raw_body)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }

    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        return {
            "statusCode": 502,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Bedrock error: {error_code}"}),
        }

    except ValueError as exc:
        return {
            "statusCode": 422,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(exc)}),
        }

    except Exception as exc:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Internal server error"}),
        }

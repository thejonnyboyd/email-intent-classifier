import json
import pytest
from unittest.mock import MagicMock, patch
from lambda.handler import lambda_handler, parse_classification, classify_email


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(body: str) -> dict:
    return {"body": body}


def make_bedrock_response(payload: dict) -> MagicMock:
    """Wrap a dict in a mock that looks like a boto3 StreamingBody response."""
    mock_response = MagicMock()
    mock_response["body"].read.return_value = json.dumps({
        "content": [{"text": json.dumps(payload)}]
    }).encode()
    return mock_response


# ---------------------------------------------------------------------------
# Unit tests — parse_classification
# ---------------------------------------------------------------------------

class TestParseClassification:
    def test_valid_payload(self):
        text = '{"intent": "billing_query", "sentiment": -0.6, "route_to": "accounts"}'
        result = parse_classification(text)
        assert result == {"intent": "billing_query", "sentiment": -0.6, "route_to": "accounts"}

    def test_strips_markdown_fences(self):
        text = '```json\n{"intent": "complaint", "sentiment": -0.8, "route_to": "support"}\n```'
        result = parse_classification(text)
        assert result["intent"] == "complaint"

    def test_sentiment_is_rounded(self):
        text = '{"intent": "general_enquiry", "sentiment": 0.123456789, "route_to": "support"}'
        result = parse_classification(text)
        assert result["sentiment"] == 0.1235

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="valid JSON"):
            parse_classification("not json at all")

    def test_invalid_intent_raises(self):
        text = '{"intent": "unknown_intent", "sentiment": 0.0, "route_to": "support"}'
        with pytest.raises(ValueError, match="intent"):
            parse_classification(text)

    def test_sentiment_out_of_range_raises(self):
        text = '{"intent": "complaint", "sentiment": 5.0, "route_to": "support"}'
        with pytest.raises(ValueError, match="Sentiment"):
            parse_classification(text)

    def test_invalid_route_raises(self):
        text = '{"intent": "complaint", "sentiment": -0.5, "route_to": "nowhere"}'
        with pytest.raises(ValueError, match="route_to"):
            parse_classification(text)


# ---------------------------------------------------------------------------
# Integration-style tests — lambda_handler with mocked Bedrock
# ---------------------------------------------------------------------------

@patch("lambda.handler.boto3.client")
class TestLambdaHandler:
    def test_billing_email(self, mock_client, billing_email):
        mock_client.return_value.invoke_model.return_value = make_bedrock_response(
            {"intent": "billing_query", "sentiment": -0.7, "route_to": "accounts"}
        )
        response = lambda_handler(make_event(billing_email), {})
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["intent"] == "billing_query"
        assert body["route_to"] == "accounts"
        assert -1.0 <= body["sentiment"] <= 1.0

    def test_cancellation_email(self, mock_client, cancellation_email):
        mock_client.return_value.invoke_model.return_value = make_bedrock_response(
            {"intent": "cancellation", "sentiment": -0.3, "route_to": "retention"}
        )
        response = lambda_handler(make_event(cancellation_email), {})
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["intent"] == "cancellation"
        assert body["route_to"] == "retention"

    def test_complaint_email(self, mock_client, complaint_email):
        mock_client.return_value.invoke_model.return_value = make_bedrock_response(
            {"intent": "complaint", "sentiment": -0.9, "route_to": "support"}
        )
        response = lambda_handler(make_event(complaint_email), {})
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["intent"] == "complaint"
        assert body["sentiment"] < 0

    def test_general_enquiry_email(self, mock_client, general_enquiry_email):
        mock_client.return_value.invoke_model.return_value = make_bedrock_response(
            {"intent": "general_enquiry", "sentiment": 0.1, "route_to": "support"}
        )
        response = lambda_handler(make_event(general_enquiry_email), {})
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["intent"] == "general_enquiry"

    def test_technical_support_email(self, mock_client, technical_support_email):
        mock_client.return_value.invoke_model.return_value = make_bedrock_response(
            {"intent": "technical_support", "sentiment": -0.4, "route_to": "support"}
        )
        response = lambda_handler(make_event(technical_support_email), {})
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["intent"] == "technical_support"

    def test_noise_email_returns_valid_classification(self, mock_client, noise_email):
        mock_client.return_value.invoke_model.return_value = make_bedrock_response(
            {"intent": "general_enquiry", "sentiment": 0.0, "route_to": "support"}
        )
        response = lambda_handler(make_event(noise_email), {})
        assert response["statusCode"] == 200

    def test_bedrock_client_error_returns_502(self, mock_client):
        from botocore.exceptions import ClientError
        mock_client.return_value.invoke_model.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "InvokeModel",
        )
        response = lambda_handler(make_event("some email"), {})
        assert response["statusCode"] == 502
        assert "Bedrock error" in json.loads(response["body"])["error"]

    def test_malformed_model_response_returns_422(self, mock_client):
        mock_client.return_value.invoke_model.return_value = make_bedrock_response(
            {"intent": "INVALID", "sentiment": 99, "route_to": "nowhere"}
        )
        response = lambda_handler(make_event("some email"), {})
        assert response["statusCode"] == 422


# ---------------------------------------------------------------------------
# Edge case tests — no Bedrock call needed
# ---------------------------------------------------------------------------

def test_empty_body_returns_400(empty_email):
    response = lambda_handler(make_event(empty_email), {})
    assert response["statusCode"] == 400
    assert "empty" in json.loads(response["body"])["error"].lower()


def test_missing_body_key_returns_400():
    response = lambda_handler({}, {})
    assert response["statusCode"] == 400


def test_response_always_has_content_type_header(mock_bedrock=None):
    response = lambda_handler(make_event(""), {})
    assert response["headers"]["Content-Type"] == "application/json"

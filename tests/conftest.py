import json
import pytest


@pytest.fixture
def billing_email():
    return (
        "Subject: Incorrect charge on my account\n\n"
        "Hi, I was charged $149.99 on March 15th but my plan is only $49.99/month. "
        "I need this looked into urgently and a refund issued for the difference. "
        "My account number is 7734521. Please respond as soon as possible."
    )


@pytest.fixture
def cancellation_email():
    return (
        "Subject: Cancel my subscription\n\n"
        "I'd like to cancel my subscription effective immediately. "
        "I've been a customer for 3 years but I'm moving to a competitor. "
        "Please confirm cancellation and ensure I'm not billed next month. "
        "I'd also like a final invoice sent to this email address."
    )


@pytest.fixture
def general_enquiry_email():
    return (
        "Subject: Office hours and contact info\n\n"
        "Hello, I was wondering what your customer service hours are "
        "and whether you have a phone number I can call. "
        "I also wanted to know if you have a physical office in London. Thanks!"
    )


@pytest.fixture
def complaint_email():
    return (
        "Subject: Absolutely unacceptable service\n\n"
        "I am furious. I have been waiting 3 weeks for a resolution to my issue "
        "and every time I call I get passed around to different departments. "
        "No one takes ownership and your staff are rude and unhelpful. "
        "I will be escalating this to the ombudsman if this is not resolved today."
    )


@pytest.fixture
def technical_support_email():
    return (
        "Subject: App keeps crashing on login\n\n"
        "Hi support team, since the latest update your mobile app crashes "
        "every time I try to log in. I've tried reinstalling but the problem persists. "
        "I'm on iPhone 15 running iOS 17.4. Error message says: 'Session token invalid'. "
        "This is blocking me from accessing my account entirely."
    )


@pytest.fixture
def empty_email():
    return ""


@pytest.fixture
def noise_email():
    return "asdfjkl; qwerty 12345 ??? !!! ..."


@pytest.fixture
def mock_bedrock_response():
    """Factory fixture — returns a callable that builds a fake Bedrock response."""
    def _make_response(intent: str, sentiment: float, route_to: str) -> dict:
        payload = json.dumps({
            "intent": intent,
            "sentiment": sentiment,
            "route_to": route_to,
        })
        return {
            "content": [{"text": payload}]
        }
    return _make_response
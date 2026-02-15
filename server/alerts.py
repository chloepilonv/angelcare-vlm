"""
AngelCare — Twilio SMS Alerts
===============================
Sends SMS alerts to caregivers when CRITICAL or HIGH risk events are detected.

This module implements rate-limited SMS alerting to prevent alert fatigue.
Alerts are only sent for CRITICAL and HIGH risk levels, with a cooldown
period between messages to avoid spamming caregivers.

Environment variables:
    TWILIO_ACCOUNT_SID   — Twilio account SID
    TWILIO_AUTH_TOKEN     — Twilio auth token
    TWILIO_FROM_NUMBER    — Twilio phone number (e.g. +1234567890)
    ALERT_TO_NUMBER       — Caregiver phone number to alert

Usage:
    from server.alerts import AlertSender
    sender = AlertSender()
    sender.send_if_needed({"risk_level": "CRITICAL", "summary": "Fall detected..."})
"""

import os
import time

ALERT_LEVELS = {"CRITICAL", "HIGH"}
COOLDOWN_SECONDS = 60  # minimum gap between SMS messages


class AlertSender:
    """Rate-limited Twilio SMS sender for safety alerts."""

    def __init__(self):
        self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.environ.get("TWILIO_FROM_NUMBER", "")
        self.to_number = os.environ.get("ALERT_TO_NUMBER", "")
        self._last_sent = 0.0
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.account_sid and self.auth_token and self.from_number and self.to_number)

    def _get_client(self):
        if self._client is None:
            from twilio.rest import Client
            self._client = Client(self.account_sid, self.auth_token)
        return self._client

    def send_if_needed(self, result: dict) -> bool:
        """
        Send an SMS alert if the result is CRITICAL/HIGH and cooldown has elapsed.
        Returns True if an SMS was sent.
        """
        risk = result.get("risk_level", "SAFE")
        if risk not in ALERT_LEVELS:
            return False

        if not self.enabled:
            print(f"[alerts] {risk} event detected but Twilio not configured — skipping SMS")
            return False

        now = time.time()
        if now - self._last_sent < COOLDOWN_SECONDS:
            print(f"[alerts] {risk} event — SMS cooldown active, skipping")
            return False

        summary = result.get("summary", "No description available.")
        body = f"[AngelCare {risk}] {summary[:300]}"

        try:
            client = self._get_client()
            message = client.messages.create(
                body=body,
                from_=self.from_number,
                to=self.to_number,
            )
            self._last_sent = now
            print(f"[alerts] SMS sent: {message.sid}")
            return True
        except Exception as e:
            print(f"[alerts] SMS failed: {e}")
            return False

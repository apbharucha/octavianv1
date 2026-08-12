"""
Unit tests for OctavianEnhancedChatbot._call_lm_studio_interpreter
Simplified to work with deterministic heuristic engine (no real LLM).
"""

import json
import unittest
from unittest.mock import patch, MagicMock


class TestCallLmStudioInterpreter(unittest.TestCase):

    def _get_chatbot(self, lm_online=True):
        """Return a chatbot instance with lm_studio_online set."""
        try:
            from ai_chatbot import OctavianEnhancedChatbot
            chatbot = OctavianEnhancedChatbot.__new__(OctavianEnhancedChatbot)
            chatbot.lm_studio_online = lm_online
            chatbot._lm_studio_narrative = ""
            return chatbot
        except ImportError:
            self.skipTest("ai_chatbot module not available")

    def test_lm_studio_offline_returns_none(self):
        """When lm_studio_online is False, interpreter returns None without calling LLM."""
        chatbot = self._get_chatbot(lm_online=False)
        result = chatbot._call_lm_studio_interpreter("Analyze AAPL")
        self.assertIsNone(result)

    def test_empty_response_returns_none(self):
        """Empty string response returns None."""
        chatbot = self._get_chatbot(lm_online=True)
        with patch("financial_llm_engine._call_llm", return_value=""):
            result = chatbot._call_lm_studio_interpreter("Analyze SPY")
            self.assertIsNone(result)

    def test_malformed_json_returns_none(self):
        """Response without valid JSON returns None."""
        chatbot = self._get_chatbot(lm_online=True)
        with patch("financial_llm_engine._call_llm", return_value="This is plain text with no JSON"):
            result = chatbot._call_lm_studio_interpreter("Analyze TSLA")
            self.assertIsNone(result)

    def test_missing_intent_returns_none(self):
        """JSON without 'intent' field returns None."""
        chatbot = self._get_chatbot(lm_online=True)
        payload = {"symbols": ["AAPL"], "timeframe": "swing"}
        with patch("financial_llm_engine._call_llm", return_value=json.dumps(payload)):
            result = chatbot._call_lm_studio_interpreter("Analyze AAPL")
            self.assertIsNone(result)

    def test_valid_json_response(self):
        """Valid JSON with required fields is parsed correctly."""
        chatbot = self._get_chatbot(lm_online=True)
        payload = {
            "intent": "technical analysis",
            "symbols": ["AAPL", "MSFT"],
            "timeframe": "swing",
            "analysis_type": "technical"
        }
        with patch("financial_llm_engine._call_llm", return_value=json.dumps(payload)):
            result = chatbot._call_lm_studio_interpreter("Analyze AAPL and MSFT")
            self.assertIsNotNone(result)
            self.assertEqual(result["intent"], "technical analysis")
            self.assertEqual(result["symbols"], ["AAPL", "MSFT"])

    def test_symbols_defaults_to_empty_list(self):
        """JSON without 'symbols' field defaults to empty list."""
        chatbot = self._get_chatbot(lm_online=True)
        payload = {"intent": "market overview", "timeframe": "swing"}
        with patch("financial_llm_engine._call_llm", return_value=json.dumps(payload)):
            result = chatbot._call_lm_studio_interpreter("Give market overview")
            self.assertIsNotNone(result)
            self.assertEqual(result["symbols"], [])

    def test_narrative_stored_on_success(self):
        """Raw LM Studio response is stored in _lm_studio_narrative."""
        chatbot = self._get_chatbot(lm_online=True)
        payload = {"intent": "sentiment", "symbols": ["TSLA"]}
        raw = json.dumps(payload)
        with patch("financial_llm_engine._call_llm", return_value=raw):
            chatbot._call_lm_studio_interpreter("Analyze sentiment")
            self.assertEqual(chatbot._lm_studio_narrative, raw)


if __name__ == "__main__":
    unittest.main()

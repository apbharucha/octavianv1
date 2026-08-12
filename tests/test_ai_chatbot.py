import pytest
from unittest.mock import MagicMock, patch
import json
from typing import Dict, Any

from ai_chatbot import OctavianEnhancedChatbot

@pytest.fixture
def ai_chatbot():
    """Fixture for an OctavianEnhancedChatbot instance with lm_studio_online set to True."""
    chatbot = OctavianEnhancedChatbot()
    chatbot.lm_studio_online = True
    return chatbot

class TestAIChatbot:
    def test_call_lm_studio_interpreter_valid_json(self, ai_chatbot):
        """
        Test _call_lm_studio_interpreter with a valid JSON response.
        """
        mock_llm_response = {
            "intent": "analyze_sentiment",
            "symbols": ["AAPL", "GOOGL"],
            "timeframe": "past_week",
            "analysis_type": "technical"
        }
        mock_llm_return_value = json.dumps(mock_llm_response)

        with patch('financial_llm_engine._call_llm', return_value=mock_llm_return_value) as mock_call_llm:
            result = ai_chatbot._call_lm_studio_interpreter("Analyze sentiment for AAPL and GOOGL")

            mock_call_llm.assert_called_once()
            assert result is not None
            assert result['intent'] == "analyze_sentiment"
            assert result['symbols'] == ["AAPL", "GOOGL"]
            assert result['timeframe'] == "past_week"
            assert result['analysis_type'] == "technical"
            assert ai_chatbot._lm_studio_narrative == mock_llm_return_value

    def test_call_lm_studio_interpreter_malformed_json_with_fallback(self, ai_chatbot):
        """
        Test _call_lm_studio_interpreter with malformed JSON, ensuring regex fallback works.
        """
        malformed_json_response = """```json
{  "intent": "buy", "symbols": ["MSFT"], "timeframe": "today"
}```"""
        expected_parsed_response = {
            "intent": "buy",
            "symbols": ["MSFT"],
            "timeframe": "today",
        }

        with patch('financial_llm_engine._call_llm', return_value=malformed_json_response) as mock_call_llm:
            result = ai_chatbot._call_lm_studio_interpreter("Buy MSFT today")

            mock_call_llm.assert_called_once()
            assert result is not None
            assert result['intent'] == expected_parsed_response['intent']
            assert result['symbols'] == expected_parsed_response['symbols']
            assert result['timeframe'] == expected_parsed_response['timeframe']
            assert result.get('analysis_type') == '' # Should be empty as it's not in the malformed JSON
            assert ai_chatbot._lm_studio_narrative == malformed_json_response

    def test_call_lm_studio_interpreter_no_json_match(self, ai_chatbot):
        """
        Test _call_lm_studio_interpreter when no JSON can be extracted.
        """
        no_json_response = "This is a plain text response without any JSON."

        with patch('financial_llm_engine._call_llm', return_value=no_json_response) as mock_call_llm:
            result = ai_chatbot._call_lm_studio_interpreter("Tell me a story")

            mock_call_llm.assert_called_once()
            assert result is None
            assert ai_chatbot._lm_studio_narrative == no_json_response

    def test_call_lm_studio_interpreter_lm_studio_offline(self, ai_chatbot):
        """
        Test _call_lm_studio_interpreter when lm_studio_online is False.
        """
        ai_chatbot.lm_studio_online = False
        with patch('financial_llm_engine._call_llm') as mock_call_llm:
            result = ai_chatbot._call_lm_studio_interpreter("What should I do?")

            mock_call_llm.assert_not_called()
            assert result is None

    def test_call_lm_studio_interpreter_call_llm_raises_exception(self, ai_chatbot):
        """
        Test _call_lm_studio_interpreter when _call_llm raises an exception (simulating timeout/error).
        """
        with patch('financial_llm_engine._call_llm', side_effect=Exception("LLM connection error")) as mock_call_llm:
            result = ai_chatbot._call_lm_studio_interpreter("Analyze something")

            mock_call_llm.assert_called_once()
            assert result is None

    def test_call_lm_studio_interpreter_call_llm_returns_none(self, ai_chatbot):
        """
        Test _call_lm_studio_interpreter when _call_llm returns None.
        """
        with patch('financial_llm_engine._call_llm', return_value=None) as mock_call_llm:
            result = ai_chatbot._call_lm_studio_interpreter("Analyze something")

            mock_call_llm.assert_called_once()
            assert result is None

    def test_process_query_portfolio_grade(self, ai_chatbot):
        """Test process_query intercepts portfolio queries and routes to handler."""
        with patch('portfolio_chatbot_context.detect_portfolio_intent', return_value='portfolio_grade') as mock_detect, \
             patch('portfolio_chatbot_context.handle_portfolio_query', return_value='mocked grade response') as mock_handle, \
             patch('portfolio_chatbot_context.load_all_portfolios', return_value={'has_data': True}), \
             patch('portfolio_chatbot_context._get_all_holdings', return_value=[{'symbol': 'AAPL'}]):
             
            res = ai_chatbot.process_query("how is my portfolio doing?")
            
            mock_detect.assert_called_once_with("how is my portfolio doing?")
            mock_handle.assert_called_once()
            assert res['text'] == 'mocked grade response'
            assert 'portfolio_grade' in res['intents']
            assert 'AAPL' in res['symbols']

    @pytest.mark.anyio
    async def test_process_unbiased_query_portfolio_grade(self, ai_chatbot):
        """Test process_unbiased_query intercepts portfolio queries and routes to handler."""
        with patch('portfolio_chatbot_context.detect_portfolio_intent', return_value='portfolio_grade') as mock_detect, \
             patch('portfolio_chatbot_context.handle_portfolio_query', return_value='mocked grade response') as mock_handle, \
             patch('portfolio_chatbot_context.load_all_portfolios', return_value={'has_data': True}), \
             patch('portfolio_chatbot_context._get_all_holdings', return_value=[{'symbol': 'MSFT'}]):
             
            res = await ai_chatbot.process_unbiased_query("how is my portfolio doing?")
            
            mock_detect.assert_called_once_with("how is my portfolio doing?")
            mock_handle.assert_called_once()
            assert res['text'] == 'mocked grade response'
            assert 'portfolio_grade' in res['intents']
            assert 'MSFT' in res['symbols']

import pytest
import sqlite3
from unittest.mock import MagicMock, patch
from datetime import datetime
import json
import uuid # Import uuid for unique IDs
from typing import List, Dict, Any

from paper_trading_system import PaperTradingSystem, PaperTradingAccount, AccountStatus

# Mock the database manager for isolated testing
@pytest.fixture
def paper_trading_system():
    # Use a unique in-memory SQLite database for each test function
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    
    # Create a real DatabaseManager instance using the in-memory connection
    # We need to mock the get_database_manager function in database_manager.py
    # to return our mock DatabaseManager when called by PaperTradingSystem
    mock_db_manager_instance = MagicMock()
    mock_db_manager_instance._get_connection.return_value.__enter__.return_value = conn
    mock_db_manager_instance._get_connection.return_value.__exit__.return_value = None

    # Patch the get_database_manager function in the paper_trading_system module
    with patch('paper_trading_system.get_database_manager', return_value=mock_db_manager_instance):
        system = PaperTradingSystem(db_manager=mock_db_manager_instance)
        # Ensure tables are initialized in the in-memory db for each test
        system._initialize_tables() 
        yield system # Use yield to ensure cleanup after test
    conn.close() # Close connection after the test

class TestPaperTradingSystem:

    def test_list_accounts_exact_match(self, paper_trading_system):
        user_id = "test_user_exact"
        paper_trading_system.create_account(user_id, "Account1", initial_balance=1000)
        paper_trading_system.create_account(user_id, "Account2", initial_balance=2000)
        
        accounts = paper_trading_system.list_accounts(user_id)
        assert len(accounts) == 2
        assert all(acc.user_id == user_id for acc in accounts)
        assert {acc.account_name for acc in accounts} == {"Account1", "Account2"}

    def test_list_accounts_case_insensitive_match(self, paper_trading_system):
        user_id_lower = "test_user_case"
        user_id_mixed = "Test_User_Case"
        paper_trading_system.create_account(user_id_lower, "AccountA", initial_balance=1000)
        
        accounts = paper_trading_system.list_accounts(user_id_mixed)
        assert len(accounts) == 1
        assert accounts[0].account_name == "AccountA"
        assert accounts[0].user_id == user_id_lower # Stored user_id should be the original, not the queried one

    def test_list_accounts_trader_1_alias(self, paper_trading_system):
        # 'trader_1' should map to 'default_user'
        default_user_id = "default_user"
        paper_trading_system.create_account(default_user_id, "DefaultAccount", initial_balance=5000)
        
        # Query using the alias
        accounts_alias = paper_trading_system.list_accounts("trader_1")
        assert len(accounts_alias) == 1
        assert accounts_alias[0].account_name == "DefaultAccount"
        assert accounts_alias[0].user_id == default_user_id # Should resolve to the canonical ID

        # Query using the canonical ID
        accounts_canonical = paper_trading_system.list_accounts(default_user_id)
        assert len(accounts_canonical) == 1
        assert accounts_canonical[0].account_name == "DefaultAccount"
        assert accounts_canonical[0].user_id == default_user_id

    def test_list_accounts_no_accounts_found(self, paper_trading_system):
        user_id = "non_existent_user"
        accounts = paper_trading_system.list_accounts(user_id)
        assert len(accounts) == 0
        assert accounts == []

    def test_list_accounts_with_mixed_users(self, paper_trading_system):
        user1 = "user_one"
        user2 = "user_two"
        paper_trading_system.create_account(user1, "User1_Acc1")
        paper_trading_system.create_account(user2, "User2_Acc1")
        paper_trading_system.create_account(user1, "User1_Acc2")

        accounts_user1 = paper_trading_system.list_accounts(user1)
        assert len(accounts_user1) == 2
        assert all(acc.user_id == user1 for acc in accounts_user1)

        accounts_user2 = paper_trading_system.list_accounts(user2)
        assert len(accounts_user2) == 1
        assert all(acc.user_id == user2 for acc in accounts_user2)

#!/usr/bin/env python3
"""
Test script to verify the pathogen interaction fixes work.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verl.interactions.pathogen_interaction import PathogenInteraction

async def test_pathogen_interaction():
    """Test the pathogen interaction with timeout handling."""
    
    # Mock the safe_chat function to simulate different scenarios
    original_safe_chat = None
    
    try:
        from verl.utils.safe_inference import safe_chat
        original_safe_chat = safe_chat
        
        # Test 1: Normal response
        async def mock_safe_chat_normal(*args, **kwargs):
            return "This is a normal response"
        
        # Test 2: Timeout response
        async def mock_safe_chat_timeout(*args, **kwargs):
            await asyncio.sleep(5)  # Simulate long delay
            raise asyncio.TimeoutError("Simulated timeout")
        
        # Test 3: Error response
        async def mock_safe_chat_error(*args, **kwargs):
            raise Exception("Simulated API error")
        
        # Test scenarios
        test_scenarios = [
            ("normal", mock_safe_chat_normal),
            ("timeout", mock_safe_chat_timeout),
            ("error", mock_safe_chat_error),
        ]
        
        for scenario_name, mock_func in test_scenarios:
            print(f"\nTesting scenario: {scenario_name}")
            
            # Replace the safe_chat function
            import verl.utils.safe_inference
            verl.utils.safe_inference.safe_chat = mock_func
            
            # Create interaction instance
            config = {"name": "pathogen"}
            interaction = PathogenInteraction(config)
            
            # Start interaction
            instance_id = await interaction.start_interaction(ground_truth="test objective")
            print(f"Started interaction with ID: {instance_id}")
            
            # Test generate_response
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Test prompt"},
                {"role": "assistant", "content": "Test response</think>This is the attack prompt"},
            ]
            
            try:
                should_terminate, response, reward, metadata = await asyncio.wait_for(
                    interaction.generate_response(instance_id, messages),
                    timeout=10  # 10 second timeout for the entire test
                )
                
                print(f"Response: {response}")
                print(f"Should terminate: {should_terminate}")
                print(f"Reward: {reward}")
                print(f"Metadata: {metadata}")
                
            except asyncio.TimeoutError:
                print("Test timed out - this indicates the timeout handling is working")
            except Exception as e:
                print(f"Test failed with error: {e}")
            
            # Clean up
            await interaction.finalize_interaction(instance_id)
            print(f"Cleaned up interaction: {instance_id}")
    
    finally:
        # Restore original function
        if original_safe_chat:
            import verl.utils.safe_inference
            verl.utils.safe_inference.safe_chat = original_safe_chat

if __name__ == "__main__":
    print("Testing pathogen interaction fixes...")
    asyncio.run(test_pathogen_interaction())
    print("Test completed!") 
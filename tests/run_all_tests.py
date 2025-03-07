#!/usr/bin/env python3
"""
Test runner script for all extractors.
This script runs all the test scripts in the tests directory.
"""

import os
import sys
import logging
import subprocess
import argparse
from dotenv import load_dotenv

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'test_runner.log'))
    ],
    force=True
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def run_test(test_script):
    """Run a test script and return the result."""
    logger.info(f"Running test: {test_script}")
    try:
        result = subprocess.run(
            [sys.executable, test_script],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            logger.info(f"✅ Test {test_script} completed successfully")
            return True
        else:
            logger.error(f"❌ Test {test_script} failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Error running test {test_script}: {e}")
        return False

def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description='Run extractor tests')
    parser.add_argument('--test', help='Specific test to run (e.g., test_tabula_extractor.py)')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--list', action='store_true', help='List available tests')
    args = parser.parse_args()
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get all test scripts
    test_scripts = [
        os.path.join(current_dir, f) for f in os.listdir(current_dir)
        if f.startswith('test_') and f.endswith('.py') and f != os.path.basename(__file__)
    ]
    
    # Sort test scripts
    test_scripts.sort()
    
    if args.list:
        logger.info("Available tests:")
        for script in test_scripts:
            logger.info(f"  - {os.path.basename(script)}")
        return
    
    if args.test:
        # Run a specific test
        test_path = os.path.join(current_dir, args.test)
        if os.path.exists(test_path):
            success = run_test(test_path)
            if success:
                logger.info(f"Test {args.test} completed successfully")
            else:
                logger.error(f"Test {args.test} failed")
                sys.exit(1)
        else:
            logger.error(f"Test script {args.test} not found")
            sys.exit(1)
    elif args.all:
        # Run all tests
        logger.info(f"Running all {len(test_scripts)} tests...")
        
        # Run CUDA test first
        cuda_test = os.path.join(current_dir, 'test_cuda.py')
        if os.path.exists(cuda_test):
            logger.info("Running CUDA test first...")
            run_test(cuda_test)
            # Remove from the list to avoid running it twice
            if cuda_test in test_scripts:
                test_scripts.remove(cuda_test)
        
        # Run the rest of the tests
        successes = 0
        failures = 0
        
        for script in test_scripts:
            if run_test(script):
                successes += 1
            else:
                failures += 1
        
        logger.info(f"Test run completed: {successes} passed, {failures} failed")
        
        if failures > 0:
            logger.error(f"❌ {failures} tests failed")
            sys.exit(1)
        else:
            logger.info("✅ All tests passed")
    else:
        logger.info("No test specified. Use --test, --all, or --list")
        logger.info("Example: python run_all_tests.py --all")
        logger.info("Example: python run_all_tests.py --test test_tabula_extractor.py")
        logger.info("Example: python run_all_tests.py --list")

if __name__ == "__main__":
    main()
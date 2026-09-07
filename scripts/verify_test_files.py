#!/usr/bin/env python3
"""Verify that all new test files are syntactically correct and discoverable."""

import ast
import sys
from pathlib import Path

def verify_test_file(filepath: Path) -> tuple[bool, list[str]]:
    """Verify a test file is syntactically correct and list test methods."""
    try:
        with open(filepath) as f:
            source = f.read()

        tree = ast.parse(source)

        # Count test classes and methods
        test_classes = []
        test_methods = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name.startswith("Test"):
                    test_classes.append(node.name)
                    # Count test methods in this class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                            test_methods.append(f"{node.name}::{item.name}")

        return True, (test_classes, test_methods)
    except SyntaxError as e:
        return False, [str(e)]
    except Exception as e:
        return False, [f"Error: {str(e)}"]


def main():
    """Verify all test files."""
    test_files = [
        "tests/test_star_core_executor.py",
        "tests/test_query_analyzer_core.py",
        "tests/test_clause_executor_core.py",
        "tests/test_mutation_engine_core.py",
        "tests/test_pattern_matcher_core.py",
        "tests/test_relation_sql_core.py",
        "tests/test_remaining_core_modules.py",
        "tests/test_advanced_feature_modules.py",
    ]

    repo_root = Path("/home/zac/git/pycypher-nmetl")

    print("=" * 80)
    print("VERIFYING TEST FILES")
    print("=" * 80)

    total_classes = 0
    total_methods = 0
    all_valid = True

    for test_file in test_files:
        filepath = repo_root / test_file

        if not filepath.exists():
            print(f"❌ {test_file:<50} NOT FOUND")
            all_valid = False
            continue

        valid, result = verify_test_file(filepath)

        if not valid:
            print(f"❌ {test_file:<50} SYNTAX ERROR")
            for error in result:
                print(f"   {error}")
            all_valid = False
        else:
            test_classes, test_methods = result
            total_classes += len(test_classes)
            total_methods += len(test_methods)

            print(f"✅ {test_file:<50} {len(test_classes)} classes, {len(test_methods)} methods")

    print("=" * 80)
    print(f"SUMMARY: {total_classes} test classes, {total_methods} test methods")
    print(f"Status: {'✅ ALL FILES VALID' if all_valid else '❌ SOME FILES INVALID'}")
    print("=" * 80)

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())

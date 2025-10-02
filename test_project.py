"""
Project validation script - checks code syntax and structure without running models.
"""

import os
import sys
import ast
import importlib.util

def check_python_syntax(file_path):
    """Check if Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            ast.parse(f.read())
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"

def check_imports(file_path):
    """Check if file can be parsed for imports."""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)

        return True, imports
    except Exception as e:
        return False, str(e)

def check_functions(file_path):
    """Extract function names from file."""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

        return True, {'functions': functions, 'classes': classes}
    except Exception as e:
        return False, str(e)

def main():
    """Run project validation."""
    print("=" * 80)
    print("MEDICAL IMAGE CLASSIFICATION - PROJECT VALIDATION")
    print("=" * 80)

    src_dir = 'src'
    python_files = [
        'src/data_loader.py',
        'src/models.py',
        'src/train.py',
        'src/evaluate.py',
        'src/predict.py'
    ]

    print("\n[1/3] Checking Python Syntax...")
    print("-" * 80)
    all_valid = True

    for file_path in python_files:
        valid, msg = check_python_syntax(file_path)
        status = "" if valid else ""
        print(f"  {status} {file_path}: {msg}")
        if not valid:
            all_valid = False

    print("\n[2/3] Checking Imports...")
    print("-" * 80)

    for file_path in python_files:
        valid, imports = check_imports(file_path)
        if valid:
            print(f"   {file_path}")
            # Print first few imports
            if imports:
                print(f"    Imports: {', '.join(imports[:5])}{'...' if len(imports) > 5 else ''}")
        else:
            print(f"   {file_path}: {imports}")
            all_valid = False

    print("\n[3/3] Checking Code Structure...")
    print("-" * 80)

    for file_path in python_files:
        valid, result = check_functions(file_path)
        if valid:
            print(f"   {file_path}")
            if result['classes']:
                print(f"    Classes: {', '.join(result['classes'])}")
            if result['functions']:
                funcs = result['functions'][:3]
                print(f"    Functions: {', '.join(funcs)}{'...' if len(result['functions']) > 3 else ''}")
        else:
            print(f"   {file_path}: {result}")
            all_valid = False

    print("\n" + "=" * 80)
    print("PROJECT STRUCTURE VALIDATION")
    print("=" * 80)

    # Check directory structure
    dirs = ['data', 'models', 'results', 'src', 'notebooks']
    print("\nDirectory Structure:")
    for dir_name in dirs:
        exists = os.path.exists(dir_name)
        status = "" if exists else ""
        print(f"  {status} {dir_name}/")

    # Check key files
    files = ['README.md', 'requirements.txt', '.gitignore']
    print("\nKey Files:")
    for file_name in files:
        exists = os.path.exists(file_name)
        status = "" if exists else ""
        print(f"  {status} {file_name}")

    print("\n" + "=" * 80)

    if all_valid:
        print(" PROJECT VALIDATION PASSED")
        print("\nAll Python files have valid syntax and structure!")
        print("\nNext Steps:")
        print("1. Download dataset from Kaggle")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run training: python src/train.py --model_type custom --epochs 5")
        return 0
    else:
        print(" PROJECT VALIDATION FAILED")
        print("\nSome files have issues. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

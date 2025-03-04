"""
Git Commit Troubleshooter

This script helps diagnose and fix common issues that prevent successful git commits.
Run this script from the root of your repository to identify and resolve problems.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import platform

def run_command(command, show_output=True):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if show_output:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"Error: {result.stderr}")
        return result
    except Exception as e:
        print(f"Failed to execute command: {command}")
        print(f"Error: {str(e)}")
        return None

def is_git_repo():
    """Check if the current directory is a git repository."""
    return os.path.isdir(".git")

def check_git_installed():
    """Check if git is installed and accessible."""
    print("Checking git installation...")
    result = run_command("git --version", show_output=False)
    if result and result.returncode == 0:
        print(f"✓ Git is installed: {result.stdout.strip()}")
        return True
    else:
        print("✗ Git is not installed or not in PATH")
        return False

def check_git_config():
    """Check if git user name and email are configured."""
    print("\nChecking git configuration...")
    user_name = run_command("git config user.name", show_output=False).stdout.strip()
    user_email = run_command("git config user.email", show_output=False).stdout.strip()
    
    if user_name:
        print(f"✓ Git user.name is set to: {user_name}")
    else:
        print("✗ Git user.name is not set")
        
    if user_email:
        print(f"✓ Git user.email is set to: {user_email}")
    else:
        print("✗ Git user.email is not set")
        
    return bool(user_name and user_email)

def fix_git_config():
    """Fix git configuration if needed."""
    if not check_git_config():
        print("\nSetting up git configuration...")
        name = input("Enter your name for git config: ")
        email = input("Enter your email for git config: ")
        
        if name:
            run_command(f'git config --local user.name "{name}"')
        if email:
            run_command(f'git config --local user.email "{email}"')
        print("Git configuration updated.")

def check_git_status():
    """Check current git status."""
    print("\nChecking git status...")
    run_command("git status")

def check_large_files():
    """Check for large files that might cause commit issues."""
    print("\nChecking for large files (>50MB)...")
    large_files = []
    
    for root, dirs, files in os.walk(".", topdown=True):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                if size_mb > 50:
                    large_files.append((file_path, size_mb))
            except:
                pass
    
    if large_files:
        print("Found large files that might cause commit issues:")
        for file_path, size_mb in large_files:
            print(f"  - {file_path}: {size_mb:.2f}MB")
        return False
    else:
        print("✓ No large files found")
        return True

def check_permissions():
    """Check if there are permission issues."""
    print("\nChecking file permissions...")
    try:
        # Try to write to a temporary file in .git directory
        test_file = os.path.join(".git", "permission_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("✓ Write permissions are OK")
        return True
    except Exception as e:
        print(f"✗ Permission issue detected: {str(e)}")
        return False

def fix_permissions():
    """Fix permission issues if necessary."""
    if not check_permissions():
        print("\nAttempting to fix permission issues...")
        if platform.system() == "Windows":
            # On Windows, try to take ownership
            run_command("takeown /f .git /r /d y")
            print("Executed takeown command. Please try committing again.")
        else:
            # On Unix-based systems, change ownership
            run_command("chmod -R u+w .git")
            print("Updated file permissions. Please try committing again.")

def check_for_conflicts():
    """Check for merge conflicts."""
    print("\nChecking for merge conflicts...")
    result = run_command("git diff --name-only --diff-filter=U", show_output=False)
    if result.stdout.strip():
        conflict_files = result.stdout.strip().split('\n')
        print(f"✗ Merge conflicts found in {len(conflict_files)} files:")
        for file in conflict_files:
            print(f"  - {file}")
        return False
    else:
        print("✓ No merge conflicts detected")
        return True

def attempt_commit_fix():
    """Try to fix common commit issues."""
    print("\nAttempting to fix common commit issues...")
    
    # Check if there are staged changes
    result = run_command("git diff --cached --name-only", show_output=False)
    if not result.stdout.strip():
        print("No changes staged for commit. Try 'git add <file>' first.")
        return
    
    # Try to commit with --no-verify flag
    print("\nTrying commit with --no-verify flag...")
    commit_msg = input("Enter commit message (or press Enter for default): ") or "Fix commit issues"
    result = run_command(f'git commit --no-verify -m "{commit_msg}"')
    
    if result.returncode == 0:
        print("✓ Commit successful!")
    else:
        print("✗ Commit failed even with --no-verify flag.")
        
        # Try to repair the repository
        print("\nAttempting to repair the repository...")
        run_command("git fsck --full")
        
        # Try to reset the index
        should_reset = input("\nWould you like to try resetting the index? (y/n): ").lower() == 'y'
        if should_reset:
            print("Backing up any staged changes...")
            run_command("git diff --cached > staged_changes.patch")
            print("Backup saved to staged_changes.patch")
            
            print("Resetting the index...")
            run_command("git reset")
            
            print("Please stage your changes again with 'git add' and then commit.")

def suggest_gitignore_fixes():
    """Check and suggest fixes for .gitignore."""
    print("\nChecking .gitignore configuration...")
    
    if not os.path.exists(".gitignore"):
        print("No .gitignore file found. Creating one with common Python patterns...")
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# ML models
*.pkl
*.model
*.keras
*.h5
*.pt
*.pth
*.bin

# IDE settings
.idea/
.vscode/
*.swp
*.swo
"""
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("Created .gitignore file with common patterns.")
    else:
        print("✓ .gitignore file exists")
        
        # Check if common directories are ignored
        with open(".gitignore", "r") as f:
            gitignore_content = f.read()
            
        missing_patterns = []
        common_patterns = ["__pycache__/", "*.py[cod]", "build/", "dist/", "*.pkl", "*.h5", "*.pt"]
        
        for pattern in common_patterns:
            if pattern not in gitignore_content:
                missing_patterns.append(pattern)
                
        if missing_patterns:
            print("Consider adding these common patterns to your .gitignore:")
            for pattern in missing_patterns:
                print(f"  - {pattern}")

def clean_repo():
    """Clean the repository to fix potential issues."""
    print("\nCleaning repository...")
    
    # Clean untracked files?
    show_untracked = run_command("git ls-files --others --exclude-standard", show_output=False).stdout
    if show_untracked.strip():
        print("\nFound untracked files:")
        print(show_untracked)
        should_clean = input("Would you like to remove untracked files? (y/n): ").lower() == 'y'
        if should_clean:
            run_command("git clean -fd")
            print("Removed untracked files and directories.")
    
    # Run garbage collection
    print("\nRunning git garbage collection...")
    run_command("git gc")

def main():
    """Main function to run all checks and fixes."""
    print("=== Git Commit Troubleshooter ===\n")
    
    if not is_git_repo():
        print("Error: Not a git repository. Please run this script from the root of your git repository.")
        return
    
    # Check git installation
    if not check_git_installed():
        print("\nPlease install git and add it to your PATH.")
        return
    
    # Fix git configuration if needed
    fix_git_config()
    
    # Check git status
    check_git_status()
    
    # Check for large files
    check_large_files()
    
    # Check permissions and fix if needed
    fix_permissions()
    
    # Check for conflicts
    check_for_conflicts()
    
    # Suggest .gitignore improvements
    suggest_gitignore_fixes()
    
    # Prompt for cleaning repository
    should_clean = input("\nWould you like to clean the repository? (y/n): ").lower() == 'y'
    if should_clean:
        clean_repo()
    
    # Prompt to attempt fixing commit issues
    should_fix = input("\nWould you like to attempt fixing commit issues? (y/n): ").lower() == 'y'
    if should_fix:
        attempt_commit_fix()
    
    print("\n=== Troubleshooting Complete ===")
    print("If you're still having issues, consider:")
    print("1. Checking your internet connection")
    print("2. Making sure your remote repository is accessible")
    print("3. Creating a new clone of the repository")
    print("4. Reaching out to your repository administrator")

if __name__ == "__main__":
    main()

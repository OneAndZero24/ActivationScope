---
name: ast-search
description: Run this skill when you need to find where a function, class, or variable is defined without blindly reading entire files.
---
Use the `ast-grep` (command `sg`) CLI tool to search the codebase contextually. 
This tool understands Abstract Syntax Trees (AST) and is superior to standard grep.

Command format: `sg -p "<search_pattern>" <directory_or_file>`

Example: `sg -p "class $A { $$$ }" csrc/` 
Example: `sg -p "def register_cxx_hook($$$): $$$" activationscope/`

Use this to quickly map out API boundaries and find exact definitions of PyTorch extensions.

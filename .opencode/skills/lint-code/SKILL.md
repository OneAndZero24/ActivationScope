---
name: lint-code
description: Run this skill to format and lint code files after any modification.
---
Execute these bash tools sequentially to clean files:

1. For Python files, run:
   `ruff check --fix <file>`
   `ruff format <file>`
2. For C++ files, run:
   `clang-format -i <file>`

If a tool outputs errors that it cannot automatically fix, analyze the output and fix the syntax manually before returning to the user.

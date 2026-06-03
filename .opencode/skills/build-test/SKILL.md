---
name: build-test
description: Run this skill to compile the C++ extension and run the unit tests.
---
Execute the following verification routine in bash:

1. Recompile the editable package: `pip install -e .`
   (Note: ccache and ninja will automatically accelerate this).
2. Run the test framework: `python -m pytest tests/`

If compilation fails, you MUST read the C++ compiler output (look for `error:` lines) to identify syntax or linking errors, and fix the C++ files before retrying.

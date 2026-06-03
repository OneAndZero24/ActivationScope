---
name: leak-check
description: Run this skill to specifically inspect the Python/C++ boundary for PyTorch tensor graph leaks or unbounded memory growth.
---
Run the target test suite using memory tracking flags:
`python -m pytest tests/ -s --durations=0` 

Monitor the resident set size (RSS) or output metrics from the tests. 
If you observe unbound memory growth across forward/backward passes, flag it immediately as a computational graph retention leak caused by missing `.clear()` calls or missing `NoGradGuard` blocks.

# agenthandoff rule for agent instruction files

Paste this into CLAUDE.md, AGENTS.md, .cursorrules or your harness's equivalent.

```markdown
## One-call file overviews (agenthandoff)

- For one specific question about a file, your own search and partial reads are usually
  cheaper.
- When you need the whole shape of a big log, data file or folder in one call, or you have
  no shell, use `handoff_read` (MCP) or `agenthandoff read PATH... -q "QUESTION"` (shell). It
  answers with code, no model:
  - a name in backticks or quotes in the question gets the lines that hold it;
  - source code gets an outline (definitions, docstrings, constants, line numbers);
  - logs get a profile (each kind of line once with counts, odd values, rare lines in full);
  - JSON and CSV get their shape (keys or columns, spread of values, last rows).
- The first line says which tool ran (ROUTE: ...). Every line number is real: read the
  pointed lines before acting on them.
- If it says "Read the files yourself" or exits with code 2, do the task yourself.
- Model answers (ROUTE: digest) appear only if the user turned the model on. They are a small
  model's wording and can be wrong. If it says RECALL CHECK: FAILED, it may have missed things.
- `script_<name>` tools are the user's own scripts; prefer one when it fits the task.
```

# Changelog

## 0.1.0 (2026-11-13)

First release. Your agent hands file work to agenthandoff, which routes it to plain code
first, then your scripts, then any model you set up. You get a short overview with real
`file:line` pointers. With no model set up, no GPU or network is needed.

### What it does
- `read`: picks a code tool by fixed rules. `find` for a named string, `outline` for source
  code, `profile` for logs, `shape` for JSON, JSON lines and CSV, or one of your own scripts.
- `profile`: each kind of log line once, with counts, the spread of its numbers, the odd one
  out, the first big jump and rare lines in full. The last lines of the file come first.
- `shape`: keys or columns, types and spread of values, keys found in only a few rows with
  their values, and the last rows of a JSON-lines file.
- `digest` (optional): a local model reads plain text. Line numbers come from code, quotes
  not in the file are dropped, and a recall check says when the model could not read a chunk.
- MCP server (`agenthandoff-mcp`), guided `setup`, `check`, `config`, `stats`.
- Windows and macOS, Python 3.10+, standard library only.

### Good to know
- In our test, an agent that could only open files read about 4 times less with
  agenthandoff, and got as many answers right.
- With a shell, an agent's own search is often cheaper for one specific question.

### Defaults
- The model is optional and off for `read`: plain text is handed back instead of sent to the
  model. Turn it on with `agenthandoff config set route.allow_model true`. `digest` uses the
  model whenever you ask for it.
- `check` exits 0 with no model set up, and 2 when a model is set up and fails.

### Known limits
- XML and other plain text get no code tool. With the model off, `read` says so and exits 2.
- With a problems question, a log profile can hide the numbers you asked about (for example,
  a final score that is not on a problem line). Ask without problem words to see them.
- Files over 64 MB are skipped by default (`tools.max_file_bytes`). Reading a 250 MB file
  took about 2 GB of memory; raise the limit only on a machine with room for that.
- A large log takes time: about 110 s for a 220 MB servicing log.
- Output is cut at 12,000 characters (`tools.max_output_chars`), and says so.
- Not yet tried with a real model on a Mac, or with LM Studio.

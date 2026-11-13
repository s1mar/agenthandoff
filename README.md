# agenthandoff

Your AI agent hands file work to agenthandoff, and agenthandoff routes it:

1. plain code first (logs, data files, source code);
2. then your own scripts;
3. then a model, only if you set one up: local (llama.cpp, Ollama, LM Studio) or any
   provider you connect (an API, a CLI, a Python function).

What comes back is a short overview with real `file:line` pointers.
With no model set up, it needs no GPU and no network.

## Example

```
$ agenthandoff read app.log -q "anything unusual?"
ROUTE: profile (1 file of logs)
LOG PROFILE: app.log - 1201 lines, 4 line patterns
PROBLEM WORDS (...): error on 3 lines (first L419), warning on 1 line (first L422)

LAST LINES:
   L1201: INFO shutdown complete

RARE LINES WITH PROBLEM WORDS (patterns seen 3 times or fewer), 4:
   L419: 2026-03-02 10:06:58 ERROR db: connection refused (pool=main, retry 1/3)
   L420: 2026-03-02 10:06:59 ERROR db: connection refused (pool=main, retry 2/3)
   L421: 2026-03-02 10:07:00 ERROR db: connection refused (pool=main, retry 3/3)
   L422: 2026-03-02 10:07:01 WARN db: switched to replica db-2

[1196x, lines 1-1200] INFO GET <path> <n> in <n> ms
   #2: min 8 (L182), median 33, max 2400 (L901)
```

A database outage at L419, one slow request at L901, and a clean shutdown.
(Shortened; the full output also lists examples and the last few lines.)

## What it handles

| Input | Tool | You get |
|---|---|---|
| Logs | `profile` | each kind of line once, counts, odd values |
| JSON, JSON lines, CSV | `shape` | keys or columns, types, spread of values |
| Source code | `outline` | definitions, docstrings, constants |
| An exact name in the question | `find` | the lines that hold it |
| Matches one of your scripts | your script | its output |
| Plain text | `digest` | a model reads it (optional) |

**When it helps.** In our test, an agent that could only open files (no search) read about
4 times less with agenthandoff, and got as many answers right. It did best on whole-file
facts in big files, like counts and error kinds.

**When it does not.** If your agent has a shell, its own search (`grep`, `rg`) is often
cheaper for one specific question. For exact detail, read the lines it points to.

## Install

You need Python 3.10 or newer, and pipx.

- macOS: `brew install python pipx && pipx ensurepath`
- Windows: `winget install Python.Python.3.12`, open a new PowerShell, then
  `py -m pip install --user pipx` and `py -m pipx ensurepath`

Then get the code and install it:

```bash
git clone https://github.com/s1mar/agenthandoff.git
cd agenthandoff
pipx install .
agenthandoff check
```

`check` should end with "Ready for the code tools".

## Connect your agent

Most coding agents can load extra tools through MCP (a standard plug-in protocol).

Claude Code:

```bash
claude mcp add agenthandoff -- agenthandoff-mcp
```

Codex, in `~/.codex/config.toml`:

```toml
[mcp_servers.agenthandoff]
command = "agenthandoff-mcp"
```

Cursor, in `~/.cursor/mcp.json`:

```json
{ "mcpServers": { "agenthandoff": { "command": "agenthandoff-mcp" } } }
```

Any other agent or script can run the command:

```bash
agenthandoff read path/to/folder -q 'Where is `retry_limit` set?' --include '*.py'
```

To tell your agent when to use it, paste [AGENTS_SNIPPET.md](AGENTS_SNIPPET.md) into its
instruction file.

## Add a model (optional)

```bash
agenthandoff setup
agenthandoff config set route.allow_model true
```

Setup picks a model that fits your computer and asks before installing anything.
The second line lets `read` send plain text to the model; `digest` uses it either way.
Models carry their own license terms.

## More

- [docs/reference.md](docs/reference.md): every command, setting, provider and fix.
- [CHANGELOG.md](CHANGELOG.md): what is in this version, and known limits.
- License: Apache-2.0 ([LICENSE](LICENSE)).

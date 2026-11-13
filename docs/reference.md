# agenthandoff reference

The [README](../README.md) covers setup. This page has every command, setting and option.

## Commands

```
agenthandoff read PATH... [-q QUESTION] [--tool find|outline|profile|shape|digest]
agenthandoff profile PATH...
agenthandoff digest PATH... -q QUESTION [--json] [--no-recall-check]
agenthandoff script NAME [PATH...] [-q QUESTION]
agenthandoff config show | get KEY | set KEY VALUE | unset KEY | path
agenthandoff stats
agenthandoff setup [--backend llamacpp|ollama|openai] [--model M] [--url U] [--llama-server PATH] [--yes]
agenthandoff check
agenthandoff-mcp          # the MCP server; your agent starts it
```

`read`, `profile`, `digest` and `script` also take:
- `--include PATTERN`: in folders, only files whose name matches, e.g. `"*.py"`. Repeatable.
- `--root DIR`: show paths relative to this folder.
- `--set KEY=VALUE`: change a setting for this run only. Repeatable.

**`read`**
- Picks the tool by fixed rules (the table in the [README](../README.md#what-it-handles)).
- Each part starts with which tool ran and why, e.g. `ROUTE: outline (3 files of source code)`.
- Put exact names in backticks or quotes to get their lines: ``-q 'where is `retry_limit` set?'``.
- `--tool` forces one tool.

**`profile`** is the log tool on its own.
- Speed: under a second for a few MB; about 110 s for a 220 MB log.
- Its `PROBLEM WORDS` line counts whole words, any case: error, fail (failed, failure),
  fatal, exception, critical, warning, denied, timeout, abort, panic, cannot (unable,
  could not).
- A line that reports none is not counted: `error = 0`, `errors: none`, `no errors`,
  `without errors`, `found 0 errors`.
- `FATAL: 0 bytes free` and `shard 0 errors: disk full` still count.
- A word on nearly every line is marked as part of the line format and ranked last.
- If your question is about problems (errors, failures, crashes...), you get only the
  problem part. Ask without problem words to see the whole profile.

**Other commands**
- `digest` always uses the model. See [Optional: a local model](#optional-a-local-model).
- `script` runs one of your scripts by name. See [Scripts](#scripts).
- `stats` shows what ran, and questions that reached the model more than once (good
  candidates for a script).

**MCP tools**: `handoff_read`, `handoff_profile`, `script_<name>` for each script, and
`handoff_digest` (only when a model is set up).

**Exit codes**: `0` done. `2` not done, and the message says why. For an agent, `2` means
"do this job yourself".

## Settings

Every setting has a default. `agenthandoff config show` lists each one, its value and
where it came from.

Settings come in layers. A later layer wins:

1. the default;
2. your `~/.agenthandoff/config.json` (`config set` changes it);
3. a `.agenthandoff.json` in the project folder or a parent folder;
4. an environment variable: `AGENTHANDOFF_` plus the key in capitals, `.` as `_`, e.g.
   `AGENTHANDOFF_PROFILE_RARE_COUNT=5`;
5. `--set KEY=VALUE` on the command line.

A project file comes with a cloned repo, so it is limited:
- It may only set tuning values: limits, `route.*`, `chunk_tokens`, `timeout`.
- It can never name a program, a server, a provider or a script.
- It can only make things safer: lower a limit, turn the model off, pick `code_only`.
- Turn project files off with `project_config false`.

A misspelled key is refused with the closest real key ("did you mean ..."). A bad value
says what it must be.

```bash
agenthandoff config set profile.rare_count 5
agenthandoff config set route.policy code_only      # never use a model
agenthandoff config get route.policy
agenthandoff config unset profile.rare_count
```

| Key | Default | Meaning |
|---|---|---|
| `route.policy` | `code_first` | See below. |
| `route.allow_model` | `false` | Let `read` send plain text to the model. |
| `tools.max_output_chars` | `12000` | Longest output. Detail is dropped first, then it is cut (and says so). |
| `tools.max_file_bytes` | `64000000` | Largest file the code tools read. See below. |
| `profile.rare_count` | `3` | A line kind seen this often or less is listed line by line. |
| `profile.max_patterns` | `40` | Line kinds shown in a profile. |
| `profile.max_rare_lines` | `60` | Rare lines shown. |
| `profile.max_line_chars` | `200` | Longest line shown. |
| `outline.max_items` | `200` | Definitions shown per source file. |
| `find.context_lines` | `2` | Lines shown around each match. |
| `find.max_matches` | `40` | Matches shown per term. |
| `shape.max_keys` | `60` | Keys or columns shown per file. |
| `project_config` | `true` | Read `.agenthandoff.json` files in project folders. |
| `digest.recall_check` | `true` | Plant the recall-check line in each digest. |
| `cache.enabled` | `true` | Reuse a model answer for the same question and files. |
| `cache.max_entries` | `500` | Model answers kept. |
| `allow_remote` | `false` | Allow providers that may send file text off this machine. |

`route.policy`:
- `code_first`: code tools; the model only for plain text, and only if `route.allow_model`
  is on.
- `code_only`: never a model.
- `model_only`: always the digest.

`tools.max_file_bytes`: reading a 250 MB file took about 2 GB of memory. Raise the limit
only on a machine with room for that. (The model digest has its own limit: 2 MB per file.)

## Scripts

A script is a program of yours that answers some questions with no model. Add it under
`scripts` in your own config.json (a project file can never add one):

```json
"scripts": [
  {"name": "git_log", "argv": ["git", "log", "--stat", "-n", "20"],
   "description": "the last 20 commits with the files they changed"},
  {"name": "junit", "argv": ["python", "C:/tools/junit_summary.py", "{paths}"],
   "match": ["*.xml"], "question": "fail|test"}
]
```

- `argv` runs with no shell. `{paths}` becomes one argument per path. `{question}` and
  `{root}` are replaced too.
- Every script is an MCP tool `script_<name>` and runs with `agenthandoff script NAME`.
- `read` picks a script by itself only when it has a `question` (a regular expression the
  question must match) and every file matches `match` (default `*`).
- A script that fails, times out (`timeout`, default 60 s) or prints nothing gives exit 2.
  Inside `read`, the built-in tools run instead.

agenthandoff never writes or runs code a model wrote.

## Troubleshooting

| You see | What to do |
|---|---|
| `agenthandoff: command not found` | Open a new terminal after `pipx ensurepath`. |
| `needs reading: no code tool fits` | Plain text or XML. Read it yourself, or turn on the model (below). |
| `raise tools.max_file_bytes to read it` | The file is over the limit. See `tools.max_file_bytes`. |
| Output ends with a cut notice | Narrow the paths, use `--include`, or ask a narrower question. |

Model problems are listed under [Optional: a local model](#model-troubleshooting).

## Optional: a local model

A local model can read plain text (notes, prose, unknown formats) for you. This is the
`digest` tool. It is off for `read` by default, because a small model can misread prose.

### Set it up

```bash
agenthandoff setup
```

Setup asks one question at a time and suggests an answer (press Enter to take it).
It never installs or downloads anything without asking first. It will:

1. **Ask how to run models:**
   - **llama.cpp** (suggested): starts the model only while a job runs, so your GPU memory
     is free the rest of the time. Most control over speed and memory.
   - **Ollama**: if you already use it, or want an app to manage models.
   - **LM Studio**: if you already use it and like a window to manage models.
2. **Install it if needed.** For llama.cpp it offers `brew install llama.cpp` (macOS) or
   `winget install ggml.llamacpp` (Windows). For Ollama and LM Studio it says where to get them.
3. **Pick a model that fits your computer**, from your GPU memory (or a Mac's memory).
   For llama.cpp it downloads the model (2.8 or 4.6 GB) into `~/.agenthandoff/models`,
   checks its SHA-256, and resumes if the download stops (run setup again).
   The models come from Hugging Face and carry their own license terms, not this project's.
4. **Save your choices** to `~/.agenthandoff/config.json`.
5. **Run `agenthandoff check`.**

Then, to let `read` use the model for plain text:

```bash
agenthandoff config set route.allow_model true
```

`digest` uses the model as soon as setup is done.

**For experts**: every question has a flag. With no terminal (for example when an agent runs
setup), nothing is installed or downloaded unless you add `--yes`.

```bash
# your own .gguf and llama-server: nothing to download or install
agenthandoff setup --backend llamacpp --model /path/to/model.gguf --llama-server /path/to/llama-server

# a model from the built-in list (gemma4-e4b or gemma4-e2b), installing llama.cpp if needed
agenthandoff setup --backend llamacpp --model gemma4-e2b --yes

# a llama-server you start yourself
agenthandoff setup --backend llamacpp --url http://127.0.0.1:8080

agenthandoff setup --backend ollama --model qwen3:8b --yes
agenthandoff setup --backend openai --url http://localhost:1234 --model my-model-id
```

With a model, `check` also tests it (paths and numbers made up):

```
[  ok] default: model: /Users/you/.agenthandoff/models/gemma-4-E4B-it-Q4_0.gguf (4378 MB)
[  ok] default: GPU: NVIDIA GeForce RTX 4070, 10900 MB of 12282 MB free
[  ok] default: test prompt: answered in 6.1 s (of which 5.2 s starting the server)
```

The GPU-busy check works only on NVIDIA GPUs, so it is skipped on a Mac.

### What a digest gives back

Format only; the numbers are made up:

```
ANSWER: The build failed because the module ./utils/formatDate could not be found [C1].

CLAIMS (quotes found in the files by code; pointers are file:line):
[C1] build.log:1135  Webpack could not resolve the formatDate module.
      > Module not found: Can't resolve './utils/formatDate' in 'src/pages/billing/history.tsx'

RECALL CHECK: passed
STATS: 1 files, 8 chunks, about 46,000 tokens read locally, 3 claims kept, 2 dropped, 95.0 s, model ...
```

| `digest` option | Meaning |
|---|---|
| `-q`, `--question` | What you want to know. |
| `--json` | Print the full result (all claims) as JSON. |
| `--no-recall-check` | Do not plant the recall-check line. |

Limits: 2 MB per file, and about 200,000 tokens per digest. Folders skip `.git`,
`node_modules`, `.venv` and similar, and binary files.

### Why you can trust it (and when not)

The model is small and can be wrong, so code checks its work:

1. **Code gives the line numbers, not the model.** The model must copy text exactly. Code
   finds that text in the file. A quote that is not in the file is dropped.
2. **Recall check.** Each digest hides one line with a random code in a copy of its largest
   chunk, then asks the model for the code in a separate call. If it does not come back, the
   output says `RECALL CHECK: FAILED`: the model could not read that much text to the end.
3. **It says no instead of guessing.** If the server is down, the GPU is busy, or the input
   is too big, it says so and exits 2.

The one-sentence summaries can still be wrong. Read the pointed lines before acting on
anything that matters.

### How it works

```
your files
   │  code: label every line (L00042: ...) and cut the text into chunks
   ▼
local model, once per chunk: "claims" that each copy an exact quote
   │  code: find each quote in the file, keep file:line, drop quotes that are not there
   ▼
local model, once: write the answer, citing claims as [C1], [C2]
   │  code: replace citations of claims that do not exist with [?]
   ▼
answer + claims with file:line + recall check  →  your agent
```

A model answer is cached in `~/.agenthandoff/cache/`. The same question about the same file
text, with the same provider and settings, answers in milliseconds. Any change is a miss.
Turn it off with `cache.enabled false`.

Every digest, failed or not, is logged in `~/.agenthandoff/logs/digest.jsonl` (question,
sizes, timings, what was dropped). Nothing is sent anywhere.

### Model settings

| config.json key | Default | Meaning |
|---|---|---|
| `backend` | set by setup | `llamacpp`, `ollama` or `openai` (LM Studio and other compatible servers). |
| `model` | set by setup | llamacpp: path to a `.gguf`. ollama: model name. openai: model id. |
| `url` | see below | Server address. |
| `llama_server` | found on PATH | Path to the `llama-server` program. |
| `llama_args` | none | Extra `llama-server` flags. See below. |
| `chunk_tokens` | `6000` | Text per model call (max 8000). Smaller is more accurate, slower. |
| `timeout` | `300` | Seconds to wait for one model call. Raise it on a CPU-only machine. |
| `external_lock` | none | Another program's JSON lock file (with a `"pid"`) to respect. |
| `ignore_gpu_busy` | `false` | Run even when another program is using the GPU (slow). |

Each key also works as an environment variable: `AGENTHANDOFF_` plus the key in capitals,
e.g. `AGENTHANDOFF_CHUNK_TOKENS`.

- `url`: Ollama `http://localhost:11434`, openai `http://localhost:1234`. For llamacpp leave
  it unset so agenthandoff runs the server, or set it to use your own `llama-server`.
- `llama_args`: in config.json a list, e.g. `["-ngl", "20", "--threads", "8"]`; as a variable
  a string. `-ngl 20` keeps 20 layers on the GPU and the rest on the CPU, for a model that
  does not fit.
- `AGENTHANDOFF_HOME` moves the whole `~/.agenthandoff` folder.
- For one run: `AGENTHANDOFF_CHUNK_TOKENS=3000 agenthandoff digest ...` on macOS, or
  `$env:AGENTHANDOFF_CHUNK_TOKENS = "3000"` first in PowerShell.
- The MCP server does not see variables set in your terminal. Put them in the client's
  config (most clients take an `env` block next to `command`).
- **LM Studio** sets the model's context length itself. Set it to at least 12288 tokens, or
  set the provider's `context_tokens` to match, or long inputs get cut off.

### Providers

`backend`, `model` and `url` set up one model the simple way. For any other model, CLI or
service, name it under `providers` and pick it with `provider`. A list in `provider` is tried
in order: when one fails, the next runs.

```json
{
  "providers": {
    "laptop": {"type": "llamacpp", "model": "C:/models/gemma-4-E4B-it-Q4_0.gguf"},
    "ollama-chat": {"type": "openai", "url": "http://localhost:11434", "model": "gemma3:4b"},
    "my-cli": {"type": "command", "argv": ["ollama", "run", "{model}"], "model": "gemma3:4b",
               "remote": false},
    "hosted": {"type": "openai", "url": "https://api.example.com", "model": "some-model",
               "api_key_env": "EXAMPLE_API_KEY"},
    "mine": {"type": "python", "factory": "C:/tools/my_provider.py:make", "options": {"x": 1}}
  },
  "provider": ["laptop", "ollama-chat"]
}
```

| `type` | Talks to | Keys |
|---|---|---|
| `llamacpp` | a `llama-server` agenthandoff starts (or yours, with `url`) | `model`, `url`, `llama_server`, `llama_args` |
| `ollama` | Ollama's own API | `model`, `url` |
| `openai` | any `/v1/chat/completions`: LM Studio, vLLM, llama-server, hosted APIs | `url`, `model`, `api_key_env`, `headers`, `extra_body` |
| `command` | any program that prints the answer | `argv` |
| `python` | a function `module:name` or `file.py:name` | `factory`, `options` |

- `command` runs with no shell. The prompt goes on stdin, or into `argv` where you write
  `{prompt}` (at most 30,000 characters). `{model}` is replaced too.
- On Windows, a `.cmd` or `.bat` program never gets `{prompt}`: cmd.exe could run text from
  your files as commands. Use stdin, or point `argv` at the real program.
- `python`: the function gets `options` and returns an object with
  `chat(system, user, schema=None, max_output=1500)`. Start from
  [examples/providers/ollama_http.py](../examples/providers/ollama_http.py).
- Every type also takes `model`, `timeout`, `description`, `context_tokens` and the three
  flags below.
- `context_tokens` is the model's context window, 4096 to 1048576 (default 12288).
  agenthandoff starts `llama-server` with it and asks Ollama for it. For other types, set it
  to match the server. Each call then sends at most half the smallest window.

The three flags:
- `uses_local_gpu`: takes the GPU lock and the GPU-busy check. True for `llamacpp` and for
  a local `ollama` or `openai` URL.
- `remote`: may send file text off this machine. True for a non-local URL, and for every
  `command` and `python` provider unless it says `"remote": false`. **A remote provider is
  refused unless you set `allow_remote true`.**
- `supports_schema`: can be asked for a JSON shape. If false (the default for `command`),
  the shape is asked for in the prompt, and a reply that is not JSON is retried once.

Keys never go in config. `api_key_env` names the environment variable that holds the key.
`"api_key"` in config is refused.

More examples:

```json
"ollama-oai": {"type": "openai", "url": "http://localhost:11434", "model": "gemma4:e4b-it-q4_K_M",
               "extra_body": {"reasoning_effort": "none"}},
"ollama-cli": {"type": "command", "argv": ["ollama", "run", "--think=false", "{model}"],
               "model": "gemma4:e4b-it-q4_K_M", "remote": false, "uses_local_gpu": true},
"agy": {"type": "command", "argv": ["agy", "--mode", "plan", "--sandbox", "-p", "{prompt}"], "timeout": 320}
```

A cloud CLI (like `agy`) needs `allow_remote true`. It is slower, so put it last in `provider`.

### Model troubleshooting

| You see | What to do |
|---|---|
| `agenthandoff is not set up yet` | Run `agenthandoff setup`. |
| `route.allow_model is off` | `agenthandoff config set route.allow_model true`. |
| `llama-server was not found` | Run setup again, or set `llama_server` to its path. |
| `llama-server stopped while starting` | Often the model does not fit. Try the smaller model, or add `-ngl 20` to `llama_args`. |
| `cannot reach http://localhost:11434` | Start Ollama (or LM Studio's server), or check `url`. |
| `... is not downloaded` / `not loaded` | `ollama pull NAME`, or load the model in LM Studio. |
| `the GPU is busy with another program` | Wait, or set `ignore_gpu_busy` to run anyway, slowly. |
| `another agenthandoff job is using the GPU` | A digest is running. A lock left by a crash clears itself. |
| `RECALL CHECK: FAILED` | Read the files yourself for anything important, or lower `chunk_tokens`. |
| It takes minutes | Each chunk is one model call: 15 to 20 s on a laptop GPU, about a minute on a CPU. |
| `spent the reply thinking and wrote no answer` | See "Thinking models" below. |
| `may send file text off this machine` | A local CLI: add `"remote": false`. A cloud service you accept: `allow_remote true`. |

`llama-server stopped while starting` shows the end of `~/.agenthandoff/logs/llama-server.log`.

**Thinking models** (such as Gemma 4) can spend the whole reply thinking:
- Ollama's OpenAI endpoint: add `"extra_body": {"reasoning_effort": "none"}`.
- Ollama itself: use type `ollama`, which turns thinking off.
- llama.cpp: update `llama-server`. agenthandoff passes `--reasoning-budget 0` when the
  server supports it.
- LM Studio: turn thinking off for the model.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e .                  # Windows: .venv\Scripts\pip install -e .
.venv/bin/python -m unittest discover -s tests
```

The tests use a fake model and local stand-in servers. They need no model, GPU or network.

## Platforms

- **Windows 11**: tested, with and without a local model.
- **macOS** (Apple Silicon): tested by CI. Not yet tried with a real model.
- **Linux**: should work, not tested.
- Python 3.10 to 3.13.

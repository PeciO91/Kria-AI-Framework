# AGENTS.md

<!-- >>> projectmem bridge >>> -->
## projectmem (MANDATORY)

This project uses projectmem for persistent memory + workflow rules.

This project is registered with projectmem as **project**.
Pass `project="project"` on any projectmem tool call.

SESSION START — call these three MCP tools, in this order, BEFORE
answering ANY question about this project:

  1. `get_instructions()` — loads the project's mandatory workflow
     rules. Without this you will not know how to log work
     correctly, when to use `add_note` vs `add_decision`, or how
     the event log is structured.
  2. `get_summary()` — loads project content. Do NOT answer from
     conversation history or by re-reading package.json / README /
     source files.
  3. `get_project_map()` — loads structural layout when relevant.

BEFORE modifying ANY file:
  - Call `precheck_file(file_path)` — check failure history first.

DURING work — use MCP write tools for events and derived memory.
NEVER edit `.projectmem/summary.md` or `.projectmem/events.jsonl`
directly via filesystem write:
  - On a bug discovery → `log_issue(summary, location)`.
  - After each fix attempt → `record_attempt(summary, outcome)`.
  - After confirmation → `record_fix(summary)`.
  - On a design choice → `add_decision(summary)`.
  - On a gotcha / setup detail → `add_note(summary)`.
  - When a new decision REPLACES an older one → pass
    `supersedes="<old event id>"` to `add_decision`. The log stays
    append-only; the old decision is tagged retired and drops out of
    summary.md, so the summary never shows two answers to the same
    question. Get ids from `get_summary()` or `search_events(query)`.

`summary.md` is auto-generated from `events.jsonl`; use MCP write tools
for events and derived memory, never manual edits.

`PROJECT_MAP.md` and `plan.md` are authored files, not derived from events.
You MAY edit `.projectmem/PROJECT_MAP.md` and `.projectmem/plan.md`
directly after `precheck_file`; follow `get_instructions()` for setup
and intent updates. Log architectural decisions and durable context
separately through MCP. After changing the map's Project purpose, use
`pjm regenerate` or an MCP write tool to refresh `summary.md`.

Do not re-scan source files when MCP tools can give you the same
answer in a few hundred tokens instead of thousands. This is not optional.
<!-- <<< projectmem bridge <<< -->

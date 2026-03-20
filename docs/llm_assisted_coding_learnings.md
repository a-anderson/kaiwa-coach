# LLM-Assisted Coding Learnings

## Purpose

One goal I had for this project was to learn how to work with an LLM coding assistant as part of a real build, not just for isolated code snippets.

I wanted to get better at:

- scoping requests so the assistant could make useful changes without breaking surrounding behaviour
- using the assistant for debugging and iteration
- treating the assistant like a collaborator that still needs review, tests, and clear constraints

## What Worked Well

### Small, bounded requests

The best results came from asking for one thing at a time:

- add a UI toggle
- update a specific callback
- fix a failing test
- refactor one section for readability

This made it much easier to spot mistakes quickly and keep momentum.

### Supplying real tracebacks

When something failed, pasting the full traceback usually led to faster fixes than describing the error in general terms. The assistant could work from concrete evidence and target the right part of the code.

### Asking for review-style feedback

Asking whether a change would pass review by senior engineers was useful. It often surfaced:

- missing tests
- unclear helper names
- brittle assumptions
- framework-specific edge cases

This was a good way to improve quality without waiting for a real PR review.

### Asking for tradeoffs before implementation

The most useful exchanges happened when I asked:

- what are the options?
- what is the simplest approach?
- what are the tradeoffs?

That helped avoid overengineering, especially in UI work where several approaches were possible.

## What Did Not Work Well

### Broad requests produced messy solutions

If I asked for a large feature without enough constraints, the first solution was sometimes technically functional but too complex or fragile. This happened most noticeably with Gradio theming and CSS overrides.

The fix was to narrow the request and ask for the simplest likely-working approach first.

### UI intent is hard to communicate in text only

Small visual issues (borders, dropdown containers, image controls, alignment) were easy to misunderstand. In those cases, tight feedback loops and concrete descriptions of what changed on screen mattered more than abstract styling requests.

### Framework version differences matter

A solution that looks valid in documentation may not work in the pinned version. For example, constructor arguments available in some Gradio versions were not available in the version used here. Version-specific constraints need to be stated early.

## Choosing a Mode

Claude Code offers different interaction modes — from requiring approval on every change to accepting edits automatically. Choosing the right mode for the task made a meaningful difference.

### Planning mode: use when the details matter

Planning mode (where the assistant proposes changes and waits for approval before applying them) worked best for:

- tasks where learning was the goal — reviewing proposed changes in real time, asking questions as they arose, and understanding the reasoning behind architectural decisions accelerated learning significantly compared to just reading the output
- high-risk changes where a mistake would be costly or hard to reverse
- unfamiliar domains where close attention to what is being done is important

The back-and-forth is not friction — it is the point. For a learning project, planning mode provided an interactive, real-time conversation about why decisions were made, which is difficult to replicate by reading finished code.

### Auto-accept mode: use when the output matters more than the details

Auto-accept worked best for:

- areas where the goal was the end result, not understanding the implementation
- low-risk, routine, or boilerplate work
- prototyping work that is expected to be thrown away

The tradeoff is reduced visibility into what was produced. This is manageable with clear upfront standards (see the CLAUDE.md section below) and checkpoint reviews.

### Switching modes mid-task

If a task changes risk level or scope mid-session — something that started as routine becomes more consequential — switch modes explicitly and state the reason in the prompt. The mode should match the task, not the session.

## Collaboration Patterns That Improved Results

### Use clarifying questions to scope a plan

For planning and problem-solving, a better pattern than writing a comprehensive upfront prompt was to give a brief summary of intent and then ask the assistant to ask clarifying questions until the scope was clear.

This worked better because:

- it is easier to answer focused questions than to anticipate everything upfront
- the questions often surfaced considerations or edge cases that had not been thought of, deepening understanding of both the problem and the solution space
- arriving at a shared, well-scoped plan before implementation meant fewer course corrections mid-task

The interaction itself — not just the output — was where much of the value came from.

### Ask for the simplest option first

This became one of the most useful working rules for the project.

It reduced:

- unnecessary complexity
- fragile hacks
- difficult-to-review diffs

### Require tests with changes

For callback-heavy UI work and orchestration changes, adding or updating tests with the code change helped catch output ordering bugs and regression issues quickly.

### Separate functional changes from refactors

A good pattern was:

1. make the feature work
2. confirm manually / run tests
3. refactor for readability
4. run tests again

This made it easier to identify whether a failure came from behaviour changes or refactoring.

### Use PR-style review prompts

Asking for a review mindset helped move from “it works” to “it is maintainable and reviewable”. For example:

```
If this branch were submitted as a PR for review by two senior Python engineers, would it pass? What would their comments be, and which points are blockers vs non-blocking comments?
```

### Use successive agents for PR reviews

A single agent reviewing its own recent work will miss things. Using a fresh agent or session at review checkpoints consistently surfaced issues the original session had not caught — including cases where a previous agent had declared something fixed when it was not, or where a fix had introduced a side effect or lacked test coverage.

The pattern that worked:

1. use one agent to implement and self-review
2. bring in a fresh agent to do a PR-style review
3. use the agent that identified a new issue to fix it
4. repeat until no blocking issues are found

Different agents will catch different things and miss different things. Playing their feedback against each other is more reliable than trusting any single review. Expect to go through more rounds than feels necessary — that is normal, not a sign something is wrong.

### Treat agent output like work from a junior-mid level engineer

Agents do not know everything, and they will make mistakes. A useful calibration: treat the output with roughly the same level of trust you would give a capable junior-mid level engineer. The work is often good, sometimes excellent, and occasionally wrong in ways that require experience to spot.

This framing also serves as a useful reminder: you are ultimately responsible for any output you produce or sign off on. The assistant accelerates the work; the judgement is still yours.

### Keep CLAUDE.md current

The project instruction file (CLAUDE.md or equivalent) is most useful as a living document, not a one-time setup. Two patterns that worked well:

- **Set standards upfront**: code quality bar, test requirements, architectural constraints, error handling rules. A good initial set of standards reduces the number of corrections needed later.
- **Add to it reactively**: when the same class of issue keeps appearing across sessions, or when a cluster of fixes is needed at once, that is a signal to encode the standard explicitly in CLAUDE.md so it does not need to be re-stated each time.

The file earns its value through iteration. Treat it like a project-specific style guide that evolves with the codebase.

## Quality Controls That Helped

- targeted tests during iteration (`poetry run pytest -q <file>`)
- broader test runs before calling a task done
- manual UI checks for user-facing changes
- regression tests for bugs found during manual testing
- refactoring after behaviour was stable, not before

## Practical Lessons for Future Projects

- Start with smaller requests than you think you need.
- Provide exact errors and traces whenever possible.
- Ask for options and tradeoffs before implementation if there is more than one reasonable path.
- For non-trivial tasks, give a brief intent summary and ask the assistant to ask clarifying questions before planning — it surfaces blind spots and produces a better-scoped plan than trying to write a comprehensive prompt upfront.
- Treat generated code as draft code that still needs review — calibrate trust to roughly junior-mid level engineer output.
- Keep the assistant aligned with explicit constraints (version, style, architecture, quality bar).
- Choose the mode to match the task: planning for learning or high-risk work, auto-accept for low-risk or routine work.
- Use fresh agents for review checkpoints; do not rely on a single session to catch its own mistakes.
- Update the project instruction file when the same issue recurs — encode the standard, not just the fix.

## How I Would Work Differently Next Time

- State framework/version constraints earlier for UI work.
- Ensure the proposed solution looks at the simplest method/s first, instead of complicated or workaround-heavy customisation.
- Add a short “acceptance criteria” checklist before implementing medium-sized changes.
- Be more deliberate about mode selection at the start of each task rather than leaving it at the session default.
- Plan review checkpoints in advance rather than deciding ad hoc when to bring in a fresh agent.

## Open Questions

- **When is good enough actually good enough?** In familiar domains, weighing the cost of different architectural decisions is straightforward. In unfamiliar ones, it is harder to know whether a flagged issue is a genuine blocker or an over-cautious suggestion. How to develop that judgement in new domains is an open area for refinement.
- **How much of multi-agent review overhead is real?** When successive agents identify many blocking issues, some may reflect genuine problems and others may reflect an agent trying to be thorough rather than accurate. Developing a reliable filter for this — especially in unfamiliar territory — is still a work in progress.

## Working Patterns and Human Factors

**Can deep work and LLM-assisted iteration coexist?**

The problem is not wait time or switching frequency — it is mindset disruption. Holding a hard problem in your head requires a particular mental state, and switching to another task breaks it regardless of how long the switch lasts. You cannot give proper attention to two things at once.

Running the assistant on longer uninterrupted tasks and using auto-accept for low-risk work reduces the cost significantly. But in planning mode, or on high-risk changes, the back-and-forth is inherent. The only alternative is to wait idle.

This feels structural rather than solvable through workflow adjustments. How to do genuine deep work while keeping LLM-assisted tasks progressing is an open question I do not have a good answer to yet.

**The loss of ambient attention**

A day of coding used to alternate cognitively demanding work with tasks that were necessary but easy — boilerplate, scaffolding, setup. The easy work was not just rest. It was a state where the foreground task was light enough that background processing could still run. That is where a lot of incidental discovery happened: noticing unexpected structure in the codebase, an error pointing to a deeper misunderstanding, duplicated logic, something that needed a conversation with someone from a different team. Many problem reframings came not from focused work but from that kind of peripheral attention while doing something easy.

LLM assistants remove most of that easy work and leave the cognitively demanding tasks. The burnout risk is obvious. The less obvious loss is the ambient attention.

Deliberately recreating it — scheduled review passes, intentional reading time — sounds reasonable but adds cognitive load rather than reducing it. For a substitute to work it would need to arise naturally, not become another thing to actively manage. Whether that ambient discovery can be preserved or replaced in an LLM-assisted workflow, and what the long-term cost is if it cannot, is an open question.

**Cognitive fatigue, blind acceptance, and the behavior change gap**

Intensive LLM-assisted work produces a recognisable degradation pattern: switching to TTS for long outputs rather than reading them → skipping large sections and reading only the opening → skimming headings → accepting changes without reviewing the output. Each step feels like a reasonable response to energy levels in the moment. By the end, the quality filter is gone.

Treating each stage as a signal rather than a preference is more useful than scheduled breaks, which override flow states and do not account for how much energy and engagement vary across a day or between people. Noticing the switch to TTS is an early indicator; skipping sections is a stronger one.

The harder problem is that self-awareness is itself degraded when fatigued. The solutions are not mysterious — take breaks, review output, work intentionally, stop when you stop being engaged. The difficulty is doing them consistently. That gap is not unique to this context, and willpower is not a reliable mechanism for closing it. Meaningful change requires support, structure, and an environment where the right behaviour is also the easiest one. What that looks like in practice, for individual habits and for team settings, is something I am still working through.

## Summary

An LLM coding assistant is most useful when the interaction is structured. Clear scope, concrete evidence, tests, and review-style prompts consistently produced better results than broad “build this feature” requests. Choosing the right mode for each task, maintaining a current project instruction file, and using fresh agents for review checkpoints are the habits that had the most impact on output quality.

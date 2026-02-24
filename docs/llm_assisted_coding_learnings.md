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

## Collaboration Patterns That Improved Results

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
- Treat generated code as draft code that still needs review.
- Keep the assistant aligned with explicit constraints (version, style, architecture, quality bar).

## How I Would Work Differently Next Time

- State framework/version constraints earlier for UI work.
- Ensure the proposed solution looks at the simplest method/s first, instead of complicated or workaround-heavy customisation.
- Add a short “acceptance criteria” checklist before implementing medium-sized changes.

## Summary

An LLM coding assistant is most useful when the interaction is structured. Clear scope, concrete evidence, tests, and review-style prompts consistently produced better results than broad “build this feature” requests.

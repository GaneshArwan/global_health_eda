# Project Completion & Security Protocol

Whenever finishing ANY project or codebase, you MUST follow these completion rules before considering the task complete and present them to the user in a checklist-like format.

- **Step 1:** Security Skill Loop — Run 5 security skills/checks. Fix any issues found, and then re-run the 5 security checks. Loop this process until 100% of the issues are fixed.
- **Step 2:** NPM Audit — If the project uses NPM, run `npm audit --audit-level high` and fix any high/critical vulnerabilities found.
- **Step 3:** Professional Language Formatting — Ensure the codebase (especially user-visible text) contains no "AI-ish" words. Use only professional wording. (Does not apply to AI prompts/guardrails).
- **Step 4:** Guardrails — Always make and implement guardrails for the project.

# Contributing to Kura

Thanks for wanting to improve Kura. This project moves fast, so a clear contribution process helps everyone ship better changes with less friction.

## Before You Start

Please make sure you can run the app locally first.

### Prerequisites

- Node.js 18+
- Rust stable toolchain
- Windows users: Visual Studio Build Tools with C++ workload

### Setup

```bash
npm install
npm run dev
```

## Development Workflow

1. Create a focused branch for one feature or fix.
2. Make changes in small, reviewable commits.
3. Run checks before opening a PR.
4. Update docs when behavior changes.

## Required Checks

Run these before opening a pull request:

```bash
npm run test
```

For Rust-heavy changes, also run:

```bash
cd src-tauri
cargo check
```

## Project Conventions

### Frontend

- React + TypeScript + Vite
- Keep UI states explicit and user-friendly
- Preserve evidence traceability between extracted values and PDF highlights

### Backend

- Tauri command signatures should stay stable and clear
- Prefer strong error messages over silent failures
- Keep database writes transactional where possible

### Documentation

- If you add or change user-facing functionality, update `README.md`.
- Keep language practical and human-friendly.
- For this repo, add a short entry in `Devlog.md` after finishing a task.

## Pull Request Checklist

- [ ] Feature/fix works end-to-end locally
- [ ] `npm run test` passes
- [ ] `cargo check` passes for backend changes
- [ ] Documentation updated (`README.md`, docs, or both)
- [ ] `Devlog.md` includes a concise task entry

## Issue Reports

When filing a bug, include:

- What you expected
- What actually happened
- Steps to reproduce
- OS + provider/model details
- Logs or screenshots when possible

## Suggestions and Feature Ideas

Open an issue with:

- Problem you are solving
- Proposed UX flow
- Any data/export/print constraints

## Code of Conduct

Be respectful and constructive. Good collaboration beats perfect code on day one.

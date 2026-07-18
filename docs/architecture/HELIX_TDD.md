# Helix — Technical Design Document (TDD) v1.0

Status: Draft for review. Once approved, this is the source of truth. Amendments get logged (see §11), not silently made.

---

## 1. Vision

### 1.1 Project Goals
- Build an owned, end-to-end AI stack: tokenizer → model → training → inference → product, all understood and controlled by us.
- Grow the model lineage incrementally (v0.1 → v1.x+), each version measurably better than the last.
- Build a platform that is model-agnostic by construction, so the "our own model" story is real, not marketing.
- Treat this as a multi-year research + engineering effort with real documentation, not a weekend hack.

### 1.2 Non-Goals
- Not competing with frontier labs (OpenAI, Anthropic, Google) on model capability, at any point in the visible roadmap.
- Not building desktop or mobile apps (web only, until explicitly revisited).
- Not optimizing for a fast demo or investor pitch — no capability is claimed until it's measured (§9).
- Not adding multi-agent orchestration, image generation, or voice until the core loop (tokenizer → model → chat) is solid and evaluated.

### 1.3 Long-Term Philosophy
- Ownership over convenience: every core capability should eventually run on Helix's own model, with external providers as optional, swappable adapters — never the foundation.
- Learning over shortcuts: implementations should be understood, not copy-pasted from tutorials without comprehension.
- Evidence over assumption: no version is "better" until the eval harness (Track A, A6) says so.
- Small and real beats big and imaginary: a working 10M-parameter model that does one thing honestly is worth more than a roadmap for a 70B model that never trains.

### 1.4 Success Criteria
- **v0.1 success:** a from-scratch tokenizer + transformer trains on owned infrastructure and produces locally-coherent text. Reproducible by anyone following the docs.
- **Platform success:** the web app runs fully against `MockModelProvider` with zero external API calls, and swapping to any other provider is a one-line config change.
- **Program success (ongoing):** every version has a documented eval score, and the trend line goes up.

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
                        ┌────────────────────────┐
                        │        Frontend         │
                        │   (Next.js web chat)     │
                        └───────────┬─────────────┘
                                    │ HTTP/WS
                        ┌───────────▼─────────────┐
                        │        Backend API        │
                        │  (Auth, Chat, Files, Tools)│
                        └───────────┬─────────────┘
                                    │
                        ┌───────────▼─────────────┐
                        │   ModelProvider Interface  │
                        │  (generate / stream / list)│
                        └───┬─────────┬─────────┬───┘
                            │         │         │
                 ┌──────────▼──┐ ┌────▼─────┐ ┌─▼────────────┐
                 │MockModel     │ │LocalHelix│ │ External      │
                 │Provider      │ │Provider  │ │ Provider(s)   │
                 │(dev/test)    │ │(our model)│ │(optional adapter)│
                 └──────────────┘ └────┬─────┘ └───────────────┘
                                       │
                        ┌──────────────▼─────────────┐
                        │      Inference Service       │
                        │  (loads checkpoint, serves)   │
                        └──────────────┬─────────────┘
                                       │
                        ┌──────────────▼─────────────┐
                        │        Model Registry         │
                        │ (checkpoints, versions, metadata)│
                        └──────────────┬─────────────┘
                                       │
                 ┌─────────────────────▼─────────────────────┐
                 │                Track A — Research             │
                 │  Tokenizer → Training Pipeline → Eval Harness  │
                 └───────────────────────────────────────────────┘
```

### 2.2 Component Interactions
- Frontend never talks to any model directly — always through the Backend API.
- Backend API never imports a vendor SDK (OpenAI, Anthropic, etc.) — only the `ModelProvider` interface.
- `LocalHelixProvider` is the only component allowed to touch the Model Registry / checkpoints directly.
- `ExternalProvider` implementations are thin adapters and are excluded from Track A entirely — they exist for comparison/fallback only.

### 2.3 Data Flow (chat message, general case)
1. User sends message → Frontend → Backend API.
2. Backend loads conversation context + memory → builds prompt.
3. Backend calls `ModelProvider.generate()`/`stream()` with the active provider (config-driven).
4. Provider returns tokens → Backend streams to Frontend → persists to DB.

### 2.4 Request Flow (platform, runtime)
`Frontend → Backend → Auth check → Rate/permission check → ModelProvider → Provider implementation → response → persistence → Frontend`

### 2.5 Training Flow (research, offline)
`Raw corpus → Data cleaning → Tokenizer training → Tokenized dataset → Training loop → Checkpoint → Eval harness → Model Registry entry (if it passes eval)`

### 2.6 Inference Flow (research → platform bridge)
`Model Registry checkpoint → Inference Service loads weights → exposes generate()/stream() → LocalHelixProvider wraps it → Backend consumes it exactly like any other provider`

---

## 3. Repository Structure

```
helix/
├── apps/
│   ├── web/                  # Next.js frontend — chat UI, auth pages, settings
│   └── api/                  # Backend service — REST/WS API, auth, persistence, tool routing
│
├── packages/
│   ├── model-provider/       # The ModelProvider interface + Mock/Local/External implementations
│   ├── ui/                   # Shared React components used only by apps/web
│   └── types/                # Shared TypeScript types/interfaces across apps and packages
│
├── research/
│   ├── tokenizer/            # Tokenizer training scripts, vocab artifacts, tests
│   ├── model/                # Transformer architecture code (config-driven, from scratch)
│   ├── training/              # Training loop, data loaders, checkpointing logic
│   ├── evaluation/            # Eval harness: fixed prompt sets, scoring scripts, perplexity tracking
│   └── experiments/           # Throwaway/exploratory notebooks and scripts — NOT production code
│
├── datasets/
│   ├── raw/                  # Original, untouched source data (gitignored, tracked via dataset versioning tool)
│   ├── processed/             # Cleaned/tokenized datasets ready for training
│   └── manifests/             # Dataset version manifests (what data, what hash, what license)
│
├── checkpoints/               # Gitignored — actual weights live in the Model Registry / object storage, not git
│   └── README.md              # Explains where real checkpoints live and how to fetch them
│
├── infra/
│   ├── docker/                 # Dockerfiles for each service
│   ├── ci/                     # CI/CD pipeline definitions
│   └── deploy/                 # Deployment configs (per environment)
│
├── scripts/                    # One-off operational scripts (migrations, data pulls, admin tasks)
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── config/
│   ├── base.yaml                # Shared defaults
│   ├── dev.yaml
│   ├── prod.yaml
│   └── model/                    # Per-model-version training configs
│
├── docs/
│   ├── architecture/             # This TDD and its amendments
│   ├── decisions/                 # ADRs (Architecture Decision Records) — one file per decision
│   ├── research-notes/            # Track A research logs, per experiment
│   └── roadmap.md
│
└── README.md
```

**Folder purpose, explicitly:**
- `apps/*` — deployable units. Nothing research-related lives here.
- `packages/*` — code shared across apps, importable, versioned internally.
- `research/*` — Track A. Never imported directly by `apps/*` — it produces artifacts (tokenizer, checkpoints) consumed via the Model Registry, not via direct code import. This boundary is what keeps research experimentation from destabilizing the product.
- `datasets/*` — raw is immutable once committed to a manifest; processed is regenerable from raw + a documented script.
- `checkpoints/` — deliberately empty in git; real weights are large binary artifacts and belong in object storage (S3-compatible) or a model registry tool, referenced by hash/version.
- `infra/*` — everything needed to build, run, and ship, kept separate from application code.
- `config/*` — no hardcoded config in code; every environment difference lives here.
- `docs/decisions/` — every non-trivial architecture choice gets one file, dated, with the alternatives considered and why they were rejected. This is what lets a future engineer (or Codex) understand *why*, not just *what*.

---

## 4. Technology Stack

| Layer | Choice | Justification |
|---|---|---|
| Frontend | Next.js (App Router) + TypeScript + Tailwind | Matches your existing stack from Muse — real reuse of skills, strong ecosystem, SSR/streaming support fits chat UX. |
| Backend | Fastify (Node/TypeScript) | Same language as frontend reduces context-switching; fast, minimal, good streaming support for token-by-token responses. |
| AI/Research framework | PyTorch | Standard for from-scratch model work; largest body of reference implementations to learn from; not tied to any vendor. |
| Tokenizer | Hugging Face `tokenizers` (Rust-backed) | Fast, well-documented, lets you train a real custom BPE tokenizer without reinventing the wheel — this is tooling, not the thing you're trying to learn, so use the good library. |
| Database | PostgreSQL | Relational integrity for users/chats/auth; battle-tested; Prisma already in your toolkit from Muse. |
| ORM | Prisma | Direct carryover from Muse experience — real leverage of what you already know. |
| Vector DB (future, memory/RAG) | Defer decision — evaluate pgvector (Postgres extension) first before adding a separate service | Avoids a whole extra piece of infrastructure until Track B actually needs semantic search; pgvector keeps it inside Postgres. |
| Package manager | pnpm (monorepo) | Matches Muse; efficient for monorepos with `apps/` + `packages/`. |
| Config system | YAML (config/) + environment variables for secrets | Human-readable, diffable in git, standard for ML configs; secrets never in YAML. |
| Experiment tracking | Weights & Biases (or self-hosted alternative like Aim if avoiding external dependency) | Standard in ML research for loss curves/sample tracking; pick self-hosted (Aim) if "own everything" philosophy should extend to tooling too — your call, flag it as an ADR. |
| Dataset versioning | DVC (Data Version Control) | Git-like versioning for large data files without bloating the repo; integrates with the `datasets/manifests/` folder. |
| Checkpoint/model registry | MLflow Model Registry, or a simple self-built registry (JSON manifest + object storage) for v0.1 | Don't adopt a heavy registry before you have more than one checkpoint worth registering — start with the simple manifest approach, graduate to MLflow when Track A has multiple model lineages. |
| Logging | Pino (backend), Python `logging` + structured JSON (research) | Structured logs from day one, not print statements — this is a coding-standards non-negotiable (§5). |
| Monitoring | Self-hosted: Grafana + Prometheus | Consistent with "own the stack" philosophy; avoids external SaaS dependency for core observability. |
| Testing | Vitest (TS), pytest (Python) | Standard, fast, well-integrated with respective ecosystems. |
| CI/CD | GitHub Actions | Repo is already the source of truth on GitHub; no reason to add another CI vendor. |
| Containerization | Docker | Standard, needed regardless of eventual deploy target. |
| Deployment | Defer to an ADR once Track B reaches B3 | Premature to lock this in before auth/DB exist — don't decide infrastructure you don't need yet (violates §9 "build simple before complex" if locked now). |

---

## 5. Coding Standards

- **Folder conventions:** feature-based grouping inside `apps/*`, not type-based (avoid a giant flat `components/` or `utils/` dumping ground).
- **Naming:** `camelCase` for TS variables/functions, `PascalCase` for components/types/classes, `snake_case` for Python, `kebab-case` for file names except React components (`PascalCase.tsx`).
- **File organization:** one exported concept per file where reasonable; index files only for public package APIs, not to hide sprawl.
- **Error handling:** no silent catches. Every caught error either handled meaningfully or re-thrown with added context. No bare `except:` in Python, no empty `catch {}` in TS.
- **Logging standards:** structured (JSON in prod, pretty in dev), leveled (debug/info/warn/error), no `console.log`/`print` in committed code outside scripts.
- **Configuration management:** no magic numbers/strings in code — goes in `config/*`; secrets only via environment variables, never committed.
- **Documentation requirements:** every package/module has a README stating purpose and public interface; every ADR-worthy decision gets a `docs/decisions/` entry.
- **Type safety:** TypeScript `strict: true` everywhere; Python uses type hints + mypy for research code, not "just get it running" untyped scripts once code leaves `experiments/`.
- **Testing expectations:** `packages/model-provider` and `apps/api` require unit tests for all public functions before merge; `research/*` requires at minimum a test that the tokenizer round-trips and the training loop runs one step without erroring — research code doesn't need product-level coverage, but it needs a smoke test.

---

## 6. Branching Strategy

- `main` — always deployable, always green CI. No direct commits.
- `develop` — integration branch; feature branches merge here first.
- `feature/*` — one feature/task per branch, e.g. `feature/tokenizer-bpe-training`.
- `release/*` — cut from `develop` when preparing a versioned release (e.g. `release/v0.1`), stabilization only, then merged to `main` and tagged.
- Research branches follow the same convention but prefixed `research/`, e.g. `research/a2-tiny-transformer`.

**Commit convention:** Conventional Commits — `type(scope): description`, e.g. `feat(tokenizer): add BPE training script`, `fix(api): correct stream chunking bug`, `docs(tdd): add risk section`. Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `research`.

---

## 7. Development Workflow

```
You (Product/Research direction)
        ↓
Claude (Architecture, planning, review — this document's owner)
        ↓
Codex-ready implementation prompt
        ↓
Codex (writes code, opens PR)
        ↓
GitHub (CI runs, PR sits for review)
        ↓
Claude reviews the diff against this TDD + the specific milestone's acceptance criteria
        ↓
Merge → next milestone planned
```

**Responsibilities:**
- **You:** set direction, make final calls on ADRs Claude flags as needing a decision, run/monitor training jobs, provide real judgment Claude can't (e.g. "this checkpoint sounds worse to me than v0.1 despite the eval score").
- **Claude:** owns this TDD and keeps it current, breaks milestones into Codex prompts, reviews PRs/commits against acceptance criteria, flags scope creep or premature complexity, maintains `docs/decisions/`.
- **Codex:** implements exactly what's specified, writes tests per §5, opens PRs with clear descriptions, does not make undocumented architecture decisions — if a spec is ambiguous, that's flagged back rather than guessed.
- **GitHub:** source of truth for code state; nothing is "done" until it's merged, not when Codex reports it wrote something.

---

## 8. Roadmap — Versioned Releases

**Helix v0.1 — "It runs"**
- Goal: custom tokenizer + tiny transformer trains end-to-end on owned infra; web platform runs fully on `MockModelProvider`.
- Exit criteria: model produces locally-coherent text (not facts); platform has working chat UI + auth + persistence with zero external API dependency; `docs/decisions/` has at least the tokenizer, architecture-size, and framework ADRs logged.

**Helix v0.2 — "It's real data"**
- Goal: better/larger dataset, cleaned pipeline, longer training run.
- Exit criteria: eval harness shows measurable perplexity improvement over v0.1's checkpoint on the same fixed eval set.

**Helix v0.3 — "It follows instructions"**
- Goal: instruction-tuned checkpoint; `LocalHelixProvider` wired into the platform as a selectable provider alongside Mock.
- Exit criteria: a user can select "Helix (ours)" in the chat UI and get a response from the actual owned model, end to end.

**Helix v0.4 — "It reasons a bit better"**
- Goal: targeted improvements (better data mix, longer context, or architecture tweak) validated by eval harness before/after.
- Exit criteria: every change shipped has a paired before/after eval score in `docs/research-notes/`.

**Helix v1.x+**
- Goal: to be planned once v0.4 exit criteria are met — deliberately not over-specified now, per §9 (don't plan complexity you don't need yet).

---

## 9. Research & Engineering Principles (never broken)

1. No premature optimization — correctness and clarity first, speed only after profiling shows it matters.
2. Prefer modular design — every component replaceable without touching its neighbors (this is *why* §2's ModelProvider boundary exists).
3. Every non-trivial architecture decision gets an ADR in `docs/decisions/` before implementation starts, not after.
4. Measure improvements — no version is called "better" without an eval harness score to back it up.
5. Build simple before complex — don't add a vector DB, a model registry tool, or a deployment pipeline before the milestone that actually needs it.
6. Research code (`research/experiments/`) is allowed to be messy; anything promoted out of `experiments/` must meet the coding standards in §5.
7. The platform (Track B) must always run and demo fully on `MockModelProvider` alone — if it can't, the provider abstraction has been violated somewhere.

---

## 10. Risks

**Technical risks**
- *Training instability / no visible progress:* mitigate with A6's eval harness built early, small-scale sanity checks before any long run, and a fixed compute budget so failed runs are cheap to notice.
- *Provider abstraction gets leaked around (someone imports a vendor SDK directly in `apps/api`):* mitigate with a lint rule/CI check that fails the build if vendor SDK imports appear outside `packages/model-provider`.

**Project risks**
- *Scope creep back into "compete with frontier labs":* mitigate by keeping §1.2 (non-goals) visible and re-reading it before any roadmap revision.
- *Stalling on planning/documentation instead of shipping v0.1:* this TDD is thorough by design, but it has a defined end — once this document is approved, the next output is Codex implementation prompts, not another planning document. Watch for the pattern of "one more design doc before we start," since that's the historical failure mode here.

**Scaling risks**
- *Dataset licensing/quality issues discovered late:* mitigate with `datasets/manifests/` requiring a license/source field from A1 onward, not retrofitted later.
- *Checkpoint storage growing unmanageably in git:* mitigated structurally — checkpoints never go in git per §3, always object storage + registry manifest.

---

## 11. Amendment Log

| Date | Change | Reason |
|---|---|---|
| (initial) | v1.0 created | Baseline TDD before v0.1 implementation begins |

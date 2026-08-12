# Octavian → Business-Grade Web Platform: Conversion & Scaling Plan

**Goal:** Turn Octavian from a single-user Streamlit monolith into a multi-tenant SaaS
that can serve thousands→millions of users, loads "like a top-end site," shows proper
loading UX, monetizes via subscriptions, and **loses zero features or quality** in the
process. **Stack constraint: free-tier infrastructure only** until revenue justifies
paid tiers.

**Status:** PLAN ONLY — approved for planning. Implementation begins per phase below.

---

## 1. Current-State Audit (what actually exists)

| Layer | Today | Verdict |
|---|---|---|
| **App UI** | Streamlit monolith — `main.py` (2,388 lines) with 16 sidebar tabs | Single-session, re-runs whole script per interaction, no real auth gate, not multi-user safe |
| **AI Chatbot** | `ai_chatbot.py` (5,956 lines) — the largest module | The crown jewel; must port 1:1 with streaming |
| **API Backend** | `api_backend.py` (757 lines, Flask) — 15+ endpoints: `/health`, `/api/market-data/<sym>`, `/api/multiple-quotes`, `/api/analyze/<sym>`, `/api/chat`, `/api/predictions`, `/api/v1/predict/advanced`, `/api/user/profile`, `/api/conversation/history`, `/api/analytics/dashboard`, `/api/admin/*` | Solid skeleton, but synchronous, in-process rate limiting, SQLite-backed |
| **Auth** | `auth_engine.py` — bcrypt hash/verify, register, authenticate, logout, onboarding | Workable; needs JWT sessions + token refresh for web |
| **DB** | SQLite via `database_manager.py` + `db_manager.py` (paper trading schema, user profiles, conversations, analytics) | Fine for 1 user; **must migrate to Postgres** for concurrency |
| **Tiers/Billing** | `backend/core/tiers.py` — FREE/PRO/INSTITUTIONAL/ENTERPRISE with rich `TierLimits` (API/min, analyses/day, chat/day, paper-trading capital, ML model gates, 13F access, email briefings) | **Already designed!** Just needs enforcement middleware + payment hookup |
| **Rate limiting** | `api_rate_limiter.py` — in-process, per-service limits, retry/backoff, TTL cache | In-process = per-process; needs **distributed** (Redis) for multi-replica |
| **LLM** | `financial_llm_engine.py` → **LM Studio localhost:1234** | Works locally; **must abstract to a provider router** (free cloud LLM APIs) for hosted multi-user |
| **Market data** | `data_sources.py`, `realtime_data_service.py`, `api_rate_limiter` w/ failover + caching (yfinance + fresh quote batch) | Good pattern; needs Redis-backed cache instead of in-memory |
| **Web frontend** | `frontend/` — **Next.js 16** scaffold: `app/` with 14 empty route folders (`chatbot`, `scanner`, `watchlist`, `portfolio`, `quant`, `paper-trading`, `simulation`, `strategy`, `financial-model`, `documents`, `news`, `profile`, `settings`, `auth/login`, `auth/register`), `layout.tsx`, `globals.css`, `page.module.css` | **Huge head start.** Routes are pre-named; pages are empty. This is the conversion target |
| **Deployment** | `Dockerfile` exists | Needs compose + prod build |
| **Tests** | 138 tests incl. 16-tab UI walkthrough (30s, down from 555s) | Solid regression net for the conversion |

**Key recent wins already banked:** OptionsEngine NN is lazy + disk-cached (70s→0.01s),
UI walkthrough 18× faster, zero emojis in production code, all 138 tests green.

---

## 2. Target Architecture (free-tier, multi-tenant)

```
                    ┌─────────────────────────────────────────────┐
                    │   Vercel (free) — Next.js 16 frontend        │
                    │   Landing · Auth · Dashboard · Feature pages │
                    │   Edge cache · CDN · ISR for public pages    │
                    └───────────────┬─────────────────────────────┘
                                    │ HTTPS (JWT bearer)
                    ┌───────────────▼─────────────────────────────┐
                    │   Render / GCP App Engine (free) — Python    │
                    │   Flask→FastAPI API  (api_backend.py + new)  │
                    │   - auth middleware (JWT, tiers)             │
                    │   - distributed rate limiting (Upstash Redis)│
                    │   - streaming SSE for chat & scans           │
                    └───┬──────────────┬──────────────┬───────────┘
                        │              │              │
        ┌───────────────▼───┐  ┌───────▼──────┐  ┌─────▼──────────────┐
        │ Neon Postgres      │  │ Upstash Redis│  │ LLM provider router │
        │ (free 512MB)       │  │ (free)       │  │ Groq·Gemini·DeepSeek│
        │ users·trades·chat  │  │ cache·limits │  │ ·OpenRouter (fallback)│
        │ ·sims·profiles     │  │ ·queues·SSE  │  └─────────────────────┘
        └────────────────────┘  └──────────────┘
                        │
        ┌───────────────▼──────────────────────────────┐
        │ Free-tier extras:                            │
        │ · Resend (email briefings, 3k/mo free)       │
        │ · Stripe (payments — no monthly fee)          │
        │ · Sentry free (errors) · UptimeRobot (pings) │
        │ · Excel/PPT engines stay pure-Python (shared) │
        └──────────────────────────────────────────────┘
```

**The golden rule:** every quant engine (M&A, LBO, DCF, futures, options, sims, grading,
news, ML) is already pure Python. **The UI is the only thing being replaced.**
Engines are ported *unchanged* — same functions, same tests, same outputs. Feature
parity is enforced by the existing 138-test suite plus a new parity harness.

---

## 3. The Free-Tier Stack (decision sheet)

| Need | Choice | Free tier | Notes |
|---|---|---|---|
| Frontend hosting | **Vercel** | Hobby: unlimited static/ISR, 100GB bandwidth, edge network | Next.js is already scaffolded |
| Backend hosting | **Render free** (web service, spins down on idle) or **GCP App Engine F1** | 750 hr/mo / F1 instance | Pick Render for simplicity first; App Engine if autoscale needed |
| Database | **Neon Postgres** | 512MB storage, autoscaling, branching, 100+ connections | SQLite→Postgres via SQLAlchemy/Alembic migrations |
| Cache + rate limit + queues | **Upstash Redis** | 256MB, 5k commands/day free | Distributed rate limiting, hot-cache quotes, chat history TTL, job queue |
| Auth | **Keep `auth_engine.py`** (JWT) phase 1; **Clerk free tier** phase 2 (social login, MFA) | 10k MAU free | Don't re-invent; existing auth works, add JWT middleware |
| Payments | **Stripe** | No monthly fee (only per-transaction ~2.9%+30¢) | No processor is "free" — Stripe is the standard; wire to existing `tiers.py` |
| Email | **Resend** | 3,000 emails/mo free | Daily briefings (already a tier feature), password resets |
| LLM | **Provider router** (new `llm_router.py`): Groq (free), Google Gemini (free), DeepSeek (cheap), OpenRouter free credits; **keep LM Studio as local fallback** | Free credits/quotas each | `financial_llm_engine.py` gets an abstraction layer; zero code changes downstream |
| Market data | **Keep** `data_sources.py` multi-provider + failover; move cache to Redis | — | Already correct architecture, just distributed |
| Monitoring | **Sentry free** + **UptimeRobot free** | 5k errors/mo, 50 monitors | Error alerts before users notice |
| Analytics | **Vercel Analytics** / Plausible free | — | Conversion funnels for the business |

---

## 4. Conversion Roadmap (phased, zero-feature-loss)

### Phase 0 — Foundations (1–2 weeks) *[partially done this session]*
- ✅ Lazy + disk-cached OptionsEngine NN (done)
- ✅ 16-tab UI walkthrough, 138 tests green (done)
- ⬜ **Loading UX in Streamlit now** (user request, do while web app builds):
  - Custom CSS skeleton/spinner overlay (`st.spinner` + `st.markdown` CSS pulse)
  - Progress bar for long jobs (scans, backtests, model builds)
  - "Streaming" placeholder in chat while awaiting LLM
- ⬜ **API contract freeze:** document every endpoint + payload in `API_CONTRACT.md`
  so the web app and tests share one source of truth
- ⬜ Inventory all `st.session_state` usage → define the server-side session model

### Phase 1 — Backend productionization (2–3 weeks)
- Migrate `database_manager.py`/`db_manager.py` → **SQLAlchemy + Alembic on Postgres (Neon)**;
  add `user_id` scoping + RLS-ready schema; keep SQLite adapter for dev/tests
- Add **JWT auth middleware** (login/refresh/logout) using existing `auth_engine.py` hashing
- Enforce `backend/core/tiers.py` limits **server-side** (Redis counters, 401/429 with retry-after)
- Move `api_rate_limiter.py` counters to Upstash Redis (works across replicas)
- Add **SSE streaming** to `/api/chat` + `/api/analyze` (chatbot already generates progressively)
- Structured error envelope: `{error: {code, message, retry_after?, details?}}` everywhere
- Pagination + caching headers on list endpoints; `ETag`/`Cache-Control` on quotes

### Phase 2 — Frontend shell & design system (2–3 weeks) — *"HTML/CSS styled website"*
- Build the **Octavian design system** in `frontend/`: tokens (colors, type scale, spacing),
  dark institutional theme, `globals.css` overhaul — **no UI library, hand-rolled CSS**
  (fast, unique, zero deps)
- Landing page (`/`): hero, product tour, pricing (from `tiers.py`), sign-up CTA — ISR + SEO
- Auth pages (`/auth/login`, `/auth/register`) wired to backend JWT
- App shell: sidebar nav (16 routes already exist), topbar, user menu, tier badge
- **Loading UX system (web):** global `PageSkeleton` component, per-feature skeletons,
  button spinners, route transition progress bar, chat typing indicator, skeleton
  tables/charts — every async surface shows a graceful loading state
- Healthcheck: app boots, auth round-trip works end-to-end

### Phase 3 — Feature migration waves (4–8 weeks)
Port each Streamlit tab → Next.js page + API route. **Engines untouched; UI thin.**
Ordered by value → effort:

| Wave | Feature | Streamlit source | Web page |
|---|---|---|---|
| 3a | **AI Chatbot** (streaming, reasoning expander, tools) | `ai_chatbot.py` | `/chatbot` |
| 3b | **Dashboard + Watchlist + Scanner** | `main.py`, `watchlist_dashboard.py`, `market_scanner.py` | `/`, `/watchlist`, `/scanner` |
| 3c | **Paper Trading + Simulation Hub** | `paper_trading_ui.py`, `simulation_viewer.py` | `/paper-trading`, `/simulation` |
| 3d | **M&A / LBO / DCF / Financial Model + Excel + PPT** | `mna_model_engine.py`, `lbo_model_engine.py`, `financial_model_generator*.py`, `spreadsheet_generator.py`, `presentation_generator.py`, `ib_excel_engine.py` | `/financial-model` |
| 3e | **Quant Portal, Strategy Lab, News/Intelligence** | `quant_portal.py`, `strategy_research_lab.py`, `news_dashboard.py` | `/quant`, `/strategy`, `/news` |
| 3f | **Profile, Settings, Docs, remaining** | `trader_profile.py`, `document_analyzer.py` | `/profile`, `/settings`, `/documents` |

**Each wave ships with:** parity tests (same inputs → same outputs vs. Streamlit golden
files), the tab walkthrough extended to the web pages, and a loading-state review.

### Phase 4 — Monetization (1–2 weeks)
- **Stripe Checkout + Customer Portal** for PRO/INSTITUTIONAL/ENTERPRISE
  (pricing already defined in `tiers.py`)
- Webhook → update user tier in Postgres; Redis tier cache
- Enforce gates in middleware + hide locked features in UI with upgrade prompts
- Free tier caps: 5 analyses/day, 10 chat/day, $100k paper capital, etc. (already in `tiers.py`)
- Email briefings via Resend for paid tiers

### Phase 5 — Scale-out & hardening (ongoing)
- Horizontal replicas (Render/App Engine autoscale) — Redis makes rate limits + cache
  distributed-safe
- Heavy jobs (backtests, model builds, Excel/PPT) → **job queue** (Upstash Q / Redis +
  worker) so HTTP stays fast; results delivered via SSE/polling
- CDN edge caching for public/static + ISR for landing
- Sentry alerts, UptimeRobot, structured logs; load-test with k6/locust (free)
- Cost dashboard: per-user token budget caps on LLM router to keep free-tier viable

---

## 5. Performance Engineering (the "ultra quick" requirement)

### Already done this session
- OptionsEngine NN: 70s→**0.01s** (lazy + disk cache)
- UI walkthrough: 555s→**30s**; full suite: ~150s→**38s**

### Web app targets
| Metric | Target | Mechanism |
|---|---|---|
| Landing page LCP | < 1.5s | ISR, edge CDN, no client JS on hero |
| App shell (TTI) | < 2s | Code splitting, route-level lazy chunks |
| Feature page load | < 1s skeleton, data streams in | Suspense + skeleton + SSE |
| Chat first token | < 1.5s | Streaming SSE + provider router fast path |
| Quote/chart fetch | cached < 50ms | Redis hot cache (30–60s TTL) + stale-while-revalidate |
| Scans/backtests | backgrounded | Job queue; spinner + progress SSE |

### Ongoing rules (applied to every ported feature)
1. **No blocking I/O on the main thread** — async HTTP, background jobs for >2s work
2. **Cache everything immutable** — quotes, index snapshots, 13F filings, models
3. **Lazy-load heavy modules** — import on first use (like the OptionsEngine fix)
4. **Stream long responses** — never make users wait for a full payload
5. **Database: index every `user_id`/`symbol`/`timestamp` column**, batch writes,
   connection pooling
6. **Measure every page** with Vercel Analytics + a timing budget table in CI

---

## 6. Loading UX Specification (user requirement — applied to both apps)

| Scenario | UX |
|---|---|
| Initial page load | Branded splash/skeleton (logo + pulse), then content fade-in |
| Tab / route change | Top progress bar + content skeleton of the destination's shape |
| Long job (scan, backtest, model) | Deterministic progress bar (0–100%) + ETA, job stays alive if user navigates |
| AI chat | Typing indicator + **streamed tokens** appear live (already the chatbot's design) |
| Excel/PPT generation | "Building workbook…" step list (formulas → sheets → styling) |
| Data refresh | Stale-while-revalidate: show old data with a subtle "updating…" chip |
| Errors/timeouts | Graceful inline error + retry button + `retry_after` countdown — never a blank page |

**Streamlit now (interim):** a `ui_loading.py` helper — CSS pulse skeletons for
tables/charts, `st.status`/`st.progress` for jobs, and an `@st.cache_data`-friendly
spinner wrapper. Ships in Phase 0 so users feel the speed immediately while the web
app is built.

---

## 7. Multi-Tenancy & Security (thousands→millions)

- **Every DB query scoped by `user_id`** (paper trading accounts, portfolios, profiles,
  conversations, simulations). Add tenant checks to existing methods; Postgres RLS as belt-and-suspenders
- **Replace in-process singletons** (`get_*_engine()`) with request-scoped factories where
  state is per-user; keep true singletons only for stateless caches
- **JWT:** short-lived access + refresh tokens, httpOnly cookies, CSRF protection,
  `require_api_key` kept for public endpoints
- **Rate limits per user + per IP** (Upstash sliding window), tier-aware 429s
- **Secrets:** all keys to env vars / Neon & Vercel secrets; **remove any hardcoded
  creds** (audit `grep -r "api_key\|password\|secret"`)
- **Input validation** on every endpoint (symbol allowlist, numeric bounds, length caps)
- **DoS protection:** request size limits, per-user concurrency caps, queue admission control

---

## 8. Feature-Parity Guarantee (the "no quality loss" contract)

1. **Engines are shared, not rewritten.** M&A/LBO/DCF/futures/options/sims/grading/news/ML
   all stay identical Python modules. The web app calls them through the API.
2. **Golden-file parity tests:** run every engine on fixed inputs, hash outputs
   (Excel bytes, PPT bytes, JSON), assert web-vs-Streamlit identical.
3. **Excel/PPT output is byte-stable** — the same institutional-grade formatting
   (IB typography, no empty space, native charts, dynamic numbers) is preserved because
   `ib_excel_engine.py`/`presentation_generator.py` are untouched.
4. **Existing 138 tests** keep passing after every wave (CI gate).
5. **UI walkthrough** extended to web pages: every route renders without exceptions,
   every async surface shows a loading state.

---

## 9. Risk Register & Mitigations

| Risk | Mitigation |
|---|---|
| Free-tier limits hit at scale | Phase 5 cost dashboard; upgrade only when revenue covers it; LLM token budgets per user |
| Render free spins down on idle | Keep-alive ping (UptimeRobot every 5 min); App Engine F1 as no-spindown alternative |
| LLM cost blowup | Provider router with per-model budgets, response caching, cheaper models for simple queries |
| Yahoo market-data rate limits | Existing failover + Redis caching + graceful stale-with-timestamp labeling |
| Feature regression during port | Golden-file parity + 138-test gate per wave |
| Single-developer bandwidth | Waves ordered by value; API contract freeze prevents rework |
| Payment/compliance | Stripe handles PCI; ToS/onboarding already exist in `terms_of_service.py` |

---

## 10. Milestones (12-week target)

| Week | Milestone |
|---|---|
| 1–2 | Phase 0: Streamlit loading UX live; API contract frozen |
| 3–5 | Phase 1: Postgres migration, JWT auth, tier enforcement, streaming, Redis |
| 6–8 | Phase 2: Design system, landing, auth, app shell, loading system |
| 9–12 | Phase 3 waves 3a–3f: chatbot → dashboard/watchlist/scanner → paper trading/sims → M&A/LBO/DCF+Excel+PPT → quant/strategy/news → profile/settings |
| 12–14 | Phase 4: Stripe + tiers live; onboarding + briefings |
| 15+ | Phase 5: autoscale, job queue, monitoring, load tests |

**Exit criteria for "business launch":** paid checkout works, tier gates enforced,
chat streams, all 16 features live with parity, page loads < 2s, loading UX on every
async surface, 99% uptime (UptimeRobot).

---

## 11. Immediate Next Actions (what happens now)

1. **Phase 0 loading UX in Streamlit** (small, do it first — user-visible win)
2. Freeze `API_CONTRACT.md`
3. Scaffold `llm_router.py` (provider abstraction) so hosted LLMs are drop-in
4. Begin Phase 1: Postgres schema + Alembic bootstrap (dev keeps SQLite)

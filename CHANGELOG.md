# Changelog

All notable changes to the Episteme (QILA) project will be documented in this file.

## [v1.0.0] - 2024-05-24 (Production Release)

### Added
- **Session Isolation & Auth**: Implemented a cookie-based `SessionManager` in `app.py`. Every browser tab or user now gets a fully isolated session securely tied to an auto-expiring cookie (1-hour TTL).
- **SQLite Persistence**: Replaced volatile in-memory dicts with a robust SQLite database adapter (`qila/persistence.py`). Memories, ghost units, and full conversation histories now survive server restarts natively, using WAL mode for concurrent access.
- **Structured Logging**: Introduced centralized, colorized logging via `qila/logger.py`. Output is directed both to the console and to `episteme.log`.
- **API Key Security**: Completely removed hardcoded API keys. The system now strictly relies on a `.env` file handled via `python-dotenv`.
- **Mobile Responsiveness**: Upgraded the workspace UI with CSS media queries (1200px, 768px, 480px) to seamlessly support tablets and mobile devices.
- **SSE Error Boundaries**: Added a 90-second timeout guard and a "Cancel Analysis" button to the frontend streaming logic, gracefully handling network dropouts or LLM timeouts.
- **Landing Page**: Added a brand-new cyber-brutalist landing page (`landing.html`) at the root `/` endpoint to explain the KAIROS and MAKS architectures to new users before they enter the workspace at `/app`.
- **Dockerization**: Shipped a production-ready `Dockerfile` and `docker-compose.yml` that configures a Python 3.11-slim environment, caches dependencies, and wires up persistent volumes.
- **Comprehensive Test Suite**: Added 43 unit and integration tests across 3 files (`test_kairos.py`, `test_maks.py`, `test_api.py`) verifying core mathematical bounds, memory transitions, and FastAPI endpoints.

### Changed
- **Pipeline Performance**: Refactored the KAIROS pipeline to heavily utilize `concurrent.futures.ThreadPoolExecutor`. Concurrent network I/O for consistency samplings and gradient perturbations reduced analysis time from ~5 minutes to ~25-56 seconds.
- **Background Polling Fix**: Fixed a frontend background polling issue where missing session cookies caused the server to eagerly spin up thousands of orphan sessions. 

## [v0.1.0] - Initial Prototype
- Initial integration of KAIROS (epistemic scoring via H, G, C signals) and MAKS (adaptive memory).
- Basic Server-Sent Events (SSE) streaming architecture.
- Cyber-brutalist / Dark Neo-brutalist UI implementation.

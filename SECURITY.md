# Security Policy

## Supported versions

Dupless Video is pre-1.0; security fixes land on `main` and ship in the next release. The most
recent `0.1.x` release is the only supported line.

| Version | Supported |
|---------|-----------|
| latest `0.1.x` | ✅ |
| older          | ❌ |

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report privately through GitHub's **[Security Advisories](https://github.com/Acr86/dupless-video/security/advisories/new)**
("Report a vulnerability"). That opens a private channel with the maintainer.

Please include, where possible:

- the affected version / commit,
- a description and impact,
- steps to reproduce or a proof of concept,
- any suggested remediation.

### What to expect

- Acknowledgement within **7 days**.
- An initial assessment and severity within **14 days**.
- Coordinated disclosure once a fix is available; credit is offered unless you prefer to remain anonymous.

## Scope notes

Dupless Video runs **locally** and reads a user's own media; it has no network service and stores no
credentials. The most relevant risks are therefore:

- handling of **untrusted media files** (a malformed video must be *skipped and reported*, never crash
  or execute) — this is a core design principle (see `CLAUDE.md`),
- the **third-party binaries** it invokes (ffmpeg / ffprobe / fpcalc) and the Python dependency chain
  (tracked by `pip-audit` and Dependabot),
- the **desktop installer** artifact.

Findings in these areas are in scope. General reports about upstream ffmpeg/torch issues are best
filed with those projects, though we welcome a heads-up if Dupless Video is exploitable through them.

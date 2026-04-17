# Security policy

## Scope

This policy covers security issues in the **TrueMapData** Python library and CLI published from this repository and on [PyPI](https://pypi.org/project/truemapdata/), including code that reads or writes TMD height-map data, visualization, and export paths.

Reports should target vulnerabilities **in this project’s code or its documented usage**. Third-party instrument firmware, closed-source vendor software, or issues that only apply to external tools without a clear link to this repository are generally out of scope unless they involve how this library processes untrusted inputs.

## How to report

**Preferred:** use [GitHub private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) for the canonical repository:

https://github.com/ETSTribology/TrueMapData

Repository maintainers should ensure **Private vulnerability reporting** is enabled under **Settings → Code security and analysis** if it is not already.

**Alternative:** if you cannot use GitHub’s reporting flow, email **antoine@antoineboucher.info** with a clear subject line (for example `[TrueMapData security]`) and enough detail to reproduce or assess the issue. Do not send exploit code as unsolicited attachments; describe steps or use a private channel if agreed with maintainers.

## Supported versions

Security fixes are considered for **release lines that still receive maintenance** and for **Python versions exercised in CI** (currently **3.8 through 3.12**; see `.github/workflows/test.yml`). Very old Python runtimes or unmaintained release tags may not receive backports.

## What to include

- Affected component (CLI command, module, version or commit).
- Steps to reproduce, or a minimal proof of concept.
- Impact (e.g. arbitrary code execution, path traversal, unsafe deserialization) if known.

## Response

Maintainers aim to **acknowledge** credible reports within a **few business days**. Timelines for fixes depend on severity and release planning; reporters will be kept informed when contact information is available.

Thank you for helping keep users of TrueMapData safe.

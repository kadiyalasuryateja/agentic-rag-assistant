# Engineering Runbook

## Deployments

Production deployments run through the CI/CD pipeline after all regression tests pass. Deployments are frozen on Fridays after 2 PM and during company holidays. Every deployment must have a rollback plan.

## Incident response

Severity 1 incidents page the on-call engineer immediately and require an incident commander within 15 minutes. A blameless postmortem is due within 5 business days of any Severity 1 or Severity 2 incident.

## Code review

Every pull request needs at least one approving review from a code owner. Pull requests larger than 400 changed lines should be split into smaller changes.

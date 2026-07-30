"""Scheduled background work.

Separate from `services/`: a service is called by a request, a job is called
by the clock. Keeping them apart makes it obvious which code paths have no
user waiting on them — and therefore which must never raise.
"""

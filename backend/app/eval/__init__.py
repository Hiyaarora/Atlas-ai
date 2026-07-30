"""Retrieval evaluation: golden set, metrics, and the harness that runs them.

Kept out of `app/services` on purpose. Nothing the application serves depends
on this package — it is a measuring instrument, imported by the CLI harness and
by tests, never by a request path.
"""

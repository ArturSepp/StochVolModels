"""Registered producers and provenance checks for the documentation exhibits.

``registry.json`` lists every image the documentation displays, with its class, consuming pages,
producer, inputs and conventions. ``run.py`` regenerates the complete bundle into a fresh
directory outside the checkout, lists the registry, and verifies the committed previews against
``docs/images/analytics_manifest.json``. ``validate.py`` reads a generated bundle back. See
``docs/documentation_standard.md`` for the publication rules.
"""

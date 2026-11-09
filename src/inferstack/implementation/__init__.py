"""Bounded implementer handoff: a source idea becomes an immutable, validated candidate (harness H3)."""
from inferstack.implementation.handoff import (ArtifactLocked, AttemptLimit, CommandImplementer, ImplementationDesk,
                                               ValidationFailed, candidate_identity)

__all__ = ['ArtifactLocked', 'AttemptLimit', 'CommandImplementer', 'ImplementationDesk', 'ValidationFailed',
           'candidate_identity']

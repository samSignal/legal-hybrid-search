"""One metadata-filter syntax shared by every retriever.

    {"act_code": "LC"}                       equality
    {"act_code": ["LC", "DP"]}               any of
    {"year": {"gte": 2018, "lte": 2020}}     range (gt / gte / lt / lte)

The same spec is translated to SQL for full-text search, evaluated in Python for the NumPy
store, and converted to a native filter for Qdrant, so all modes return consistent results.
"""
from __future__ import annotations

from typing import Any

Filters = dict[str, Any]
_OPS = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


def validate(spec: Filters | None) -> Filters:
    spec = spec or {}
    for key, cond in spec.items():
        if not key.replace("_", "").isalnum():
            raise ValueError(f"Invalid filter field: {key!r}")
        if isinstance(cond, dict) and not set(cond) <= set(_OPS):
            raise ValueError(f"Unknown range operator in {cond!r}; use {sorted(_OPS)}")
    return spec


def matches(metadata: dict[str, Any], spec: Filters | None) -> bool:
    for key, cond in validate(spec).items():
        value = metadata.get(key)
        if isinstance(cond, dict):
            if value is None:
                return False
            for op, bound in cond.items():
                if not {"gt": value > bound, "gte": value >= bound, "lt": value < bound, "lte": value <= bound}[op]:
                    return False
        elif isinstance(cond, (list, tuple, set)):
            if value not in cond:
                return False
        elif value != cond:
            return False
    return True


def to_sql(spec: Filters | None, column: str = "meta") -> tuple[str, list[Any]]:
    """Return a WHERE fragment over a JSON column plus its parameters (never string-formatted values)."""
    clauses, params = [], []
    for key, cond in validate(spec).items():
        expr = f"json_extract({column}, '$.{key}')"
        if isinstance(cond, dict):
            for op, bound in cond.items():
                clauses.append(f"{expr} {_OPS[op]} ?")
                params.append(bound)
        elif isinstance(cond, (list, tuple, set)):
            cond = list(cond)
            clauses.append(f"{expr} IN ({', '.join('?' * len(cond))})")
            params.extend(cond)
        else:
            clauses.append(f"{expr} = ?")
            params.append(cond)
    return (" AND ".join(clauses) or "1=1"), params


def to_qdrant(spec: Filters | None):
    from qdrant_client import models

    must = []
    for key, cond in validate(spec).items():
        field = f"metadata.{key}"
        if isinstance(cond, dict):
            must.append(models.FieldCondition(key=field, range=models.Range(**cond)))
        elif isinstance(cond, (list, tuple, set)):
            must.append(models.FieldCondition(key=field, match=models.MatchAny(any=list(cond))))
        else:
            must.append(models.FieldCondition(key=field, match=models.MatchValue(value=cond)))
    return models.Filter(must=must) if must else None

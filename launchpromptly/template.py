from __future__ import annotations

import re

_VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def extract_variables(template: str) -> list[str]:
    """Extract unique variable names from a prompt template.

    Matches ``{{variableName}}`` patterns.
    """
    seen: set[str] = set()
    result: list[str] = []
    for match in _VARIABLE_PATTERN.finditer(template):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def interpolate(template: str, variables: dict[str, str]) -> str:
    """Replace ``{{variable}}`` placeholders with values.

    Unmatched variables are left as-is.
    """

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return variables[name] if name in variables else match.group(0)

    return _VARIABLE_PATTERN.sub(_replace, template)

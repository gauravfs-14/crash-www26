import re
from typing import Dict


UNIT_PATTERN = re.compile(r"\b(?:unit[-\s#]?0*([1-9]\d*)|u[-\s]?0*([1-9]\d*))\b", re.IGNORECASE)
DRIVER_OF_UNIT_PATTERN = re.compile(r"\bdriver of unit[-\s#]?0*([1-9]\d*)\b", re.IGNORECASE)


def build_id_maps(row: Dict[str, str]) -> Dict[str, str]:
    """Build maps from narrative unit numbers to canonical Unit_ID/Person_ID.

    Expects within the same Crash_ID:
    - UnitNbr_Un -> Unit_ID
    - (UnitNbr_Pr, Prsn_Nbr) -> Person_ID (when Driver present)
    """
    mapping: Dict[str, str] = {}

    crash_id = str(row.get("Crash_ID", "")).strip()

    # Unit number to Unit_ID
    unit_nbr = str(row.get("UnitNbr_Un", "")).strip()
    unit_id = str(row.get("Unit_ID", "")).strip()
    if crash_id and unit_nbr and unit_id:
        mapping[f"unit:{crash_id}:{unit_nbr}"] = unit_id

    # Driver of unit_n -> Person_ID when driver row
    prsn_type = str(row.get("Prsn_Type_ID", "")).strip().lower()
    unit_nbr_pr = str(row.get("UnitNbr_Pr", "")).strip()
    prsn_nbr = str(row.get("Prsn_Nbr", "")).strip()
    person_id = str(row.get("Person_ID", "")).strip()
    if crash_id and prsn_type == "driver" and unit_nbr_pr and prsn_nbr and person_id:
        mapping[f"driver:{crash_id}:{unit_nbr_pr}"] = person_id

    return mapping


def replace_mentions(narrative: str, maps: Dict[str, str], crash_id: str) -> str:
    """Replace 'Unit 1' etc. with Unit_ID, and 'Driver of Unit 1' with Person_ID when available."""

    def replace_driver(match):
        unit_num = match.group(1)
        key = f"driver:{crash_id}:{unit_num}"
        if key in maps:
            return maps[key]
        # Fallback to Unit_ID if no driver person id
        key_unit = f"unit:{crash_id}:{unit_num}"
        if key_unit in maps:
            return maps[key_unit]
        return match.group(0)

    def replace_unit(match):
        unit_num = match.group(1) or match.group(2)
        key = f"unit:{crash_id}:{unit_num}"
        return maps.get(key, match.group(0))

    # Replace "Driver of Unit X" first for specificity
    out = DRIVER_OF_UNIT_PATTERN.sub(replace_driver, narrative)
    # Replace generic Unit X
    out = UNIT_PATTERN.sub(replace_unit, out)
    return out


def map_unit_label_to_id(label: str, maps: Dict[str, str], crash_id: str) -> str:
    """Map strings like 'Unit 1', 'UNIT-01', 'u2' to Unit_ID using provided maps.
    Returns original label if no mapping available.
    """
    m = UNIT_PATTERN.fullmatch(label.strip())
    if not m:
        return label
    unit_num = m.group(1) or m.group(2)
    key = f"unit:{crash_id}:{unit_num}"
    return maps.get(key, label)

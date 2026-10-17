"""SMILES validation (spec 03, Req 1; FR-1, FR-2).

Parsing is split from sanitization so a failure can be explained: an unparseable string
("unexpected token") is a different problem from a chemically impossible molecule ("explicit
valence for atom #1 C is 5"). RDKit reports both through its logger, which is captured and
returned as the user-facing reason. Invalid input never raises on the inference path.
"""

import contextlib
import io
import re
from dataclasses import dataclass

from rdkit import Chem, rdBase

from oncora.chem.standardize import standardize

MAX_SMILES_LENGTH = 2000


@dataclass(frozen=True)
class ParsedMol:
    valid: bool
    input_smiles: str
    canonical_smiles: str | None = None
    inchi_key: str | None = None
    error: str | None = None
    mol: Chem.Mol | None = None  # standardized molecule, for featurization


_POSITION = re.compile(r"check for mistakes around position (\d+)")


def _rdkit_messages(buf: io.StringIO) -> str:
    """Reduce RDKit's multi-line parse log to its core message plus the error position."""
    text = buf.getvalue()
    core = []
    for line in text.splitlines():
        line = line.split("] ", 1)[-1].strip().removeprefix("SMILES Parse Error:").strip()
        if not line or line.startswith(("check for mistakes", "Failed parsing")) or set(line) <= set("~^ "):
            continue
        line = re.sub(r"\s*(while parsing|for input):.*$", "", line)
        if line and line not in core:
            core.append(line)
    msg = "; ".join(core[:1])  # the first line is the cause; the rest echo the input
    if m := _POSITION.search(text):
        msg += f" (around character {m.group(1)})"
    return msg


def parse_smiles(smiles: str) -> ParsedMol:
    text = (smiles or "").strip()
    if not text:
        return ParsedMol(False, smiles or "", error="Empty input: enter a SMILES string.")
    if len(text) > MAX_SMILES_LENGTH:
        return ParsedMol(False, text, error=f"SMILES is longer than {MAX_SMILES_LENGTH} characters.")
    if any(ch.isspace() for ch in text):
        return ParsedMol(False, text, error="SMILES must not contain spaces; submit one molecule at a time.")

    buf = io.StringIO()
    rdBase.LogToPythonStderr()
    with contextlib.redirect_stderr(buf):
        mol = Chem.MolFromSmiles(text, sanitize=False)
        if mol is None:
            return ParsedMol(False, text, error=f"Could not parse SMILES: {_rdkit_messages(buf) or 'syntax error'}")
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:  # RDKit raises several sanitization exception types
            hint = " Check aromatic (lowercase) ring notation." if "kekulize" in str(exc) else ""
            return ParsedMol(False, text, error=f"Invalid molecule: {' '.join(str(exc).split())}.{hint}")
    if mol.GetNumAtoms() == 0:
        return ParsedMol(False, text, error="SMILES contains no atoms.")

    try:
        std = standardize(mol)
    except Exception as exc:
        return ParsedMol(False, text, error=f"Could not standardize molecule: {exc}")
    return ParsedMol(
        True, text,
        canonical_smiles=Chem.MolToSmiles(std),
        inchi_key=Chem.MolToInchiKey(std) or None,
        mol=std,
    )

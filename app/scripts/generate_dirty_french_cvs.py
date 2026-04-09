from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

from faker import Faker
from fpdf import FPDF
from fpdf.enums import XPos, YPos


INDUSTRIES = [
    "Informatique",
    "Finance",
    "Sante",
    "Marketing",
    "Ingenierie",
    "Education",
    "Logistique",
    "Energie",
]

SKILLS_FR = [
    "Python",
    "SQL",
    "Docker",
    "React",
    "Flask",
    "Gestion de projet",
    "Gestion des risques",
    "Planification budgetaire",
    "Coordination des equipes",
    "Scrum",
    "Kanban",
    "Leadership",
    "Communication",
    "Analyse de donnees",
]

SKILLS_EN = [
    "project management",
    "risk management",
    "data analysis",
    "stakeholder communication",
    "problem solving",
]

TOOLS = ["Git", "Jira", "Confluence", "Power BI", "Tableau", "VMware", "Burp Suite"]
LANGS = [("Francais", "C2"), ("Anglais", "B2"), ("Arabe", "C1"), ("Espagnol", "B1")]

MESSY_FONTS = ["Helvetica", "Times", "Courier"]
MESSY_SIZES = [7, 8, 9, 10, 11, 12, 14]

SECTION_TITLES = [
    "COMPETENCES",
    "C O M P E T E N C E S",
    "EXPERTISE",
    "SKILLS",
    "Langues",
    "LANGUES / LANGUAGES",
    "EXPERIENCE PROFESSIONNELLE",
    "Work Experience",
    "FORMATION / EDUCATION",
]


NOISE_LEVEL_SCALES = {
    "low": 0.45,
    "medium": 0.75,
    "high": 1.0,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _scaled_prob(base_p: float, noise_scale: float) -> float:
    return _clamp01(base_p * noise_scale)


def _chance(rng: random.Random, p: float) -> bool:
    return rng.random() < _clamp01(p)


def _pdf_safe_text(text: str) -> str:
    """Coerce text to a representation compatible with core FPDF fonts."""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = (
        normalized.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2026", "...")
        .replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )
    return normalized.encode("latin-1", errors="replace").decode("latin-1")


def _cell_ln(pdf: FPDF, height: float, text: str) -> None:
    safe_text = _pdf_safe_text(text)
    try:
        # fpdf2 API
        pdf.cell(0, height, safe_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    except TypeError:
        # Legacy pyfpdf API
        pdf.cell(0, height, safe_text, ln=1)


def _multi_cell(pdf: FPDF, width: float, height: float, text: str) -> None:
    pdf.multi_cell(width, height, _pdf_safe_text(text))


def _slug(s: str) -> str:
    x = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    x = re.sub(r"[^a-zA-Z0-9]+", "-", x).strip("-").lower()
    return x or "cv"


def _shuffle_case(s: str, rng: random.Random) -> str:
    out = []
    for ch in s:
        if ch.isalpha() and rng.random() < 0.25:
            out.append(ch.upper() if ch.islower() else ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def _insert_ocr_noise(s: str, rng: random.Random, p: float = 0.08) -> str:
    swaps = {"e": "3", "o": "0", "i": "1", "l": "I", "a": "@", "s": "5"}
    out = []
    for ch in s:
        low = ch.lower()
        if low in swaps and rng.random() < p:
            out.append(swaps[low])
        else:
            out.append(ch)
    noisy = "".join(out)
    noisy = re.sub(r"\s{2,}", " ", noisy)
    return noisy


def _maybe_break_words(s: str, rng: random.Random, noise_scale: float = 1.0) -> str:
    if _chance(rng, _scaled_prob(0.4, noise_scale)):
        return " ".join(list(s)) if len(s) < 20 else s
    if _chance(rng, _scaled_prob(0.5, noise_scale)):
        return s.replace(" ", "  ")
    return s


def _messy_line(s: str, rng: random.Random, very_dirty: bool, noise_scale: float = 1.0) -> str:
    line = s
    if very_dirty:
        line = _insert_ocr_noise(line, rng, p=max(0.02, 0.12 * _clamp01(noise_scale)))
    if _chance(rng, _scaled_prob(0.35, noise_scale)):
        line = _shuffle_case(line, rng)
    if _chance(rng, _scaled_prob(0.2, noise_scale)):
        line = line.replace("-", " - ")
    if _chance(rng, _scaled_prob(0.2, noise_scale)):
        line = line + rng.choice(["", ".", " ..", " ;", " /"])
    return line


def _year_range(rng: random.Random) -> str:
    start = rng.randint(2008, 2023)
    end = rng.randint(start, 2026)
    sep = rng.choice(["-", "–", "/", " -> "])
    return f"{start}{sep}{end}"


def _pick_many(pool: list[str], min_n: int, max_n: int, rng: random.Random) -> list[str]:
    n = rng.randint(min_n, max_n)
    return rng.sample(pool, k=min(n, len(pool)))


def _draw_header(
    pdf: FPDF,
    fake: Faker,
    rng: random.Random,
    very_dirty: bool,
    noise_scale: float = 1.0,
) -> dict[str, Any]:
    identity = {
        "name": fake.name(),
        "email": fake.email(),
        "phone": fake.phone_number(),
        "industry": rng.choice(INDUSTRIES),
    }
    header = f"Curriculum Vitae - {identity['name']}"
    if very_dirty and _chance(rng, _scaled_prob(0.5, noise_scale)):
        header = _maybe_break_words(header, rng, noise_scale=noise_scale)
    _cell_ln(pdf, 7, _messy_line(header, rng, very_dirty, noise_scale=noise_scale))
    _cell_ln(
        pdf,
        6,
        _messy_line(f"Email: {identity['email']} | Tel: {identity['phone']}", rng, very_dirty, noise_scale=noise_scale),
    )
    _cell_ln(pdf, 6, _messy_line(f"Secteur: {identity['industry']}", rng, very_dirty, noise_scale=noise_scale))
    pdf.ln(rng.choice([2, 3, 4, 5]))
    return identity


def _draw_skills_section(
    pdf: FPDF,
    rng: random.Random,
    very_dirty: bool,
    noise_scale: float = 1.0,
) -> dict[str, Any]:
    sec = rng.choice(["COMPETENCES", "C O M P E T E N C E S", "SKILLS", "EXPERTISE"])
    _cell_ln(pdf, 6, _messy_line(sec, rng, very_dirty, noise_scale=noise_scale))
    chosen_fr = _pick_many(SKILLS_FR, 3, 6, rng)
    chosen_tools = _pick_many(TOOLS, 1, 3, rng)
    chosen_en = _pick_many(SKILLS_EN, 1, 3, rng) if rng.random() < 0.7 else []
    for item in chosen_fr + chosen_tools + chosen_en:
        bullet = rng.choice(["- ", "* ", "+ ", ""])
        line = f"{bullet}{item}"
        if very_dirty and _chance(rng, _scaled_prob(0.35, noise_scale)):
            line = _maybe_break_words(line, rng, noise_scale=noise_scale)
        _cell_ln(pdf, 5, _messy_line(line, rng, very_dirty, noise_scale=noise_scale))
    pdf.ln(rng.choice([1, 2, 3]))
    return {"skills": chosen_fr + chosen_en, "tools": chosen_tools}


def _draw_experience_section(
    pdf: FPDF,
    fake: Faker,
    rng: random.Random,
    very_dirty: bool,
    noise_scale: float = 1.0,
) -> list[str]:
    title = rng.choice(["EXPERIENCE PROFESSIONNELLE", "Work Experience", "EXPERIENCE / EXPERIENCE"])
    _cell_ln(pdf, 6, _messy_line(title, rng, very_dirty, noise_scale=noise_scale))
    bullets: list[str] = []
    for _ in range(rng.randint(2, 5)):
        line = f"- {fake.job()} chez {fake.company()} ({_year_range(rng)})"
        bullets.append(line)
        _cell_ln(pdf, 5, _messy_line(line, rng, very_dirty, noise_scale=noise_scale))
        if _chance(rng, _scaled_prob(0.55, noise_scale)):
            en = "Managed cross-functional teams and delivered key projects."
            fr = "Pilotage de projets transverses, coordination des parties prenantes."
            mix = rng.choice([en, fr, f"{fr} {en}"])
            bullets.append(mix)
            _multi_cell(pdf, 0, 5, _messy_line(mix, rng, very_dirty, noise_scale=noise_scale))
    pdf.ln(rng.choice([1, 2, 3]))
    return bullets


def _draw_education_section(
    pdf: FPDF,
    fake: Faker,
    rng: random.Random,
    very_dirty: bool,
    noise_scale: float = 1.0,
) -> str:
    title = rng.choice(["FORMATION", "FORMATION / EDUCATION", "EDUCATION"])
    _cell_ln(pdf, 6, _messy_line(title, rng, very_dirty, noise_scale=noise_scale))
    deg = f"{fake.job()} - {fake.city()} ({rng.randint(2005, 2025)})"
    _cell_ln(pdf, 5, _messy_line(deg, rng, very_dirty, noise_scale=noise_scale))
    pdf.ln(rng.choice([1, 2]))
    return deg


def _draw_languages_section(
    pdf: FPDF,
    rng: random.Random,
    very_dirty: bool,
    noise_scale: float = 1.0,
) -> list[dict[str, str]]:
    title = rng.choice(["LANGUES", "LANGUES / LANGUAGES", "LANGUAGES"])
    _cell_ln(pdf, 6, _messy_line(title, rng, very_dirty, noise_scale=noise_scale))
    k = rng.randint(2, 4)
    chosen = rng.sample(LANGS, k=k)
    out = []
    for lang, lvl in chosen:
        if rng.random() < 0.25:
            lvl = rng.choice(["A1", "A2", "B1", "B2", "C1", "C2"])
        fmt = rng.choice(
            [
                f"- {lang} ({lvl})",
                f"- {lang} : {lvl}",
                f"{lang} {lvl}",
                f"{lang} ({lvl} en negociation)",
            ]
        )
        _cell_ln(pdf, 5, _messy_line(fmt, rng, very_dirty, noise_scale=noise_scale))
        out.append({"language": _slug(lang), "level": lvl})
    pdf.ln(rng.choice([1, 2]))
    return out


def _draw_garbage_blocks(
    pdf: FPDF,
    fake: Faker,
    rng: random.Random,
    very_dirty: bool,
    noise_scale: float = 1.0,
) -> list[str]:
    noise_lines: list[str] = []
    if _chance(rng, _scaled_prob(0.7, noise_scale)):
        _cell_ln(pdf, 5, _messy_line(rng.choice(SECTION_TITLES), rng, very_dirty, noise_scale=noise_scale))
    line_min = max(1, int(round(2 * _clamp01(noise_scale))))
    line_max = max(line_min + 1, int(round(2 + (6 * _clamp01(noise_scale)))))
    for _ in range(rng.randint(line_min, line_max)):
        parts = [
            fake.catch_phrase(),
            fake.word(),
            rng.choice(["N/A", "###", "::::", "2025/2026", "B2", "C1", ""]),
            fake.city(),
        ]
        line = " ".join([p for p in parts if p]).strip()
        if very_dirty:
            line = _insert_ocr_noise(line, rng, p=max(0.03, 0.18 * _clamp01(noise_scale)))
        if _chance(rng, _scaled_prob(0.3, noise_scale)):
            line = line + " " + rng.choice(SKILLS_FR + SKILLS_EN)
        noise_lines.append(line)
        if _chance(rng, _scaled_prob(0.3, noise_scale)):
            pdf.set_x(rng.randint(10, 120))
        if _chance(rng, _scaled_prob(0.25, noise_scale)):
            _multi_cell(pdf, rng.randint(70, 160), 5, _messy_line(line, rng, very_dirty, noise_scale=noise_scale))
        else:
            _cell_ln(pdf, 5, _messy_line(line, rng, very_dirty, noise_scale=noise_scale))
    return noise_lines


def generate_cv(
    output_path: Path,
    fake: Faker,
    rng: random.Random,
    very_dirty: bool = True,
    noise_scale: float = 1.0,
    noise_level: str = "high",
) -> dict[str, Any]:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()

    font = rng.choice(MESSY_FONTS)
    size = rng.choice(MESSY_SIZES if very_dirty else [10, 11, 12])
    pdf.set_font(font, size=size)

    if very_dirty and _chance(rng, _scaled_prob(0.35, noise_scale)):
        pdf.set_left_margin(rng.randint(5, 25))
        pdf.set_right_margin(rng.randint(5, 25))

    identity = _draw_header(pdf, fake, rng, very_dirty, noise_scale=noise_scale)
    skills = _draw_skills_section(pdf, rng, very_dirty, noise_scale=noise_scale)
    experience = _draw_experience_section(pdf, fake, rng, very_dirty, noise_scale=noise_scale)
    education = _draw_education_section(pdf, fake, rng, very_dirty, noise_scale=noise_scale)
    languages = _draw_languages_section(pdf, rng, very_dirty, noise_scale=noise_scale)

    garbage = []
    if very_dirty:
        garbage = _draw_garbage_blocks(pdf, fake, rng, very_dirty, noise_scale=noise_scale)

    if very_dirty and _chance(rng, _scaled_prob(0.4, noise_scale)):
        pdf.set_y(max(20, pdf.get_y() - rng.randint(5, 40)))
        pdf.set_font(rng.choice(MESSY_FONTS), size=rng.choice([7, 8, 9]))
        _cell_ln(
            pdf,
            4,
            _messy_line(
                "References available upon request / Références sur demande",
                rng,
                True,
                noise_scale=noise_scale,
            ),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))

    return {
        "file": output_path.name,
        "very_dirty": very_dirty,
        "noise_level": noise_level,
        "noise_scale": round(_clamp01(noise_scale), 3),
        "identity": identity,
        "labels": {
            "skills": skills["skills"],
            "tools": skills["tools"],
            "languages": languages,
            "education": education,
            "experience_bullets": experience,
            "garbage_lines": garbage,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic dirty French CV PDFs for parser stress testing.")
    parser.add_argument("--out-dir", default="artifacts/synth_cvs")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--clean-ratio", type=float, default=0.15, help="Fraction of less-dirty CVs in [0,1].")
    parser.add_argument(
        "--noise-level",
        choices=sorted(NOISE_LEVEL_SCALES.keys()),
        default="high",
        help="Noise intensity for dirty CVs: low|medium|high",
    )
    parser.add_argument("--manifest", default="manifest.jsonl", help="JSONL metadata file name inside out-dir.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / args.manifest

    rng = random.Random(int(args.seed))
    fake = Faker("fr_FR")
    Faker.seed(args.seed)

    rows: list[dict[str, Any]] = []
    dirty_noise_scale = NOISE_LEVEL_SCALES[args.noise_level]
    for i in range(int(args.count)):
        very_dirty = rng.random() > max(0.0, min(1.0, float(args.clean_ratio)))
        row_noise_level = args.noise_level if very_dirty else "low"
        row_noise_scale = dirty_noise_scale if very_dirty else min(0.45, dirty_noise_scale * 0.55)
        name_part = _slug(fake.name())[:24]
        path = out_dir / f"cv_{i:04d}_{name_part}.pdf"
        row = generate_cv(
            path,
            fake=fake,
            rng=rng,
            very_dirty=very_dirty,
            noise_scale=row_noise_scale,
            noise_level=row_noise_level,
        )
        rows.append(row)

    manifest_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )
    print(f"Generated {len(rows)} PDFs in: {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

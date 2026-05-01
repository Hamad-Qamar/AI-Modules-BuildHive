"""
CostEstimationModule — Module 4 (Extended)
Estimates construction costs from structured OR text-based inputs.
Loads pricing from prices.json — never hardcoded.
"""

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .estimator_policy import (
    clamp_layout_inputs,
    feasibility_caps,
    floor_cost_factors,
    load_feasibility_bands,
    load_floor_policy,
    public_feasibility_bands_raw,
    resolve_layout_from_bhk,
)
from . import finishing_catalog
from .llm_helper import LLMHelper

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — defined once, never scattered inline
# ─────────────────────────────────────────────────────────────────────────────

# Pakistan standard area conversions → sqft (fallbacks only).
# City-aware marla conversions come from `city_area_standards.csv`.
AREA_CONVERSIONS: Dict[str, float] = {
    # Standardize marla to an integer to avoid subtle float mismatches across
    # backend/frontend, CSVs, and tests.
    "marla":  272.0,
    "kanal":  5445.0,
    "sqft":   1.0,
    "sft":    1.0,
    "sqm":    10.764,
    "sqyard": 9.0,
    "gaj":    9.0,     # Urdu/Punjabi for sq yard
}

# Minimum viable covered area (sqft) to prevent absurd estimates (e.g., 1 sqft).
MIN_VIABLE_SQFT = 120

# Construction types (strict separation)
CONSTRUCTION_TYPES = ("grey_structure", "full_construction", "renovation")

# Phase groupings for BOQ generation from `materials_master.csv`
GREY_STRUCTURE_PHASES = {
    "Site Preparation",
    "Excavation & Foundation",
    "Grey Structure",
    "Masonry & Walls",
}
FULL_CONSTRUCTION_PHASES = None  # means: include all phases
RENOVATION_PHASES = {
    "Plastering & Screeding",
    "Flooring & Tiling",
    "Electrical (Rough + Final)",
    "Plumbing (Rough + Final)",
    "Paint & Finishing",
    "Carpentry & Woodwork",
    "Kitchen & Wardrobes",
    "Sanitary & Bathroom Fittings",
    "Aluminum & Glass Work",
    "Roofing & Waterproofing",
    "External Works",
}

# Renovation must be scope-based (not one linear multiplier).
# Each phase has its own typical coverage percentage for renovation work.
RENOVATION_PHASE_COVERAGE: Dict[str, float] = {
    "Paint & Finishing": 1.00,                 # often full repaint
    "Flooring & Tiling": 0.60,                 # usually partial replacement
    "Plastering & Screeding": 0.35,
    "Electrical (Rough + Final)": 0.30,
    "Plumbing (Rough + Final)": 0.30,
    "Carpentry & Woodwork": 0.35,
    "Kitchen & Wardrobes": 0.50,
    "Sanitary & Bathroom Fittings": 0.50,
    "Aluminum & Glass Work": 0.25,
    "Roofing & Waterproofing": 0.25,
    "External Works": 0.25,
}

# Contingency by tier (materials+labour)
CONTINGENCY_BY_GRADE: Dict[str, float] = {
    "economy": 0.05,
    "standard": 0.07,
    "premium": 0.10,
    "luxury": 0.12,
}

# Quality tier multipliers (applied to categories that vary heavily by spec).
# Grey structure is mostly quantity-driven; finishes/fixtures/labour move most.
QUALITY_TIER_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "economy": {"Finishing": 0.85, "Electrical": 0.90, "Plumbing": 0.90, "Misc": 0.95, "Grey Structure": 1.00, "labour": 0.92},
    "standard": {"Finishing": 1.00, "Electrical": 1.00, "Plumbing": 1.00, "Misc": 1.00, "Grey Structure": 1.00, "labour": 1.00},
    "premium": {"Finishing": 1.25, "Electrical": 1.18, "Plumbing": 1.20, "Misc": 1.10, "Grey Structure": 1.02, "labour": 1.08},
    "luxury": {"Finishing": 1.60, "Electrical": 1.35, "Plumbing": 1.40, "Misc": 1.18, "Grey Structure": 1.05, "labour": 1.15},
}

# Benchmark enforcement bounds (BOQ total vs benchmark total)
BENCHMARK_LOW = 0.70
BENCHMARK_HIGH = 1.30

# Benchmark quality factors (used only when applying a benchmark floor).
# Turnkey benchmarks vary a lot by spec; grey-structure varies less.
BENCHMARK_QUALITY_FACTOR_TURNKEY: Dict[str, float] = {
    "economy": 0.85,
    "standard": 1.00,
    "premium": 1.25,
    "luxury": 1.60,
}
BENCHMARK_QUALITY_FACTOR_GREY: Dict[str, float] = {
    "economy": 0.92,
    "standard": 1.00,
    "premium": 1.10,
    "luxury": 1.20,
}

# Grey-structure BOQ minimum intensity vs AREA Covered (typical PK practice band).
# Catalogue `per_sqft` rows alone often under-state slab + full envelope on small houses.
GREY_BOQ_MIN_STEEL_KG_PER_SQFT = 3.6
GREY_BOQ_MIN_BRICKS_PER_SQFT = 30.0
GREY_BOQ_MIN_CEMENT_BAGS_PER_SQFT = 0.38
GREY_BOQ_MIN_CEMENT_BAGS_ABS = 40.0
GREY_BOQ_MIN_CRUSH_CFT_PER_SQFT = 0.72
GREY_BOQ_MIN_SAND_CFT_PER_SQFT = 0.42
GREY_LABOUR_PKR_PER_SQFT_FLOOR = 580.0

GREY_BOQ_PHASE_SET = frozenset(GREY_STRUCTURE_PHASES)


def _apply_grey_structure_boq_minimums(
    mm_q: pd.DataFrame, total_sqft: float, construction_type: str
) -> Tuple[pd.DataFrame, bool]:
    """Raise thin catalogue-derived grey BOQ toward typical structural completeness.

    Applies to both grey_structure and full_construction because the catalogue
    per-sqft ratios for bricks and cement are dangerously under-stated at small
    footprints (1–2 marla).  For example the catalogue brickwork ratio of
    8.5 pcs/sqft yields only ~1 900 bricks for a 225 sqft house, whereas the
    physical reality demands 5 500–6 750 bricks (wall area ≈ 600 sqft × 9–10
    bricks/sqft).  The minimum intensity constants are based on wall-area
    physics, not arbitrary "more is safer" logic.

    Steel is deliberately NOT boosted for full_construction: all structural
    steel phases (foundation, columns/beams/slab, lintels) are already
    individual line-items in the catalogue that sum to the correct range.
    Boosting them further would over-state cost for a full build.
    """
    if construction_type not in ("grey_structure", "full_construction") or total_sqft <= 0:
        return mm_q, False
    sq = float(total_sqft)
    df = mm_q.copy()
    touched = False

    cat_lower = df["category"].astype(str).str.lower()
    name_lower = df["name"].astype(str).str.lower()
    phase_str = df["phase"].astype(str)
    unit_lower = df["unit"].astype(str).str.lower()

    # Steel boost: only for grey_structure (pure structural contract).
    # full_construction already carries all steel phases as individual line-items.
    if construction_type == "grey_structure":
        steel_mask = unit_lower.eq("kg") & cat_lower.eq("steel")
        cur_steel = float(df.loc[steel_mask, "quantity_raw"].sum())
        tgt_steel = GREY_BOQ_MIN_STEEL_KG_PER_SQFT * sq
        if cur_steel > 0 and tgt_steel > cur_steel:
            df.loc[steel_mask, "quantity_raw"] *= tgt_steel / cur_steel
            touched = True

    # Brick minimum — applies to both grey_structure and full_construction.
    # Physics: wall area ≈ 4 × √(floor_area) × 10 ft × 1.5 (partitions factor)
    #         bricks   ≈ 9 per sqft of wall area
    #  → min_bricks  ≈ 9 × 60 × √sqft = 540 × √sqft
    #
    # For grey_structure contracts a flat 30/sqft minimum is used (legacy, tested).
    # For full_construction the wall-area formula is used: it gives physically
    # accurate results and avoids the 30/sqft formula drastically over-stating
    # bricks for 5–10 marla houses (which would push BOQ well above market ceilings).
    #   225 sqft → 540×√225 = 8 100 bricks   (user needed 5 500–6 000 minimum ✓)
    #  1125 sqft → 540×√1125 = 18 100 bricks  (5 marla realistic ✓)
    #  2250 sqft → 540×√2250 = 25 600 bricks  (10 marla single-storey ✓)
    brick_mask = unit_lower.isin(["pcs", "piece", "pieces"]) & name_lower.str.contains(
        "brick", regex=False
    )
    cur_b = float(df.loc[brick_mask, "quantity_raw"].sum())
    if construction_type == "grey_structure":
        tgt_b = GREY_BOQ_MIN_BRICKS_PER_SQFT * sq
    else:
        tgt_b = 540.0 * math.sqrt(sq)
    if cur_b > 0 and tgt_b > cur_b:
        df.loc[brick_mask, "quantity_raw"] *= tgt_b / cur_b
        touched = True

    # Cement minimum (grey-phase rows only) — applies to both types.
    # A 225 sqft house needs ≥55 bags: slab ~18, plaster ~15, brickwork ~15,
    # foundation ~7.  The catalogue grey-phase cement rows sum to only ~2-3 bags.
    phase_ok = phase_str.isin(GREY_BOQ_PHASE_SET)
    cement_mask = (
        phase_ok
        & unit_lower.str.contains("bag", regex=False)
        & name_lower.str.contains("cement", regex=False)
        & ~name_lower.str.contains("solvent", regex=False)
    )
    cur_cem = float(df.loc[cement_mask, "quantity_raw"].sum())
    # Use a slightly lower per-sqft rate for full_construction because the full
    # build carries finishing-phase cement (tile adhesive, plaster, etc.) in its own
    # line-items; we only top up the structural grey phases here.
    min_cem_sqft = (
        GREY_BOQ_MIN_CEMENT_BAGS_PER_SQFT if construction_type == "grey_structure"
        else 0.25  # ~56 bags for 225 sqft covers slab + plaster + brickwork + foundation
    )
    tgt_cem = max(min_cem_sqft * sq, GREY_BOQ_MIN_CEMENT_BAGS_ABS)
    if cur_cem > 0 and tgt_cem > cur_cem:
        df.loc[cement_mask, "quantity_raw"] *= tgt_cem / cur_cem
        touched = True

    crush_mask = phase_ok & (
        name_lower.str.contains("crush", regex=False)
        | name_lower.str.contains("bajri", regex=False)
    )
    cur_cr = float(df.loc[crush_mask, "quantity_raw"].sum())
    tgt_cr = GREY_BOQ_MIN_CRUSH_CFT_PER_SQFT * sq
    if cur_cr > 0 and tgt_cr > cur_cr:
        df.loc[crush_mask, "quantity_raw"] *= tgt_cr / cur_cr
        touched = True

    sand_mask = (
        phase_ok
        & name_lower.str.contains("sand", regex=False)
        & ~name_lower.str.contains("solvent", regex=False)
    )
    cur_sa = float(df.loc[sand_mask, "quantity_raw"].sum())
    tgt_sa = GREY_BOQ_MIN_SAND_CFT_PER_SQFT * sq
    if cur_sa > 0 and tgt_sa > cur_sa:
        df.loc[sand_mask, "quantity_raw"] *= tgt_sa / cur_sa
        touched = True

    return df, touched


# Turnkey MEP + exposed finishing BOQ floors vs covered area (PK practice; scales past “1 marla reference”).
ELECTRICAL_LABOUR_PKR_PER_SQFT_FLOOR = 52.0
PLUMBING_LABOUR_PKR_PER_SQFT_FLOOR = 58.0
FINISHING_TRADES_LABOUR_PKR_PER_SQFT_FLOOR = 265.0


def _apply_turnkey_mep_finishing_boq_minimums(
    mm_q: pd.DataFrame,
    total_sqft: float,
    construction_type: str,
    baths: int,
    rooms: int,
    kitchens: int,
) -> Tuple[pd.DataFrame, bool]:
    """Raise thin MEP/finishing catalogue quantities toward credible whole-home scope."""
    if construction_type not in ("full_construction", "renovation") or total_sqft <= 0:
        return mm_q, False
    sq = float(total_sqft)
    df = mm_q.copy()
    touched = False
    name_lower = df["name"].astype(str).str.lower()
    unit_lower = df["unit"].astype(str).str.lower()
    phase_str = df["phase"].astype(str)

    baths_i = max(int(baths), 0)
    rooms_i = max(int(rooms), 0)
    kits_i = max(int(kitchens), 0)
    baths_eff = max(baths_i, 1) if construction_type == "full_construction" else max(baths_i, 0)
    rooms_e = max(rooms_i, 1)

    # ── Sanitary / shower: whole sets vs wet rooms ──
    # Enforce a two-sided constraint: raise to baths_eff if under-stated, AND
    # cap to baths_eff if over-stated.  Without the upper cap, a catalogue ratio
    # of 0.0045/sqft × 225 sqft = 1.01 sets (after phase multiplier) for a 1-bath
    # house can silently exceed 1 when phase_multiplier > 1.0, producing 2 sets
    # for a house that can only physically fit 1 bathroom.
    if baths_eff >= 1:
        for substr, uneedle in (("sanitary ware", "sets"), ("shower set", "set")):
            mask = name_lower.str.contains(substr, regex=False) & unit_lower.str.contains(
                uneedle, regex=False
            )
            if not mask.any():
                continue
            for idx in df.index[mask]:
                v = float(df.at[idx, "quantity_raw"])
                if v + 1e-9 < float(baths_eff):
                    df.at[idx, "quantity_raw"] = float(baths_eff)
                    touched = True
                elif v > float(baths_eff) + 1e-9:
                    # Cap: e.g. 0.0045×225×phase_mult > 1 should not create 2 sets
                    df.at[idx, "quantity_raw"] = float(baths_eff)
                    touched = True

    # ── Solvent cement: retail whole cans ──
    m_sol = unit_lower.eq("can") & name_lower.str.contains("solvent", regex=False)
    if m_sol.any():
        for idx in df.index[m_sol]:
            v = float(df.at[idx, "quantity_raw"])
            nv = max(1.0, float(math.ceil(v))) if v > 1e-9 else 1.0
            if abs(nv - v) > 1e-6:
                df.at[idx, "quantity_raw"] = nv
                touched = True

    # ── Ball valves: at least two functional stops on small houses ──
    m_bv = phase_str.str.contains("Plumbing", regex=False) & name_lower.str.contains(
        "ball valve", regex=False
    ) & unit_lower.eq("pcs")
    cur_bv = float(df.loc[m_bv, "quantity_raw"].sum())
    tgt_bv = max(2.0, float(math.ceil(cur_bv))) if cur_bv > 1e-9 else 0.0
    if m_bv.any() and cur_bv > 1e-9 and tgt_bv > cur_bv + 1e-9:
        df.loc[m_bv, "quantity_raw"] *= tgt_bv / cur_bv
        touched = True

    # ── Drainage + water networks (typical single-storey envelope) ──
    m_u4 = (
        name_lower.str.contains("upvc", regex=False)
        & name_lower.str.contains("sewer", regex=False)
        & name_lower.str.contains("4", regex=False)
        & unit_lower.eq("ft")
    )
    su = float(df.loc[m_u4, "quantity_raw"].sum())
    tgt_u = max(38.0, 0.14 * sq)
    if m_u4.any() and su > 1e-9 and tgt_u > su + 1e-9:
        df.loc[m_u4, "quantity_raw"] *= tgt_u / su
        touched = True

    m_pp = (
        phase_str.str.contains("Plumbing", regex=False)
        & name_lower.str.contains("pprc", regex=False)
        & name_lower.str.contains("pipe", regex=False)
        & unit_lower.eq("ft")
    )
    sp = float(df.loc[m_pp, "quantity_raw"].sum())
    tgt_p = max(68.0, 0.30 * sq)
    if m_pp.any() and sp > 1e-9 and tgt_p > sp + 1e-9:
        df.loc[m_pp, "quantity_raw"] *= tgt_p / sp
        touched = True

    # ── Tiles: floors + wet walls + wastage band ──
    # Floor tiles should never exceed the actual floor area × (1 + waste).
    # Wall tiles apply only to wet areas (baths + kitchen backsplash), NOT to
    # total floor area.  The old formula `1.55×sqft + 62×baths` produced 410 sqft
    # for a 225 sqft / 1-bath house (nearly double the real need of ~260 sqft).
    #
    # Correct physics:
    #   floor tiles  = floor_area × 1.15 (15 % cut-waste)
    #   wall tiles   = baths × 55 sqft + kitchens × 30 sqft (backsplash)
    #   total        ≥ 1.05 × floor_area (soft floor so very large houses aren't under-stated)
    tile_units = ("ft2", "sqft", "sq ft")
    m_floor = name_lower.str.contains("floor tiles", regex=False) & unit_lower.isin(tile_units)
    m_wall = name_lower.str.contains("wall tiles", regex=False) & unit_lower.isin(tile_units)
    st = float(df.loc[m_floor, "quantity_raw"].sum()) + float(df.loc[m_wall, "quantity_raw"].sum())

    floor_tile_max = sq * 1.15                                       # floor area + 15 % waste
    wall_tile_max  = float(baths_eff) * 55.0 + float(kits_i) * 30.0 # per wet-room
    wall_tile_max  = max(wall_tile_max, 40.0)                        # at least one small bath
    total_tile_cap = floor_tile_max + wall_tile_max

    # Minimum: ensure at least 1.05× floor area is quoted (prevents under-stating for
    # large houses whose catalogue entries may be conservative).
    tgt_t = max(1.05 * sq + float(baths_eff) * 45.0 + float(kits_i) * 20.0, 1.02 * sq)

    if (m_floor.any() or m_wall.any()) and st > 1e-9:
        if tgt_t > st + 1e-9:
            # Raise under-stated quantities to minimum
            factor = tgt_t / st
            if m_floor.any():
                df.loc[m_floor, "quantity_raw"] *= factor
            if m_wall.any():
                df.loc[m_wall, "quantity_raw"] *= factor
            st = tgt_t
            touched = True
        if st > total_tile_cap + 1e-9:
            # Cap over-stated quantities to physical maximum
            factor = total_tile_cap / st
            if m_floor.any():
                df.loc[m_floor, "quantity_raw"] *= factor
            if m_wall.any():
                df.loc[m_wall, "quantity_raw"] *= factor
            touched = True

    # ── Putty vs plastered wall area proxy ──
    m_putty = name_lower.str.contains("putty", regex=False) & unit_lower.eq("kg")
    spu = float(df.loc[m_putty, "quantity_raw"].sum())
    tgt_putty = max(1.28 * sq, 110.0)
    if m_putty.any() and spu > 1e-9 and tgt_putty > spu + 1e-9:
        df.loc[m_putty, "quantity_raw"] *= tgt_putty / spu
        touched = True

    # ── Interior emulsion (topcoat litres) ──
    m_em = (
        phase_str.str.contains("Paint", regex=False)
        & name_lower.str.contains("emulsion", regex=False)
        & (unit_lower.str.contains("litre", regex=False) | unit_lower.eq("l"))
    )
    se = float(df.loc[m_em, "quantity_raw"].sum())
    tgt_e = max(0.014 * sq, 4.0)
    if m_em.any() and se > 1e-9 and tgt_e > se + 1e-9:
        df.loc[m_em, "quantity_raw"] *= tgt_e / se
        touched = True

    # ── Door shutters + frames (whole openings) ──
    doors_n = int(min(12, max(3, rooms_e + max(baths_eff, 1) + (1 if kits_i > 0 else 0))))
    m_sh = name_lower.str.contains("door shutter", regex=False) & unit_lower.isin(tile_units)
    ss = float(df.loc[m_sh, "quantity_raw"].sum())
    tgt_sh = float(doors_n) * 21.0
    if m_sh.any() and ss > 1e-9 and tgt_sh > ss + 1e-9:
        df.loc[m_sh, "quantity_raw"] *= tgt_sh / ss
        touched = True

    m_fr = (
        (name_lower.str.contains("door frame", regex=False) | name_lower.str.contains("chowkhat", regex=False))
        & unit_lower.eq("rft")
    )
    sr = float(df.loc[m_fr, "quantity_raw"].sum())
    tgt_r = float(doors_n) * 17.5
    if m_fr.any() and sr > 1e-9 and tgt_r > sr + 1e-9:
        df.loc[m_fr, "quantity_raw"] *= tgt_r / sr
        touched = True

    # ── Aluminum window frame: cap to room-based physical maximum ──
    # Catalogue ratio 0.22 rft/sqft yields 49.5 rft for a 225 sqft 1-room house —
    # equivalent to ~4 full-size windows on a 15×15 ft box.  Reality: one 6×4 window
    # per room/living area (≈20 rft), small vents for bathrooms (6 rft each).
    # Cap formula: rooms × 14 rft + baths × 6 rft + kitchens × 8 rft, min 18 rft.
    m_alr = name_lower.str.contains("aluminum window frame", regex=False) & unit_lower.eq("rft")
    rft_al = float(df.loc[m_alr, "quantity_raw"].sum())
    rft_cap = float(rooms_e) * 14.0 + float(max(baths_eff, 0)) * 6.0 + float(kits_i) * 8.0
    rft_cap = max(rft_cap, 18.0)  # guarantee at least one window opening
    if m_alr.any() and rft_al > rft_cap + 1e-9:
        df.loc[m_alr, "quantity_raw"] *= rft_cap / rft_al
        touched = True
    # Re-read (may have been capped above)
    rft_al = float(df.loc[m_alr, "quantity_raw"].sum())

    # ── Aluminum hardware sets vs frame length ──
    openings = (
        max(1.0, float(math.ceil(rft_al / 11.0))) if rft_al > 1e-9 else max(1.0, float(math.ceil(0.20 * sq / 11.0)))
    )
    m_als = name_lower.str.contains("aluminum sliding", regex=False) & unit_lower.eq("set")
    ss_acc = float(df.loc[m_als, "quantity_raw"].sum())
    if m_als.any() and ss_acc + 1e-9 < openings:
        df.loc[m_als, "quantity_raw"] *= openings / max(ss_acc, 1e-6)
        touched = True

    # ── Electrical: conduit, copper runs, devices ──
    m_con = (
        phase_str.str.contains("Electrical", regex=False)
        & name_lower.str.contains("conduit", regex=False)
        & unit_lower.eq("ft")
    )
    sc = float(df.loc[m_con, "quantity_raw"].sum())
    tgt_c = max(1.22 * sq, 260.0)
    if m_con.any() and sc > 1e-9 and tgt_c > sc + 1e-9:
        df.loc[m_con, "quantity_raw"] *= tgt_c / sc
        touched = True

    def _wire_band(sub: str, tgt: float) -> None:
        nonlocal touched
        m = (
            phase_str.str.contains("Electrical", regex=False)
            & name_lower.str.contains(sub, regex=False)
            & unit_lower.eq("m")
        )
        s = float(df.loc[m, "quantity_raw"].sum())
        if m.any() and s > 1e-9 and tgt > s + 1e-9:
            df.loc[m, "quantity_raw"] *= tgt / s
            touched = True

    _wire_band("2.5mm", max(0.78 * sq, 180.0))
    _wire_band("4mm", max(0.32 * sq, 75.0))
    _wire_band("earth wire", max(0.44 * sq, 95.0))
    _wire_band("10mm2", max(0.07 * sq, 15.0))

    m_mcb = name_lower.str.contains("mcb", regex=False) & unit_lower.eq("pcs")
    sm = float(df.loc[m_mcb, "quantity_raw"].sum())
    tgt_m = max(6.0, float(math.ceil(0.034 * sq)))
    if m_mcb.any() and sm > 1e-9 and tgt_m > sm + 1e-9:
        df.loc[m_mcb, "quantity_raw"] *= tgt_m / sm
        touched = True

    m_sw = name_lower.str.contains("switches & sockets", regex=False) & unit_lower.eq("pcs")
    swp = float(df.loc[m_sw, "quantity_raw"].sum())
    tgt_sw = max(22.0, float(math.ceil(0.096 * sq)))
    if m_sw.any() and swp > 1e-9 and tgt_sw > swp + 1e-9:
        df.loc[m_sw, "quantity_raw"] *= tgt_sw / swp
        touched = True

    m_fix = (
        phase_str.str.contains("Electrical", regex=False)
        & unit_lower.eq("pcs")
        & (
            name_lower.str.contains("led", regex=False)
            | name_lower.str.contains("fan", regex=False)
            | name_lower.str.contains("panel", regex=False)
            | name_lower.str.contains("downlight", regex=False)
            | name_lower.str.contains("ceiling light", regex=False)
        )
    )
    sfx = float(df.loc[m_fix, "quantity_raw"].sum())
    tgt_fx = max(8.0, float(rooms_e + max(baths_eff, 0) + 5))
    if m_fix.any() and sfx > 1e-9 and tgt_fx > sfx + 1e-9:
        df.loc[m_fix, "quantity_raw"] *= tgt_fx / sfx
        touched = True

    return df, touched


# Simple point/fixture heuristics for labour mapping
ELECTRICAL_POINTS_PER_SQFT = 0.10  # ~1 electrical point / 10 sqft covered (basic multi-room home)
PLUMBING_FIXTURES_BASE = 8  # rough-in + fixtures; avoids collapse when bath count is unset/low

# Material quantities per sqft of total covered area
QUANTITIES_PER_SQFT: Dict[str, float] = {
    "cement_bag":  0.25,   # bags
    "brick":       12.5,   # units
    "steel_kg":    1.0,    # kg
    "sand_cft":    0.15,   # cft
    "gravel_cft":  0.08,   # cft
}

# Finishing quantities per sqft (applied on top of structural)
FINISHING_PER_SQFT: Dict[str, float] = {
    "tiles_sqft":    1.0,   # 1 sqft of tile per sqft (floors + some walls)
    "paint_liter":   0.083, # 1 liter covers ~12 sqft
    "wood_cft":      0.02,  # doors, windows
}

# Additional fixed costs per room of each type (in economy units)
ROOM_ADDITION: Dict[str, Dict[str, Any]] = {
    "bedroom": {
        "cement_bag": 20,
        "brick":      1000,
        "labour_sqft_extra": 150,
    },
    "washroom": {
        "cement_bag": 10,
        "brick":      500,
        "plumbing_lumpsum": 1,   # 1 unit = 1 full plumbing lumpsum cost
    },
    "kitchen": {
        "cement_bag": 10,
        "brick":      500,
        "electrical_lumpsum": 1,
    },
}

# City adjustment multipliers
CITY_MULTIPLIER: Dict[str, float] = {
    "Karachi":    1.05,
    "Lahore":     1.00,
    "Islamabad":  1.08,
    "Rawalpindi": 1.05,
    "Faisalabad": 0.95,
    "Multan":     0.92,
    "Peshawar":   1.02,
    "Quetta":     1.15,
    "Sialkot":    0.88,
    "Gujranwala": 0.90,
}

# Timeline cost adjustment factors
TIMELINE_FACTOR: Dict[str, float] = {
    "rush":     1.30,  # <= 3 months
    "standard": 1.00,  # <= 6 months
    "extended": 0.85,  # > 6 months
}

# Contingency percentage
CONTINGENCY_PCT = 0.15

# Prices staleness threshold (days)
PRICE_STALENESS_DAYS = 30

# Some entries in prices.json quote a price per *pack* (e.g. "1000 bricks",
# "100 cft (one trolley)"). The cost engine works in base units (per brick,
# per cft, etc.), so we divide the raw price by the pack size before
# multiplying by quantity. Keys are entry names in prices.json.
PRICE_PACK_SIZE: Dict[str, float] = {
    "brick":         1000.0,   # priced per 1000 bricks
    "sand_cft":      100.0,    # priced per 100 cft (one trolley)
    "sand_cft_fine": 100.0,    # priced per 100 cft (one trolley)
    "gravel_cft":    100.0,    # priced per 100 cft (one trolley)
}

# Display label for the base unit when a pack divisor applies.
PRICE_BASE_UNIT_LABEL: Dict[str, str] = {
    "brick":         "per brick",
    "sand_cft":      "per cft",
    "sand_cft_fine": "per cft",
    "gravel_cft":    "per cft",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LineItem:
    quantity: str
    unit: str
    unit_cost: float
    total: float


@dataclass
class EstimateResult:
    status: str
    area_sqft: float
    grade: str
    city: str
    itemized_breakdown: Dict[str, LineItem] = field(default_factory=dict)
    total_material_cost: float = 0.0
    total_labour_cost: float = 0.0
    contingency: float = 0.0
    grand_total: float = 0.0
    cost_per_sqft: float = 0.0
    cost_reduction_tips: List[str] = field(default_factory=list)
    comparison: Optional[Dict[str, float]] = None
    currency: str = "PKR"
    message: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# MODULE
# ─────────────────────────────────────────────────────────────────────────────

class CostEstimationModule:
    """
    Advanced Construction Cost Estimation Engine.

    Supports:
    - Marla / Kanal / sqft / sqm area input (parsed from text)
    - room-type additions (bedroom, washroom, kitchen)
    - Itemized material + labour breakdown
    - Dynamic cost-reduction tips
    - Comparison mode (economy / standard / premium)
    - City-based price adjustments
    - Timeline adjustment factor
    """

    def __init__(
        self,
        pricing_data_path: str = "pricing_data.csv",
        materials_master_path: str = "materials_master.csv",
        city_area_standards_path: str = "city_area_standards.csv",
        construction_rates_path: str = "construction_rates.csv",
        labor_rates_path: str = "labor_rates.csv",
        phase_labor_mapping_path: str = "phase_labor_mapping.csv",
        products_df: Optional[pd.DataFrame] = None,
        llm: Optional[LLMHelper] = None,
    ):
        self.products_df = products_df
        self.llm = llm or LLMHelper()
        # Backward-compat: keep prices.json loader for legacy helpers/tests
        # (pack-size pricing logic, unit labels, etc.). Phase-2 estimation does
        # NOT depend on this, but older endpoints/tests may.
        self.prices: Dict[str, Any] = {"items": {}}
        try:
            self._load_prices("prices.json")
        except Exception:
            pass
        # Dataframes (Phase-2)
        self.city_area: pd.DataFrame = pd.DataFrame()
        self.materials_master: pd.DataFrame = pd.DataFrame()
        self.pricing_data: pd.DataFrame = pd.DataFrame()
        self.construction_rates: pd.DataFrame = pd.DataFrame()
        self.labor_rates: pd.DataFrame = pd.DataFrame()
        self.phase_labor_mapping: pd.DataFrame = pd.DataFrame()

        # Lookups
        self._marla_sqft_by_city: Dict[str, float] = {}
        self._price_avg: Dict[Tuple[str, str], float] = {}  # (material_id, city) -> avg
        self._price_minmax: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._price_conf: Dict[Tuple[str, str], float] = {}  # (material_id, city) -> confidence_score
        self._labor_rate_avg: Dict[Tuple[str, str], float] = {}  # (city, work_type) -> avg per day
        self._bench_rates: Dict[Tuple[str, str], Tuple[float, float, float]] = {}  # (city, construction_type)->(min,max,avg)
        self._supported_pricing_cities: set = set()

        self._load_phase2_data(
            pricing_data_path=pricing_data_path,
            materials_master_path=materials_master_path,
            city_area_standards_path=city_area_standards_path,
            construction_rates_path=construction_rates_path,
            labor_rates_path=labor_rates_path,
            phase_labor_mapping_path=phase_labor_mapping_path,
        )
        self._feasibility_bands = load_feasibility_bands()
        self._floor_policy: Dict[str, Any] = load_floor_policy()
        logger.info("✓ CostEstimationModule initialised")

    def get_estimator_catalog(self) -> Dict[str, Any]:
        """Public config for UI: supported cities, feasibility bands, floor policy."""
        cities: List[Dict[str, Any]] = []
        for c in sorted(self._supported_pricing_cities):
            cities.append(
                {
                    "id": c,
                    "label": str(c).strip().title(),
                    "marla_sqft": float(self._marla_sqft_by_city.get(c, AREA_CONVERSIONS["marla"])),
                    "rates_available": True,
                }
            )
        fin_raw = finishing_catalog.load_finishing_catalog()
        fin_pub = {k: fin_raw[k] for k in ("tiers", "tier_order", "feature_labels", "package_phases", "upsells") if k in fin_raw}
        return {
            "cities": cities,
            "feasibility_bands": public_feasibility_bands_raw(),
            "floor_policy": dict(self._floor_policy),
            "finishing_catalog": fin_pub,
        }

    # ── Phase-2 loaders ───────────────────────────────────────────────────────

    def _load_phase2_data(
        self,
        pricing_data_path: str,
        materials_master_path: str,
        city_area_standards_path: str,
        construction_rates_path: str,
        labor_rates_path: str,
        phase_labor_mapping_path: str,
    ) -> None:
        """Load Phase-2 CSVs and build fast lookup maps."""
        def _read(path: str) -> pd.DataFrame:
            if os.path.exists(path):
                return pd.read_csv(path)
            # allow running from other cwd
            alt = os.path.join(os.getcwd(), path)
            if os.path.exists(alt):
                return pd.read_csv(alt)
            logger.warning("Phase-2 data file missing: %s", path)
            return pd.DataFrame()

        self.city_area = _read(city_area_standards_path)
        self.materials_master = _read(materials_master_path)
        self.pricing_data = _read(pricing_data_path)
        self.construction_rates = _read(construction_rates_path)
        self.labor_rates = _read(labor_rates_path)
        self.phase_labor_mapping = _read(phase_labor_mapping_path)

        # city marla sqft
        if not self.city_area.empty:
            for _, r in self.city_area.iterrows():
                city = str(r.get("city", "")).strip()
                sqft = float(r.get("marla_sqft", 0) or 0)
                # Normalize legacy fractional standards to avoid UI/backend rounding bugs.
                # Some datasets store "272.25" or "272.5" — we standardize these to 272.
                if abs(sqft - 272.25) < 1e-6 or abs(sqft - 272.5) < 1e-6:
                    sqft = 272.0
                if city and sqft > 0:
                    self._marla_sqft_by_city[city.lower()] = sqft

        # pricing maps
        if not self.pricing_data.empty:
            for _, r in self.pricing_data.iterrows():
                mid = str(r.get("material_id", "")).strip()
                city = str(r.get("city", "")).strip()
                if not mid or not city:
                    continue
                key = (mid, city.lower())
                avg = float(r.get("price_avg", 0) or 0)
                pmin = float(r.get("price_min", 0) or 0)
                pmax = float(r.get("price_max", 0) or 0)
                conf = float(r.get("confidence_score", 0) or 0)
                if avg > 0:
                    self._price_avg[key] = avg
                if pmin > 0 and pmax > 0:
                    self._price_minmax[key] = (pmin, pmax)
                if conf > 0:
                    self._price_conf[key] = conf

        # labour rates (per_day)
        if not self.labor_rates.empty:
            for _, r in self.labor_rates.iterrows():
                city = str(r.get("city", "")).strip().lower()
                wt = str(r.get("work_type", "")).strip()
                avg = float(r.get("rate_avg", 0) or 0)
                unit = str(r.get("unit", "")).strip().lower()
                if city and wt and avg > 0 and "per_day" in unit:
                    self._labor_rate_avg[(city, wt)] = avg

        # benchmark construction rates (per sqft)
        if not self.construction_rates.empty:
            for _, r in self.construction_rates.iterrows():
                city = str(r.get("city", "")).strip().lower()
                ctype = str(r.get("construction_type", "")).strip().lower()
                mn = float(r.get("cost_min_per_sqft", 0) or 0)
                mx = float(r.get("cost_max_per_sqft", 0) or 0)
                av = float(r.get("cost_avg_per_sqft", 0) or 0)
                if city and ctype and mn > 0 and mx > 0 and av > 0:
                    self._bench_rates[(city, ctype)] = (mn, mx, av)

        # Cities that have at least one row in pricing_data (PKR rate card).
        self._supported_pricing_cities = set()
        if not self.pricing_data.empty and "city" in self.pricing_data.columns:
            self._supported_pricing_cities = {
                str(c).strip().lower()
                for c in self.pricing_data["city"].dropna().unique()
                if str(c).strip()
            }

    # ── public API ────────────────────────────────────────────────────────────

    def estimate_from_text(self, text: str, use_llm: bool = False) -> Dict[str, Any]:
        """
        Parse a free-text query and produce a cost estimate.
        Extracts: area, city, grade, room counts from natural language.
        """
        city = self._extract_city(text)
        area_sqft = self._extract_area_from_text(text, city=city)
        grade = self._extract_grade(text)
        rooms = self._extract_rooms(text)
        ctype = self._extract_construction_type(text)
        renovation_scope = self._extract_renovation_scope(text) if ctype == "renovation" else None

        result = self.estimate_project_cost(
            sqft=int(area_sqft),
            grade=grade,
            city=city,
            construction_type=ctype,
            bedrooms=rooms.get("bedroom", 0),
            washrooms=rooms.get("washroom", 0),
            kitchens=rooms.get("kitchen", 0),
            renovation_scope=renovation_scope,
            use_llm=use_llm,
        )
        return result

    def estimate_project_cost(
        self,
        sqft: int,
        floors: int = 1,
        grade: str = "standard",
        city: str = "Lahore",
        construction_type: str = "full_construction",
        bedrooms: int = 0,
        washrooms: int = 0,
        kitchens: int = 0,
        bhk: Optional[int] = None,
        timeline_months: Optional[int] = None,
        compare: bool = False,
        use_llm: bool = False,
        renovation_scope: Optional[str] = None,
        apply_market_benchmark: bool = True,
        # ── Advanced material/system toggles (Section 7 tests) ───────────────
        wall_system: Optional[str] = None,          # red_brick | hollow_block | fly_ash_brick | aac_block
        flooring_system: Optional[str] = None,      # tile | cement_floor | marble
        tile_spec: Optional[str] = None,            # standard | imported_premium
        roof_system: Optional[str] = None,          # concrete | steel_truss | compare
        waterproofing_scope: Optional[str] = None,  # default | all_slabs
        exterior_wall_insulation: bool = False,
        window_glazing: Optional[str] = None,       # single | double
        door_frame: Optional[str] = None,           # steel | wood
        plumbing_system: Optional[str] = None,      # pvc | gi | compare
        add_ons: Optional[List[str]] = None,        # solar | rainwater_harvesting
    ) -> Dict[str, Any]:
        """
        Full cost estimate — structured input version.
        """
        grade = (grade or "standard").lower()
        if grade not in ("economy", "standard", "premium", "luxury"):
            grade = "standard"

        construction_type = (construction_type or "full_construction").lower().replace(" ", "_")
        if construction_type not in CONSTRUCTION_TYPES:
            construction_type = "full_construction"

        if sqft <= 0:
            return self._error("Area must be greater than zero.")

        city_norm = (city or "Lahore").strip()
        marla_sqft = self._marla_sqft_by_city.get(city_norm.lower(), AREA_CONVERSIONS["marla"])

        warnings: List[str] = []
        input_sqft = int(sqft)
        # Strict validation: minimum 1 marla (city-specific) or equivalent sqft.
        # City standards can be fractional (e.g. 272.25 sqft/marla). Since UI inputs
        # are integers, accept the rounded marla minimum to avoid rejecting exactly-1-marla
        # entries like 272 sqft in Rawalpindi.
        min_sqft = int(round(float(marla_sqft)))
        if int(sqft) < min_sqft:
            return self._error(
                f"Minimum area is 1 marla (~{min_sqft} sqft in {city_norm}). "
                f"Please increase area or enter marla/kanal."
            )

        if self._supported_pricing_cities and city_norm.lower() not in self._supported_pricing_cities:
            supported = ", ".join(sorted(c.title() for c in self._supported_pricing_cities))
            return self._error(
                f"City '{city_norm}' is not supported for PKR rate cards yet. Supported cities: {supported}"
            )

        layout_assumption: Optional[str] = None
        if bhk is not None:
            br, wr, kr, layout_assumption = resolve_layout_from_bhk(int(bhk))
            beds_req, baths_req, kits_req = br, wr, kr
        else:
            beds_req = int(bedrooms or 0)
            baths_req = int(washrooms or 0)
            kits_req = int(kitchens or 0)
        mxb_max, mxbt_max, mxk_max = feasibility_caps(input_sqft, self._feasibility_bands)
        bedrooms, washrooms, kitchens, layout_clamped, layout_msg = clamp_layout_inputs(
            input_sqft, beds_req, baths_req, kits_req, self._feasibility_bands
        )
        if layout_clamped and layout_msg:
            warnings.append(layout_msg)

        total_sqft = int(sqft * max(floors, 1))

        floor_factors = floor_cost_factors(floors, self._floor_policy)
        foundation_phases_f = {str(x) for x in (self._floor_policy.get("foundation_phases") or [])}
        structural_phases_f = {str(x) for x in (self._floor_policy.get("structural_per_floor_phases") or [])}

        def _phase_floor_qty_mult(phase_name: str) -> float:
            ph = str(phase_name or "")
            if ph in foundation_phases_f:
                return float(floor_factors["foundation_mult"])
            if ph in structural_phases_f:
                return float(floor_factors["structural_blend_mult"])
            return 1.0

        # Determine which phases to include
        if construction_type == "grey_structure":
            included_phases = GREY_STRUCTURE_PHASES
            phase_coverage: Optional[Dict[str, float]] = None
        elif construction_type == "renovation":
            included_phases = RENOVATION_PHASES
            phase_coverage = dict(RENOVATION_PHASE_COVERAGE)
            # If a scope is specified (paint/flooring/kitchen/bathroom), narrow phases.
            if renovation_scope:
                scope = renovation_scope.lower()
                if "paint" in scope:
                    included_phases = {"Paint & Finishing"}
                elif "floor" in scope or "tile" in scope:
                    included_phases = {"Flooring & Tiling"}
                elif "kitchen" in scope:
                    included_phases = {"Kitchen & Wardrobes", "Plumbing (Rough + Final)", "Electrical (Rough + Final)"}
                elif "bath" in scope or "wash" in scope:
                    included_phases = {"Sanitary & Bathroom Fittings", "Plumbing (Rough + Final)", "Paint & Finishing"}
        else:
            included_phases = None  # all phases
            phase_coverage = None

        # Build BOQ from materials_master ratios
        if self.materials_master.empty:
            return self._error(
                "materials_master.csv not found/loaded. Please restore Phase-2 data files."
            )

        marla_count = total_sqft / max(marla_sqft, 1.0)

        # NOTE: finishing "packages" (PKR/sqft bundles + negative offsets) produce
        # unrealistic results on small footprints (e.g., 1 marla / 225 sqft).
        # This engine must remain quantity-driven; therefore we disable bundled
        # finishing packages and always rely on itemized Phase-2 ratios.
        use_finishing_package = False
        finishing_pkg_phases: set = set()

        mm = self.materials_master.copy()
        # Standard/Economy tiers must not include high-spec roof layers.
        # (e.g., XPS insulation / geotextile) which disproportionately distort
        # small-budget estimates.
        if grade in ("standard", "economy") and "name" in mm.columns:
            n = mm["name"].fillna("").astype(str).str.lower()
            mm = mm.loc[~(n.str.contains("xps") | n.str.contains("geotextile"))].copy()
        if included_phases is not None:
            mm = mm[mm["phase"].isin(included_phases)]

        # Select ONE variant per functional group to avoid double counting
        # (e.g., Awwal + Doem bricks together).
        mm = mm.copy()
        mm["quality_grade_norm"] = mm["quality_grade"].fillna("standard").astype(str).str.lower()
        mm["functional_tag_norm"] = mm.get("functional_tag", "").fillna("").astype(str).str.strip().str.lower()
        # Grouping is per-phase to avoid collapsing unrelated materials.
        # Use subcategory (most specific) when present; else name.
        mm["group_key_local"] = (
            mm.get("subcategory", "").fillna("").astype(str).str.strip().str.lower()
        )
        mm.loc[mm["group_key_local"] == "", "group_key_local"] = (
            mm.get("name", "").fillna("").astype(str).str.strip().str.lower()
        )
        mm["group_key"] = (
            mm.get("phase", "").fillna("").astype(str).str.strip().str.lower()
            + "|" + mm["group_key_local"]
        )

        # Priority order by grade
        if grade == "economy":
            order = ["economy", "standard", "premium"]
        elif grade == "standard":
            order = ["standard", "economy", "premium"]
        else:  # premium/luxury
            order = ["premium", "standard", "economy"]
        rank = {q: i for i, q in enumerate(order)}
        mm["q_rank"] = mm["quality_grade_norm"].apply(lambda q: rank.get(q, 99))
        # pick best-ranked variant per group
        mm_q = mm.sort_values(["group_key", "q_rank"]).groupby("group_key", as_index=False).head(1).copy()

        # Quantity calc per usage_type
        def _qty(row: pd.Series) -> float:
            ut = str(row.get("usage_type", "")).strip().lower()
            ratio = float(row.get("usage_ratio", 0) or 0)
            if ratio <= 0:
                return 0.0
            if ut == "per_sqft":
                return ratio * total_sqft
            if ut == "per_marla":
                return ratio * marla_count
            if ut in ("per_house", "per_site"):
                return ratio
            return ratio * total_sqft

        mm_q["quantity_raw"] = mm_q.apply(_qty, axis=1)
        # Per-floor structural weighting (ground vs upper floors).
        mm_q["quantity_raw"] = mm_q.apply(
            lambda r: float(r["quantity_raw"]) * _phase_floor_qty_mult(str(r.get("phase", ""))),
            axis=1,
        )

        # ── Input-driven scaling (rooms/washrooms/kitchens) ──────────────────
        # This keeps the estimator responsive to layout complexity without needing
        # a full architectural plan. We scale only the phases that genuinely grow
        # with more rooms/wet areas.
        rooms = max(int(bedrooms or 0), 0)
        baths = max(int(washrooms or 0), 0)
        kits  = max(int(kitchens or 0), 0)

        # National “typical PK single-family” anchor (not tied to BHK): scaling is
        # symmetric so fewer baths/rooms reduces wet/room-heavy BOQ vs this anchor.
        anchor_rooms, anchor_baths, anchor_kits = 3, 2, 1
        room_delta = rooms - anchor_rooms
        bath_delta = baths - anchor_baths
        kit_delta = kits - anchor_kits

        room_factor = 1.0 + room_delta * 0.055
        bath_factor = 1.0 + bath_delta * 0.14
        kit_factor = 1.0 + kit_delta * 0.10
        room_factor = min(max(room_factor, 0.88), 1.38)
        bath_factor = min(max(bath_factor, 0.82), 1.48)
        kit_factor = min(max(kit_factor, 0.88), 1.35)

        phase_multipliers: Dict[str, float] = {}
        # Internal partitions, doors, paint, electrical scale with rooms.
        for ph in ("Paint & Finishing", "Electrical (Rough + Final)", "Carpentry & Woodwork", "Masonry & Walls"):
            phase_multipliers[ph] = max(phase_multipliers.get(ph, 1.0), room_factor)
        # Wet-area phases scale with bathrooms.
        for ph in ("Plumbing (Rough + Final)", "Sanitary & Bathroom Fittings", "Flooring & Tiling", "Roofing & Waterproofing"):
            phase_multipliers[ph] = max(phase_multipliers.get(ph, 1.0), bath_factor)
        # Kitchen-specific phases scale with kitchens.
        for ph in ("Kitchen & Wardrobes", "Electrical (Rough + Final)", "Plumbing (Rough + Final)", "Flooring & Tiling"):
            phase_multipliers[ph] = max(phase_multipliers.get(ph, 1.0), kit_factor)

        if phase_multipliers:
            mm_q["quantity_raw"] = mm_q.apply(
                lambda r: float(r["quantity_raw"]) * float(phase_multipliers.get(str(r.get("phase", "")), 1.0)),
                axis=1,
            )
        if phase_coverage is not None:
            mm_q["quantity_raw"] = mm_q.apply(
                lambda r: float(r["quantity_raw"]) * float(phase_coverage.get(str(r.get("phase", "")), 0.35)),
                axis=1,
            )

        # Structural BOQ minimums are now applied inside _apply_grey_structure_boq_minimums
        # for BOTH grey_structure and full_construction.  The old ad-hoc brick patch
        # (marla_count >= 2 only) is intentionally removed — it missed 1-marla houses
        # and double-counted on top of the proper minimum function below.

        mm_q, grey_boq_boosted = _apply_grey_structure_boq_minimums(
            mm_q, float(total_sqft), str(construction_type)
        )
        if grey_boq_boosted:
            warnings.append(
                "Grey structure quantities were raised to minimum typical steel / brick / cement / "
                "aggregate intensity for covered area (PK practice band). Validate with your structural drawings."
            )

        mm_q, mep_fin_boosted = _apply_turnkey_mep_finishing_boq_minimums(
            mm_q,
            float(total_sqft),
            str(construction_type),
            int(washrooms),
            int(bedrooms),
            int(kitchens),
        )
        if mep_fin_boosted:
            warnings.append(
                "Electrical / plumbing / finishing BOQ quantities were aligned to minimum credible "
                "whole-home scope vs covered area (discrete retail units, wiring bands, drainage/runs). "
                "Validate against your drawings and tier selections."
            )

        # Price lookup helpers
        def _price_avg(material_id: str) -> float:
            key = (material_id, city_norm.lower())
            if key in self._price_avg:
                return float(self._price_avg[key])
            # fallback: average across cities
            vals = [v for (mid, _), v in self._price_avg.items() if mid == material_id]
            return float(np.mean(vals)) if vals else 0.0

        def _clamp_market_unit_rate(name: str, unit: str, unit_cost: float) -> Tuple[float, Optional[str]]:
            """
            Clamp obvious core material rates to 2025–2026 PK market bands (user-provided).
            Returns (new_unit_cost, warning_or_none).
            """
            n = (name or "").lower()
            u = (unit or "").lower()
            x = float(unit_cost or 0.0)
            if x <= 0:
                return x, None

            # Cement: PKR/bag
            if "cement" in n and ("bag" in u or "bags" in u):
                lo, hi = 1100.0, 1400.0
                if x < lo or x > hi:
                    return min(max(x, lo), hi), f"Clamped cement rate to market band {int(lo)}–{int(hi)} PKR/bag (was {x:.0f})."

            # Steel: PKR/ton => 230–270 PKR/kg
            if ("steel" in n or "rebar" in n or "sarya" in n) and "kg" in u:
                lo, hi = 230.0, 270.0
                if x < lo or x > hi:
                    return min(max(x, lo), hi), f"Clamped steel rate to market band {int(lo)}–{int(hi)} PKR/kg (was {x:.0f})."

            # Bricks: PKR/brick
            if ("brick" in n or "bricks" in n) and ("pcs" in u or "piece" in u):
                lo, hi = 15.0, 22.0
                if x < lo or x > hi:
                    return min(max(x, lo), hi), f"Clamped brick rate to market band {int(lo)}–{int(hi)} PKR/brick (was {x:.1f})."

            return x, None

        def _price_conf(material_id: str) -> float:
            key = (material_id, city_norm.lower())
            if key in self._price_conf:
                return float(self._price_conf[key])
            vals = [v for (mid, _), v in self._price_conf.items() if mid == material_id]
            return float(np.mean(vals)) if vals else 0.0

        def _price_range(material_id: str) -> Tuple[float, float]:
            key = (material_id, city_norm.lower())
            if key in self._price_minmax:
                return self._price_minmax[key]
            # fallback: min/max across cities
            vals = [v for (mid, _), v in self._price_minmax.items() if mid == material_id]
            if vals:
                mins = [a for a, _ in vals]
                maxs = [b for _, b in vals]
                return float(min(mins)), float(max(maxs))
            return 0.0, 0.0

        def _round_qty(name: str, unit: str, qty: float) -> float:
            """Realism rounding rules."""
            n = (name or "").lower()
            u = (unit or "").lower()
            if qty <= 0 or not np.isfinite(qty):
                return 0.0
            if "brick" in n and ("pcs" in u or "piece" in u):
                return float(max(500, int(round(qty / 500.0) * 500)))
            if "cement" in n and ("bag" in u or "bags" in u):
                return float(max(5, int(round(qty / 5.0) * 5)))
            if ("rebar" in n or "steel" in n or "sarya" in n) and "kg" in u:
                return float(max(10, int(round(qty / 10.0) * 10)))
            if "paint" in n and ("l" == u or "liter" in u):
                return float(max(1, int(round(qty))))
            if ("tile" in n or "tiles" in n) and ("sqft" in u or "ft2" in u):
                return float(max(10, int(round(qty / 10.0) * 10)))
            if ("mcb" in n or "rccb" in n or "elcb" in n) and "pcs" in u:
                return float(max(1, int(math.ceil(qty))))
            if (
                ("led" in n or "downlight" in n or "panel" in n or ("ceiling fan" in n) or ("exhaust fan" in n))
                and "pcs" in u
            ):
                return float(max(1, int(math.ceil(qty))))
            if ("sets" == u or u == "set") and ("sanitary" in n or "shower" in n):
                return float(max(1, int(math.ceil(qty))))
            if "ball valve" in n and "pcs" in u:
                return float(max(1, int(math.ceil(qty))))
            if u == "can":
                return float(max(1, int(math.ceil(qty))))
            if u == "pair":
                return float(max(1, int(math.ceil(qty))))
            if "wire" in n and ("m" in u or "meter" in u):
                return float(max(10, int(round(qty / 10.0) * 10)))
            return round(float(qty), 2)

        # Category bucket helper (used for UI summary + item tagging)
        def _bucket(phase_name: str) -> str:
            p = (phase_name or "").lower()
            if any(k in p for k in ("excavation", "foundation", "site preparation", "grey structure", "masonry")):
                return "Grey Structure"
            if "plumbing" in p or "sanitary" in p:
                return "Plumbing"
            if "electrical" in p:
                return "Electrical"
            if any(k in p for k in ("floor", "tiling", "paint", "finishing", "carpentry", "kitchen", "wardrobe", "aluminum", "glass")):
                return "Finishing"
            return "Misc"

        # Build itemized breakdown
        breakdown: Dict[str, LineItem] = {}
        materials_total = 0.0
        seen_low_conf: set = set()
        phase_totals: Dict[str, float] = {}
        item_bucket: Dict[str, str] = {}
        item_floor_factors: Dict[str, float] = {}
        for _, row in mm_q.iterrows():
            mid = str(row.get("material_id", "")).strip()
            name = str(row.get("name", "")).strip()
            unit = str(row.get("unit", "")).strip()
            phase = str(row.get("phase", "")).strip() or "Misc"
            qty = float(row.get("quantity_raw", 0) or 0)
            if not mid or not name or qty <= 0:
                continue
            unit_cost = _price_avg(mid)
            if unit_cost <= 0:
                continue
            conf = _price_conf(mid)
            if conf and conf < 0.6 and mid not in seen_low_conf:
                warnings.append(f"Low pricing confidence for {name} in {city_norm} (confidence={conf:.2f}).")
                seen_low_conf.add(mid)

            qty = _round_qty(name, unit, qty)
            if qty <= 0:
                continue
            # Apply quality multiplier by bucket (finishing/fixtures move most by tier)
            bucket = _bucket(phase)
            qmul = QUALITY_TIER_MULTIPLIERS.get(grade, QUALITY_TIER_MULTIPLIERS["standard"]).get(bucket, 1.0)
            unit_cost_adj = float(unit_cost) * float(qmul)
            unit_cost_adj, w = _clamp_market_unit_rate(name, unit, unit_cost_adj)
            if w:
                # Avoid spamming the same warning many times.
                if w not in warnings:
                    warnings.append(w)
            total = qty * unit_cost_adj
            materials_total += total
            phase_totals[phase] = float(phase_totals.get(phase, 0.0)) + float(total)
            item_bucket[name] = _bucket(phase)  # store for UI drill-down
            breakdown[name] = LineItem(
                quantity=self._fmt_qty(qty, name.lower().replace(" ", "_")),
                unit=unit or "unit",
                unit_cost=round(unit_cost_adj, 2),
                total=round(total),
            )
            pfm = _phase_floor_qty_mult(phase)
            if abs(float(pfm) - 1.0) > 1e-6:
                item_floor_factors[name] = round(float(pfm), 4)

        # (Finishing packages intentionally disabled; see note above.)

        # Labour cost using phase_labor_mapping + labor_rates
        labour_total = 0.0
        labour_by_phase: Dict[str, float] = {}
        if not self.phase_labor_mapping.empty and self._labor_rate_avg:
            # Compute a few signals for productivity units
            steel_kg = 0.0
            steel_rows = mm_q[mm_q["category"].astype(str).str.lower().eq("steel")]
            if not steel_rows.empty:
                steel_kg = float(steel_rows["quantity_raw"].sum())

            # More rooms → more points; more baths/kitchens → more fixtures.
            electrical_points = max(
                1,
                int(
                    total_sqft
                    * ELECTRICAL_POINTS_PER_SQFT
                    * (1.0 + max(-1.0, min(4.0, rooms - anchor_rooms)) * 0.08)
                ),
            )
            plumbing_fixtures = max(
                PLUMBING_FIXTURES_BASE,
                (baths * 5) + (kits * 3),
            )

            plm = self.phase_labor_mapping.copy()
            if included_phases is not None:
                plm = plm[plm["phase"].isin(included_phases)]

            for _, r in plm.iterrows():
                phase = str(r.get("phase", "")).strip()
                if use_finishing_package and phase in finishing_pkg_phases:
                    continue
                work_type = str(r.get("work_type", "")).strip()
                unit = str(r.get("unit", "")).strip().lower()
                prod = float(r.get("productivity_rate", 0) or 0)
                if not work_type or prod <= 0:
                    continue
                day_rate = self._labor_rate_avg.get((city_norm.lower(), work_type))
                if not day_rate:
                    continue

                days = 0.0
                if "sqft_per_day" in unit:
                    days = total_sqft / prod
                elif "kg_per_day" in unit:
                    days = steel_kg / prod
                elif "point_per_day" in unit:
                    days = electrical_points / prod
                elif "fixture_per_day" in unit:
                    days = plumbing_fixtures / prod
                elif "cft_per_day" in unit:
                    # fallback: approximate by area (earthwork is loosely correlated)
                    days = (total_sqft / 150) / max(prod / 180, 1e-6)

                if days <= 0:
                    continue

                cost = days * day_rate
                if phase in foundation_phases_f:
                    cost = float(cost) * float(floor_factors["foundation_mult"])
                elif phase in structural_phases_f:
                    cost = float(cost) * float(floor_factors["structural_blend_mult"])
                # Apply quality tier labour multiplier
                lmul = QUALITY_TIER_MULTIPLIERS.get(grade, QUALITY_TIER_MULTIPLIERS["standard"]).get("labour", 1.0)
                cost = float(cost) * float(lmul)
                labour_total += cost
                labour_by_phase[phase or "Misc"] = float(labour_by_phase.get(phase or "Misc", 0.0)) + float(cost)

        if construction_type == "grey_structure" and float(total_sqft) > 0:
            floor_labour = float(total_sqft) * GREY_LABOUR_PKR_PER_SQFT_FLOOR
            if labour_total < floor_labour:
                adj = floor_labour - labour_total
                gkey = "Grey package labour (realism floor)"
                labour_by_phase[gkey] = float(labour_by_phase.get(gkey, 0.0)) + float(adj)
                labour_total += float(adj)

        if construction_type in ("full_construction", "renovation") and float(total_sqft) > 0:
            sqf = float(total_sqft)

            el_floor = sqf * ELECTRICAL_LABOUR_PKR_PER_SQFT_FLOOR
            el_cur = sum(
                float(v)
                for ph, v in labour_by_phase.items()
                if "electrical" in str(ph).lower()
            )
            if el_cur + 1e-6 < el_floor:
                adj = el_floor - el_cur
                nk = "Electrical (labour realism floor)"
                labour_by_phase[nk] = float(labour_by_phase.get(nk, 0.0)) + adj
                labour_total += adj

            pl_floor = sqf * PLUMBING_LABOUR_PKR_PER_SQFT_FLOOR
            pl_cur = sum(
                float(v)
                for ph, v in labour_by_phase.items()
                if ("plumbing" in str(ph).lower() or "sanitary" in str(ph).lower())
            )
            if pl_cur + 1e-6 < pl_floor:
                adj = pl_floor - pl_cur
                nk = "Plumbing & sanitary (labour realism floor)"
                labour_by_phase[nk] = float(labour_by_phase.get(nk, 0.0)) + adj
                labour_total += adj

            finish_keys = (
                "plaster",
                "floor",
                "tile",
                "paint",
                "carpentry",
                "kitchen",
                "wardrobe",
                "aluminum",
                "glass",
            )
            fin_cur = sum(
                float(v)
                for ph, v in labour_by_phase.items()
                if any(k in str(ph).lower() for k in finish_keys)
                and "sanitary" not in str(ph).lower()
            )
            ft_floor = sqf * FINISHING_TRADES_LABOUR_PKR_PER_SQFT_FLOOR
            if fin_cur + 1e-6 < ft_floor:
                adj = ft_floor - fin_cur
                nk = "Finishing trades (labour realism floor)"
                labour_by_phase[nk] = float(labour_by_phase.get(nk, 0.0)) + adj
                labour_total += adj

        # Tiered contingency
        contingency_pct = float(CONTINGENCY_BY_GRADE.get(grade, 0.07))
        subtotal = materials_total + labour_total
        contingency = subtotal * contingency_pct
        grand_total = subtotal + contingency
        cost_per_sqft = grand_total / max(total_sqft, 1)

        # ── Post-fix: septic sanity — grade price cap by household size ─────────
        # A full 3-chamber in-situ concrete tank (Rs 80–100k) is overkill for small
        # homes.  Graded caps reflect actual pre-cast concrete ring alternatives:
        #   1-bed (≤2 occupants) — 3-ring pre-cast set + lid: Rs 25–28k  → cap 28 000
        #   2-bed (3–4 occupants) — 4-ring set:                Rs 30–35k  → cap 35 000
        #   3-bed (5–6 occupants) — 5-ring set:                Rs 35–42k  → cap 42 000
        # Larger homes may legitimately need a full masonry tank.
        if construction_type == "full_construction":
            occupants = (bedrooms * 2 + 1) if bedrooms else None
            if occupants is not None:
                if occupants <= 4:
                    # 0–2 bed household: pre-cast 3-ring set + lid + labour ≈ Rs 25–28k
                    septic_cap = 28_000.0
                    septic_note = "Rs 28,000 (pre-cast 3-ring set, 1–2 bed house)"
                elif occupants <= 6:
                    # 3 bed: 4-ring set ≈ Rs 32–35k
                    septic_cap = 35_000.0
                    septic_note = "Rs 35,000 (pre-cast 4-ring set, 3 bed house)"
                elif occupants <= 8:
                    # 4 bed: 5-ring set ≈ Rs 38–42k
                    septic_cap = 42_000.0
                    septic_note = "Rs 42,000 (pre-cast 5-ring set, 4 bed house)"
                else:
                    septic_cap = None
                    septic_note = ""
                if septic_cap is not None:
                    for k in list(breakdown.keys()):
                        if "septic" in str(k).lower():
                            li = breakdown[k]
                            if (li.unit or "").lower() in ("lot", "lumpsum") and float(li.total or 0) > septic_cap:
                                warnings.append(
                                    f"Septic allowance adjusted to {septic_note} (pre-cast ring alternative)."
                                )
                                breakdown[k] = LineItem(
                                    quantity=li.quantity, unit=li.unit,
                                    unit_cost=septic_cap, total=septic_cap,
                                )
                                delta = float(li.total or 0) - septic_cap
                                materials_total = max(0.0, float(materials_total) - delta)
                                subtotal = float(materials_total) + float(labour_total)
                                contingency = float(subtotal) * float(contingency_pct)
                                grand_total = float(subtotal) + float(contingency)
                                cost_per_sqft = grand_total / max(total_sqft, 1)
                            break

        # Add labour as explicit line-items (for consistent category totals in UI)
        for ph, tot in labour_by_phase.items():
            if tot <= 0:
                continue
            key = f"Labour — {ph}"
            breakdown[key] = LineItem(
                quantity="1",
                unit="lumpsum",
                unit_cost=round(float(tot), 2),
                total=round(float(tot)),
            )
            item_bucket[key] = _bucket(ph)
            lmf = _phase_floor_qty_mult(ph)
            if abs(float(lmf) - 1.0) > 1e-6:
                item_floor_factors[key] = round(float(lmf), 4)

        # Benchmark enforcement (70%–130% of benchmark total)
        bench_key = (city_norm.lower(), "grey_structure" if construction_type == "grey_structure" else "turnkey")
        bench = None if construction_type == "renovation" else self._bench_rates.get(bench_key)
        adjust_added = 0.0
        bench_info: Optional[Dict[str, Any]] = None
        if bench:
            mn, mx, av = bench
            qf = (
                BENCHMARK_QUALITY_FACTOR_GREY.get(grade, 1.0)
                if construction_type == "grey_structure"
                else BENCHMARK_QUALITY_FACTOR_TURNKEY.get(grade, 1.0)
            )
            # Layout complexity should influence the market floor.
            # Otherwise, different BHK / bathroom counts can be "flattened" by the benchmark floor,
            # yielding non-monotonic grand totals (more rooms costing the same or slightly less).
            #
            # We apply only a small *positive-only* multiplier so simpler layouts don't get inflated.
            anchor_rooms, anchor_baths, anchor_kits = 3, 2, 1
            rooms_i = max(int(bedrooms or 0), 0)
            baths_i = max(int(washrooms or 0), 0)
            kits_i = max(int(kitchens or 0), 0)
            # Per extra room/wet area, lift benchmark floor slightly.
            # Tuned so that common "more beds/baths" inputs reliably increase the final total even
            # when the benchmark floor is the dominant driver.
            extra_rooms = max(0, rooms_i - anchor_rooms)
            extra_baths = max(0, baths_i - anchor_baths)
            extra_kits = max(0, kits_i - anchor_kits)
            layout_floor_mult = 1.0 + (0.040 * extra_rooms) + (0.060 * extra_baths) + (0.040 * extra_kits)
            bench_total = float(av) * float(total_sqft) * float(qf) * float(layout_floor_mult)
            if grand_total < BENCHMARK_LOW * bench_total and bool(apply_market_benchmark):
                # Apply a transparent floor adjustment to reach the minimum realistic band.
                floor_target = BENCHMARK_LOW * bench_total
                adjust = floor_target - grand_total
                if adjust > 0:
                    breakdown["Market benchmark adjustment"] = LineItem(
                        quantity="1",
                        unit="lumpsum",
                        unit_cost=round(adjust, 2),
                        total=round(adjust),
                    )
                    item_bucket["Market benchmark adjustment"] = "Misc"
                    adjust_added = float(adjust)
                    subtotal += adjust
                    contingency = subtotal * contingency_pct
                    grand_total = subtotal + contingency
                    cost_per_sqft = grand_total / max(total_sqft, 1)
                warnings.append(
                    f"Under-estimation warning: BOQ total is below {int(BENCHMARK_LOW*100)}% of benchmark "
                    f"for {city_norm} ({int(av)} PKR/sqft avg)."
                )
            if grand_total > BENCHMARK_HIGH * bench_total:
                warnings.append(
                    f"Over-estimation warning: BOQ total exceeds {int(BENCHMARK_HIGH*100)}% of benchmark "
                    f"for {city_norm} ({int(av)} PKR/sqft avg)."
                )
            bench_info = {
                "benchmark_type": "grey_structure" if construction_type == "grey_structure" else "turnkey",
                "benchmark_avg_pkr_per_sqft": float(av),
                "quality_factor": float(qf),
                "benchmark_total_pkr": float(bench_total),
                "floor_pct": float(BENCHMARK_LOW),
                "floor_target_pkr": float(BENCHMARK_LOW * bench_total),
                "applied": bool(apply_market_benchmark and adjust_added > 0),
                "adjustment_pkr": float(adjust_added),
                "layout_floor_mult": float(layout_floor_mult),
            }

        # ── BOQ integrity checks (critical material presence) ─────────────────
        # If the dataset or inputs remove these, fail safely (engineered tests).
        # Only enforce for structural build types (grey/full). Renovation scopes can legitimately omit these.
        if construction_type in ("grey_structure", "full_construction"):
            has_cement = any(
                ("cement" in str(k).lower()) and ("solvent" not in str(k).lower()) for k in breakdown.keys()
            )
            has_sand = any("sand" in str(k).lower() for k in breakdown.keys())
            if not has_cement:
                return self._error("Critical BOQ error: cement is missing; estimate is invalid.")
            if not has_sand:
                return self._error("Critical BOQ error: sand is missing; estimate is invalid.")

        # ── Steel realism warnings (over/under reinforcement) ─────────────────
        try:
            steel_total_kg = 0.0
            for k, li in breakdown.items():
                if "steel" in str(k).lower() and str(li.unit or "").lower() == "kg":
                    steel_total_kg += float(li.quantity or 0)
            if float(total_sqft) > 0 and steel_total_kg > 0:
                kg_per_sqft = steel_total_kg / float(total_sqft)
                typical = GREY_BOQ_MIN_STEEL_KG_PER_SQFT if construction_type == "grey_structure" else 3.2
                if kg_per_sqft > typical * 1.8:
                    warnings.append("⚠ Steel quantity appears unusually high for covered area; possible over-engineering.")
                if kg_per_sqft < typical * 0.65:
                    warnings.append("⚠ Steel quantity appears low for covered area; risk of under-reinforcement.")
        except Exception:
            pass

        # Fallback warning heuristic by cost share (handles cases where unit labels differ)
        try:
            steel_cost = sum(
                float(li.total or 0.0)
                for k, li in breakdown.items()
                if "steel" in str(k).lower() and "labour" not in str(k).lower()
            )
            if float(subtotal) > 0 and steel_cost > 0:
                share = steel_cost / float(subtotal)
                if share > 0.22:
                    warnings.append("⚠ Steel cost share is high vs typical residential; possible over-engineering.")
                if share < 0.06:
                    warnings.append("⚠ Steel cost share is low vs typical residential; risk of under-reinforcement.")
        except Exception:
            pass

        # ── Advanced material/system adjustments (adds separate line items) ──
        add_ons_norm = [str(x).strip().lower() for x in (add_ons or []) if str(x).strip()]
        wall_sys = (wall_system or "").strip().lower() or "red_brick"
        floor_sys = (flooring_system or "").strip().lower() or "tile"
        tile_spec_n = (tile_spec or "").strip().lower() or "standard"
        roof_sys = (roof_system or "").strip().lower() or "concrete"
        wp_scope = (waterproofing_scope or "").strip().lower() or "default"
        glazing = (window_glazing or "").strip().lower() or "single"
        frame = (door_frame or "").strip().lower() or "steel"
        pl_sys = (plumbing_system or "").strip().lower()

        roof_compare: Optional[Dict[str, float]] = None
        plumbing_compare: Optional[Dict[str, float]] = None

        def _add_adjustment(label: str, amount: float) -> None:
            nonlocal subtotal, contingency, grand_total, cost_per_sqft
            if abs(float(amount)) < 1e-6:
                return
            breakdown[label] = LineItem(
                quantity="1",
                unit="lumpsum",
                unit_cost=round(float(amount), 2),
                total=round(float(amount)),
            )
            item_bucket[label] = "Misc"
            subtotal = float(subtotal) + float(amount)
            contingency = float(subtotal) * float(contingency_pct)
            grand_total = float(subtotal) + float(contingency)
            cost_per_sqft = float(grand_total) / max(float(total_sqft), 1.0)

        wall_cost = sum(
            float(li.total or 0.0)
            for k, li in breakdown.items()
            if ("brick" in str(k).lower() or "block" in str(k).lower()) and "labour" not in str(k).lower()
        )
        # Apply wall-system as a *total* delta: wall lines are a small subset of BOQ,
        # but the product requirement expects system-level impact on total.
        if wall_sys in ("hollow_block", "hollow_blocks", "concrete_block", "block"):
            _add_adjustment(
                "Wall system adjustment — hollow block (≈15% cheaper than red brick)",
                -0.15 * float(grand_total),
            )
        elif wall_sys in ("fly_ash", "fly_ash_brick", "flyash"):
            _add_adjustment(
                "Wall system adjustment — fly-ash brick (≈10% cheaper than red brick)",
                -0.10 * float(grand_total),
            )
        elif wall_sys in ("aac", "aac_block", "aac_blocks"):
            _add_adjustment(
                "Wall system adjustment — AAC block (≈5% above red brick)",
                +0.05 * float(grand_total),
            )

        if tile_spec_n in ("imported", "imported_premium", "premium_imported"):
            _add_adjustment("Imported premium tiles add-on (finishing upgrade)", 0.12 * float(grand_total))

        if floor_sys in ("cement", "cement_floor", "screed"):
            tile_lines = sum(
                float(li.total or 0.0)
                for k, li in breakdown.items()
                if ("tile" in str(k).lower() or "flooring" in str(k).lower()) and "labour" not in str(k).lower()
            )
            if tile_lines > 0:
                _add_adjustment("Flooring change — cement floor only (reduce tiling scope ~40%)", -0.40 * float(tile_lines))

        if floor_sys in ("marble",):
            _add_adjustment("Marble flooring upgrade (vs ceramic baseline)", 0.15 * float(grand_total))

        if wp_scope in ("all_slabs", "all", "full"):
            _add_adjustment("Waterproofing scope — all slabs (3–5% add-on)", 0.04 * float(grand_total))

        if bool(exterior_wall_insulation):
            _add_adjustment("Thermal insulation on exterior walls (4–8% of wall package)", 0.03 * float(grand_total))

        if glazing in ("double", "double_glazed", "double-glazed"):
            _add_adjustment("Double-glazed windows upgrade", 0.04 * float(grand_total))

        if frame in ("wood", "timber"):
            _add_adjustment("Wood door frames upgrade (vs steel)", 0.02 * float(grand_total))

        plumbing_cost = sum(
            float(li.total or 0.0)
            for k, li in breakdown.items()
            if ("plumb" in str(k).lower() or "sanitary" in str(k).lower() or "upvc" in str(k).lower() or "pipe" in str(k).lower())
            and "labour" not in str(k).lower()
        )
        if pl_sys in ("pvc", "upvc"):
            _add_adjustment("Plumbing system — PVC/UPVC (≈30% cheaper than GI)", -0.30 * float(plumbing_cost))
        elif pl_sys in ("compare", "both"):
            gi_total = float(grand_total)
            pvc_total = float(gi_total) - 0.30 * float(plumbing_cost) * (1.0 + float(contingency_pct))
            plumbing_compare = {"gi": gi_total, "pvc": pvc_total, "plumbing_cost_basis": float(plumbing_cost)}

        roof_pkg = sum(
            float(li.total or 0.0)
            for k, li in breakdown.items()
            if ("roof" in str(k).lower() or "slab" in str(k).lower() or "waterproof" in str(k).lower())
            and "labour" not in str(k).lower()
        )
        # Ensure the compare has a meaningful basis even if roof lines are thin in the catalog.
        roof_pkg = max(float(roof_pkg), 0.20 * float(grand_total))
        if roof_sys in ("steel_truss", "truss"):
            _add_adjustment("Roof system — steel truss (≈20% cheaper roof package)", -0.20 * float(roof_pkg))
        elif roof_sys in ("compare", "both"):
            conc_total = float(grand_total)
            truss_total = float(conc_total) - 0.20 * float(roof_pkg) * (1.0 + float(contingency_pct))
            roof_compare = {"concrete": conc_total, "steel_truss": truss_total, "roof_pkg_basis": float(roof_pkg)}

        if "solar" in add_ons_norm and construction_type != "grey_structure":
            _add_adjustment("Solar panels add-on (separate system, not in grey structure)", 0.08 * float(grand_total))
        if "rainwater_harvesting" in add_ons_norm or "rainwater" in add_ons_norm:
            _add_adjustment("Rainwater harvesting system add-on", 0.02 * float(grand_total))

        # ── Layout complexity adjustments ─────────────────────────────────────
        # Prior versions used signed lump-sum "layout adders" vs a 3BHK anchor.
        # That caused absurd negative finishing totals for small homes (e.g. 1 marla / 1 BHK),
        # broke item→phase reconciliation (lumpsums), and violated benchmark guards.
        #
        # We rely on quantity-driven phase scaling (room_factor/bath_factor/kit_factor)
        # and explicit BOQ minimums instead. No signed lump-sum offsets.

        # Category breakdown for UI charts:
        # include materials + labour and allocate contingency (+ benchmark adjustment impact) proportionally
        category_pre: Dict[str, float] = {}
        for name, li in breakdown.items():
            cat = item_bucket.get(name) or "Misc"
            category_pre[cat] = float(category_pre.get(cat, 0.0)) + float(li.total or 0.0)

        pre_total = sum(category_pre.values()) or 1.0
        # Contingency allocated proportionally across categories
        contingency_by_cat: Dict[str, float] = {
            c: (float(v) / pre_total) * float(contingency) for c, v in category_pre.items()
        }
        # Benchmark adjustment already included as a line item; its extra contingency effect is within `contingency`
        # and is naturally allocated above.

        category_breakdown: Dict[str, int] = {}
        for c, v in category_pre.items():
            category_breakdown[c] = int(round(float(v) + float(contingency_by_cat.get(c, 0.0))))

        # ── Phase-wise + floor-wise totals (auditability) ─────────────────────
        # Phase totals combine material+labour by phase and allocate contingency proportionally.
        phase_pre: Dict[str, float] = {}
        for ph, amt in phase_totals.items():
            phase_pre[ph] = float(phase_pre.get(ph, 0.0)) + float(amt or 0.0)
        for ph, amt in labour_by_phase.items():
            # labour_by_phase can include synthetic labels; keep them as phases.
            phase_pre[ph] = float(phase_pre.get(ph, 0.0)) + float(amt or 0.0)
        phase_base_total = float(sum(phase_pre.values()) or 1.0)
        phase_breakdown: Dict[str, int] = {}
        for ph, v in phase_pre.items():
            phase_breakdown[ph] = int(round(float(v) + (float(v) / phase_base_total) * float(contingency)))

        # Floor-wise totals:
        # - Foundation/site setup happens only on ground.
        # - Remainder distributed by per-floor multipliers.
        floor_breakdown: Dict[str, int] = {}
        if int(floors) <= 1:
            floor_breakdown["ground"] = int(round(float(grand_total)))
        else:
            per_floor = list(floor_factors.get("per_floor_multipliers") or [])
            if not per_floor:
                per_floor = [1.0] * int(floors)
            foundation_share = 0.18  # typical residential: excavation/foundation/site set-up share (guideline)
            base = float(grand_total)
            ground = base * foundation_share
            rem = base * (1.0 - foundation_share)
            s = float(sum(per_floor)) or 1.0
            for i in range(int(floors)):
                key = "ground" if i == 0 else f"floor_{i+1}"
                floor_breakdown[key] = int(round((ground if i == 0 else 0.0) + rem * (float(per_floor[i]) / s)))

        pricing_notes: List[str] = []
        # Keep a stable UX note for clients/tests: bundled packages exist conceptually,
        # but this estimator intentionally runs itemized (quantity-driven) pricing.
        if use_finishing_package:
            pricing_notes.append(
                "Finishing tier includes a bundled PKR/sqft package from `config/finishingTiers.json`."
            )
        else:
            pricing_notes.append(
                "Bundled finishing packages are disabled for realism on small plots; estimates use itemized BOQ + labour floors."
            )
        if construction_type == "grey_structure":
            pricing_notes.append(
                "Grey structure mode excludes finishing-phase materials. Layout adders (wet-room PKR deltas) apply only to "
                "full construction; grey totals use BOQ minimum bands for structural completeness plus a labour floor vs covered area."
            )
        if layout_assumption:
            pricing_notes.append(layout_assumption)

        # ── Cost reduction tips (rule-based; optionally augmented by LLM)
        tips = self._generate_tips(breakdown, grand_total, grade, city)
        llm_tips: List[str] = []
        if use_llm:
            try:
                llm_tips = self.llm.cost_saving_tips(
                    total_pkr=grand_total, city=city, grade=grade, max_tips=2,
                )
            except Exception as exc:
                logger.warning("LLM cost tips skipped: %s", exc)
                llm_tips = []

        # ── Comparison mode
        comparison: Optional[Dict[str, float]] = None
        if compare:
            comparison = {}
            for g in ("economy", "standard", "premium"):
                sub_result = self.estimate_project_cost(
                    sqft=input_sqft, floors=floors, grade=g, city=city,
                    construction_type=construction_type,
                    bedrooms=bedrooms, washrooms=washrooms, kitchens=kitchens,
                    bhk=bhk,
                    timeline_months=timeline_months, compare=False,
                    apply_market_benchmark=apply_market_benchmark,
                )
                comparison[g.capitalize()] = sub_result["breakdown"]["summary"]["grand_total"]

        # ── Serialise LineItems for JSON response
        breakdown_serial: Dict[str, Any] = {}
        for k, v in breakdown.items():
            row: Dict[str, Any] = {
                "quantity": v.quantity,
                "unit": v.unit,
                "unit_cost": v.unit_cost,
                "total": v.total,
                "category": item_bucket.get(k),
            }
            if k in item_floor_factors:
                row["floor_factor"] = item_floor_factors[k]
            breakdown_serial[k] = row

        return {
            "status": "success",
            "project": {
                "sqft": input_sqft,
                "total_sqft": total_sqft,
                "floors": floors,
                "quality": grade.capitalize(),
                "city": city,
                "construction_type": construction_type,
                "timeline_months": timeline_months,
                "bhk": int(bhk) if bhk is not None else None,
                "layout_assumption": layout_assumption,
                "bedrooms": bedrooms,
                "washrooms": washrooms,
                "kitchens": kitchens,
                "bedrooms_requested": beds_req,
                "washrooms_requested": baths_req,
                "kitchens_requested": kits_req,
            },
            "pricing_notes": pricing_notes or None,
            "advanced_materials": {
                "wall_system": wall_sys,
                "flooring_system": floor_sys,
                "tile_spec": tile_spec_n,
                "roof_system": roof_sys,
                "waterproofing_scope": wp_scope,
                "exterior_wall_insulation": bool(exterior_wall_insulation),
                "window_glazing": glazing,
                "door_frame": frame,
                "plumbing_system": pl_sys or None,
                "add_ons": add_ons_norm,
                "roof_compare": roof_compare,
                "plumbing_compare": plumbing_compare,
            },
            "feasibility": {
                "plot_sqft": int(input_sqft),
                "max_bedrooms": int(mxb_max),
                "max_washrooms": int(mxbt_max),
                "max_kitchens": int(mxk_max),
                "clamped": bool(layout_clamped),
                "message": layout_msg or None,
            },
            "floor_factors": {
                **floor_factors,
                "foundation_phases": sorted(foundation_phases_f),
                "structural_per_floor_phases": sorted(structural_phases_f),
            },
            "breakdown": {
                "summary": {
                    "total_sqft": int(total_sqft),
                    "total_material": round(materials_total),
                    "total_labour": round(labour_total),
                    "subtotal": round(subtotal),
                    "contingency_pct": contingency_pct,
                    "contingency": round(contingency),
                    "grand_total": round(grand_total),
                }
            },
            "itemized_breakdown": breakdown_serial,
            "category_breakdown": category_breakdown,
            "phase_breakdown": phase_breakdown,
            "floor_breakdown": floor_breakdown,
            "cost_per_sqft": round(cost_per_sqft),
            "cost_reduction_tips": tips,
            "llm_cost_tips": llm_tips or None,
            "comparison": comparison,
            "currency": "PKR",
            "warnings": warnings or None,
            "benchmark": bench_info,
            "rate_breakdown": {
                "quality_grade": grade,
            },
            "llm": {
                "enabled": bool(use_llm),
                "available": (use_llm and self.llm.is_available) if use_llm else None,
                "model": self.llm.model_name if use_llm else None,
            },
        }

    def estimate_phase_cost(
        self,
        phase_name: str,
        sqft: int,
        quality: str = "Standard",
        city: str = "Lahore",
    ) -> Dict[str, Any]:
        """Estimate cost for a single construction phase."""
        PHASE_COSTS = {
            "foundation":           0.08,
            "grey_structure":       0.35,
            "roofing":              0.12,
            "plumbing":             0.10,
            "electrical":           0.08,
            "flooring":             0.15,
            "finishing":            0.12,
        }
        phase_name = phase_name.lower()
        if phase_name not in PHASE_COSTS:
            return {"status": "error", "message": f"Unknown phase: {phase_name}. Choose from: {list(PHASE_COSTS)}"}

        full = self.estimate_project_cost(sqft, 1, quality.lower(), city)
        grand = full["breakdown"]["summary"]["grand_total"]
        pct = PHASE_COSTS[phase_name]
        phase_cost = grand * pct

        return {
            "status": "success",
            "phase": phase_name,
            "sqft": sqft,
            "quality": quality,
            "city": city,
            "estimated_cost": round(phase_cost),
            "percentage_of_total": f"{int(pct * 100)}%",
            "cost_per_sqft": round(phase_cost / max(sqft, 1)),
        }

    # ── v2 extraction helpers (construction type + renovation scope) ──────────

    @staticmethod
    def _extract_construction_type(text: str) -> str:
        t = (text or "").lower()
        if any(k in t for k in ("grey structure", "gray structure", "grey_structure", "structure only")):
            return "grey_structure"
        if any(k in t for k in ("renovation", "renovate", "remodel", "upgrade", "finishing only")):
            return "renovation"
        return "full_construction"

    @staticmethod
    def _extract_renovation_scope(text: str) -> Optional[str]:
        t = (text or "").lower()
        if any(k in t for k in ("paint", "repaint", "whitewash")):
            return "paint"
        if any(k in t for k in ("floor", "tile", "tiles", "marble")):
            return "flooring"
        if "kitchen" in t:
            return "kitchen"
        if any(k in t for k in ("bath", "bathroom", "washroom", "toilet")):
            return "bathroom"
        return None

    def estimate_material_cost(
        self,
        materials: List[Dict],
        city: str = "Lahore",
    ) -> Dict[str, Any]:
        """Estimate cost for a list of specific materials."""
        city_mult = CITY_MULTIPLIER.get(city, 1.0)
        total = 0
        detailed = []

        for mat in materials:
            name = mat.get("name", "").lower().replace(" ", "_")
            qty = float(mat.get("quantity", 0))
            grade = mat.get("grade", "standard")
            unit_cost = self._unit_price_per_base(name, grade) * city_mult
            item_total = unit_cost * qty
            total += item_total
            detailed.append({
                "material": name,
                "quantity": qty,
                "unit": mat.get("unit", self._unit(name)),
                "unit_cost_pkr": round(unit_cost),
                "total_cost_pkr": round(item_total),
            })

        return {
            "status": "success",
            "city": city,
            "city_multiplier": city_mult,
            "materials": detailed,
            "total_estimated_cost": round(total),
            "currency": "PKR",
        }

    def compare_quality_tiers(
        self,
        sqft: int,
        floors: int = 1,
        city: str = "Lahore",
    ) -> Dict[str, Any]:
        """Compare costs across Economy / Standard / Premium tiers."""
        comparison: Dict[str, Any] = {}
        for grade in ("economy", "standard", "premium"):
            r = self.estimate_project_cost(sqft, floors, grade, city)
            grand = r["breakdown"]["summary"]["grand_total"]
            comparison[grade.capitalize()] = {
                "total_cost": round(grand),
                "cost_per_sqft": r["cost_per_sqft"],
            }

        base = comparison["Economy"]["total_cost"]
        for g in comparison:
            diff = ((comparison[g]["total_cost"] - base) / max(base, 1)) * 100
            comparison[g]["percentage_diff_vs_economy"] = round(diff, 1)

        return {
            "status": "success",
            "project": {"sqft": sqft, "total_sqft": sqft * floors, "floors": floors, "city": city},
            "quality_comparison": comparison,
            "cost_variation": {
                "min": comparison["Economy"]["total_cost"],
                "max": comparison["Premium"]["total_cost"],
                "range": comparison["Premium"]["total_cost"] - comparison["Economy"]["total_cost"],
            },
        }

    # ── Universal (all building types) estimator ─────────────────────────────

    def _universal_rate_lookup(self, city_lower: str, key: str) -> Optional[float]:
        """
        Best-effort material rate lookup from Phase-2 pricing CSVs.
        Returns PKR per unit for logical keys:
          - cement_bag, steel_kg, sand_cft, crush_cft, brick
        """
        c = (city_lower or "").strip().lower()
        k = (key or "").strip().lower()

        # If we have materials_master/pricing_data, try semantic matching by name.
        if self.materials_master.empty or self.pricing_data.empty:
            return None

        name_col = "material_name" if "material_name" in self.materials_master.columns else None
        id_col = "material_id" if "material_id" in self.materials_master.columns else None
        if not name_col or not id_col:
            return None

        # Candidate name patterns (broad, but safe).
        patterns: List[str]
        if k == "cement_bag":
            patterns = ["cement", "opc", "portland"]
        elif k == "steel_kg":
            patterns = ["steel", "rebar", "sariya"]
        elif k == "sand_cft":
            patterns = ["sand", "ravi", "chenab"]
        elif k == "crush_cft":
            patterns = ["crush", "aggregate", "bajri", "gravel"]
        elif k == "brick":
            patterns = ["brick", "bricks", "eent"]
        else:
            return None

        # Find material_ids whose names match.
        mm = self.materials_master
        names = mm[name_col].astype(str).str.lower()
        mask = False
        for p in patterns:
            mask = mask | names.str.contains(p, na=False)
        mids = mm.loc[mask, id_col].astype(str).str.strip().tolist()
        if not mids:
            return None

        # Find the first mid that has a city rate avg.
        for mid in mids[:25]:
            avg = self._price_avg.get((mid, c))
            if avg and float(avg) > 0:
                return float(avg)

        return None

    def estimate_universal_boq(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Universal estimator: derive quantities from first principles and return
        structured JSON + validation warnings (no lump-sum packages).
        """
        from .universal_cost_engine import _RateLookup, estimate_universal

        rl = _RateLookup(get_rate_pkr=self._universal_rate_lookup)
        return estimate_universal(
            building_type=str(payload.get("building_type") or ""),
            city=str(payload.get("city") or "Lahore"),
            area=dict(payload.get("area") or {}),
            floors=payload.get("floors"),
            bhk=payload.get("bhk"),
            finishing_tier=payload.get("finishing_tier"),
            capacity=dict(payload.get("capacity") or {}),
            wall_height_ft=payload.get("wall_height_ft"),
            span_m=payload.get("span_m"),
            industrial_heavy=bool(payload.get("industrial_heavy") or False),
            rate_lookup=rl,
        )

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "status": "online",
            "prices_loaded": bool(self.prices),
            "price_items": len(self.prices.get("items", {})),
            "last_updated": self.prices.get("last_updated", "unknown"),
            "features": [
                "project_cost_estimation",
                "text_based_estimation",
                "room_based_additions",
                "itemized_breakdown",
                "cost_reduction_tips",
                "comparison_mode",
                "phase_cost_breakdown",
                "material_cost_calculator",
                "quality_tier_comparison",
                "city_based_pricing",
                "timeline_optimization",
                "marla_kanal_area_parsing",
            ],
        }

    # ── text parsers ─────────────────────────────────────────────────────────

    def parse_area(self, raw: str, city: Optional[str] = None) -> float:
        """
        Parse area from a string like '5 marla', '1 kanal', '2000 sqft', '185 sqm'.
        Returns area in sqft.
        Raises ValueError with user-readable message if unparseable.
        """
        raw = raw.strip().lower()
        # Try "number unit" or "number unit" with optional spaces
        for unit, factor in sorted(AREA_CONVERSIONS.items(), key=lambda x: -len(x[0])):
            pattern = rf"(\d+(?:\.\d+)?)\s*{re.escape(unit)}"
            m = re.search(pattern, raw)
            if m:
                value = float(m.group(1))
                if unit == "marla" and city:
                    marla_sqft = self._marla_sqft_by_city.get(city.strip().lower(), factor)
                    return value * float(marla_sqft)
                return value * factor
        raise ValueError(f"Could not parse area from: '{raw}'. Use formats like '5 marla', '2000 sqft', '1 kanal'.")

    def _extract_area_from_text(self, text: str, city: Optional[str] = None) -> float:
        """Extract area from free text; default to 1000 sqft if not found."""
        text_lower = text.lower()
        for unit, factor in sorted(AREA_CONVERSIONS.items(), key=lambda x: -len(x[0])):
            pattern = rf"(\d+(?:\.\d+)?)\s*{re.escape(unit)}"
            m = re.search(pattern, text_lower)
            if m:
                value = float(m.group(1))
                if unit == "marla" and city:
                    marla_sqft = self._marla_sqft_by_city.get(city.strip().lower(), factor)
                    return value * float(marla_sqft)
                return value * factor
        # Try standalone number (assume sqft)
        m = re.search(r"\b(\d{3,5})\b", text_lower)
        if m:
            return float(m.group(1))
        return 1000.0  # Sensible default

    def _extract_city(self, text: str) -> str:
        text_lower = text.lower()
        for city in CITY_MULTIPLIER:
            if city.lower() in text_lower:
                return city
        return "Lahore"

    def _extract_grade(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ("luxury", "luxurious")):
            return "luxury"
        if any(w in text_lower for w in ("premium", "high quality", "best quality")):
            return "premium"
        if any(w in text_lower for w in ("economy", "cheap", "low cost", "budget", "sasta")):
            return "economy"
        return "standard"

    def _extract_rooms(self, text: str) -> Dict[str, int]:
        rooms: Dict[str, int] = {"bedroom": 0, "washroom": 0, "kitchen": 0}
        patterns: Dict[str, List[str]] = {
            "bedroom":  [r"(\d+)\s*bed", r"(\d+)\s*room", r"(\d+)\s*kamr"],
            "washroom": [r"(\d+)\s*wash", r"(\d+)\s*bath", r"(\d+)\s*toilet", r"(\d+)\s*wc"],
            "kitchen":  [r"(\d+)\s*kitchen", r"(\d+)\s*rasoi"],
        }
        text_lower = text.lower()
        for room_type, pats in patterns.items():
            for pat in pats:
                m = re.search(pat, text_lower)
                if m:
                    rooms[room_type] = int(m.group(1))
                    break
        return rooms

    # ── pricing helpers ───────────────────────────────────────────────────────

    def _price(self, item: str, grade: str) -> float:
        """
        Raw price from prices.json (per pack, exactly as recorded). Use
        `_unit_price_per_base()` when computing line totals — that respects
        the pack size declared in prices.json (e.g. "1000 bricks").
        """
        items = self.prices.get("items", {})
        entry = items.get(item, {})
        return float(entry.get(grade, entry.get("standard", 100)))

    def _pack_size(self, item: str) -> float:
        """
        Pack size for a price entry (default 1 → already per base unit).

        Only honours the PRICE_PACK_SIZE entry when the item actually exists
        in prices.json. If it doesn't, the price came from the fallback
        (100), which is already per base unit — applying a divisor there
        would silently shrink the total.
        """
        items = self.prices.get("items", {})
        if item not in items:
            return 1.0
        return float(PRICE_PACK_SIZE.get(item, 1.0))

    def _unit_price_per_base(self, item: str, grade: str) -> float:
        """
        Per-base-unit price (PKR per brick, per cft, etc.).
        This is the price you should multiply by raw quantity.
        """
        return self._price(item, grade) / self._pack_size(item)

    def _unit(self, item: str) -> str:
        """
        Display label for the *base* unit. For pack-priced items (bricks,
        sand-trolley etc.) we override the raw "1000 bricks" label so the
        line item reads naturally: quantity × per-base-unit price = total.
        """
        if item in PRICE_BASE_UNIT_LABEL:
            return PRICE_BASE_UNIT_LABEL[item]
        items = self.prices.get("items", {})
        return items.get(item, {}).get("unit", "unit")

    @staticmethod
    def _fmt_qty(qty: float, mat: str) -> str:
        if mat == "brick":
            return f"{qty:,.0f} units"
        if "_kg" in mat:
            return f"{qty:,.0f} kg"
        if "_bag" in mat:
            return f"{qty:,.0f} bags"
        if "_cft" in mat:
            return f"{qty:,.1f} cft"
        if "_sqft" in mat:
            return f"{qty:,.0f} sqft"
        if "_liter" in mat:
            return f"{qty:,.1f} liters"
        if "_cft" in mat:
            return f"{qty:,.1f} cft"
        return f"{qty:,.1f}"

    @staticmethod
    def _parse_qty_num(qty_str: str) -> float:
        """Extract the numeric part from a formatted quantity string."""
        m = re.search(r"[\d,]+(?:\.\d+)?", qty_str.replace(",", ""))
        return float(m.group()) if m else 0.0

    @staticmethod
    def _timeline_factor(months: Optional[int]) -> float:
        if months is None:
            return 1.0
        if months <= 3:
            return TIMELINE_FACTOR["rush"]
        if months <= 6:
            return TIMELINE_FACTOR["standard"]
        return TIMELINE_FACTOR["extended"]

    # ── cost reduction tips ───────────────────────────────────────────────────

    def _generate_tips(
        self,
        breakdown: Dict[str, LineItem],
        grand_total: float,
        grade: str,
        city: str,
    ) -> List[str]:
        """
        Generate short, actionable tips.

        v2: the cost engine is driven by `materials_master.csv` + `pricing_data.csv`,
        so breakdown keys are human names, not `prices.json` keys. Keep tips generic
        and market-aligned (Pakistan), never dependent on price keys existing.
        """
        tips: List[str] = []
        grade = (grade or "standard").lower()

        # Highlight top cost drivers (by total) for relevance.
        top = sorted(
            [(k, v.total) for k, v in breakdown.items() if getattr(v, "total", 0) > 0],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        if top:
            drivers = ", ".join(k for k, _ in top)
            tips.append(f"Your biggest cost drivers are: {drivers}. Compare suppliers for these first.")

        # Grade guidance
        if grade in ("premium", "luxury"):
            tips.append("Use Premium only for wet areas/exterior; Standard is usually enough for internal walls to save cost.")
        if grade == "economy":
            tips.append("Economy grade is cost-efficient. Prioritise quality checks (fresh cement date, brick strength) to avoid rework.")

        # Market actions
        tips.append(f"In {city}, get at least 3 quotes before bulk purchasing cement/steel/tiles.")
        tips.append("Keep a 10% contingency buffer for price fluctuations and site surprises.")
        return tips[:4]

    def _load_prices(self, prices_path: str) -> None:
        """Load prices.json; log warning if file is stale."""
        checked_paths = [prices_path, "prices.json"]
        for path in checked_paths:
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        self.prices = json.load(f)
                    # Staleness check
                    last_updated_str = self.prices.get("last_updated", "")
                    if last_updated_str:
                        try:
                            last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d").date()
                            age = (date.today() - last_updated).days
                            if age > PRICE_STALENESS_DAYS:
                                logger.warning(
                                    "prices.json is %d days old (last updated %s). "
                                    "Consider updating prices for accuracy.",
                                    age, last_updated_str,
                                )
                        except ValueError:
                            pass
                    logger.info("Prices loaded from %s (%d items)", path, len(self.prices.get("items", {})))
                    return
                except Exception as exc:
                    logger.warning("Failed to load prices from %s: %s", path, exc)

        logger.error("prices.json not found — using fallback prices of 100 PKR per unit")
        self.prices = {"items": {}}

    @staticmethod
    def _error(msg: str) -> Dict[str, Any]:
        return {"status": "error", "message": msg}

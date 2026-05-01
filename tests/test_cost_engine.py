"""
Tests for CostEstimationModule's pack-size pricing fix (Step K) plus
area parsing and rule-based BOQ. These tests construct the module
against the real prices.json so the regression from the v3 change is
locked in.
"""

import pytest

from ai_modules.cost_estimation_module import (
    AREA_CONVERSIONS,
    CostEstimationModule,
    PRICE_PACK_SIZE,
)


@pytest.fixture(scope="module")
def cost():
    return CostEstimationModule()


# ── Pack-size lookup ─────────────────────────────────────────────────────────


def test_pack_size_known_brick(cost):
    assert cost._pack_size("brick") == 1000.0


def test_pack_size_known_sand(cost):
    assert cost._pack_size("sand_cft") == 100.0


def test_pack_size_unknown_item_defaults_to_1(cost):
    assert cost._pack_size("nonexistent_material") == 1.0


def test_pack_size_only_applies_when_entry_exists_in_prices(cost):
    # If we add a fake entry to PRICE_PACK_SIZE that prices.json doesn't
    # know about, the helper must NOT divide — otherwise the fallback
    # price (100) would silently shrink to (100 / pack).
    PRICE_PACK_SIZE.setdefault("definitely_not_in_prices", 50.0)
    try:
        assert cost._pack_size("definitely_not_in_prices") == 1.0
    finally:
        PRICE_PACK_SIZE.pop("definitely_not_in_prices", None)


# ── Per-base-unit price ──────────────────────────────────────────────────────


def test_brick_per_base_unit_price_is_pack_divided(cost):
    raw = cost._price("brick", "standard")        # PKR per 1000 bricks
    base = cost._unit_price_per_base("brick", "standard")
    assert base == pytest.approx(raw / 1000.0)
    # Sanity: must be a small per-brick rate, not a multi-thousand pack rate.
    assert 5.0 < base < 100.0


def test_sand_per_base_unit_price_is_pack_divided(cost):
    raw = cost._price("sand_cft", "standard")     # PKR per 100 cft
    base = cost._unit_price_per_base("sand_cft", "standard")
    assert base == pytest.approx(raw / 100.0)


def test_labour_has_no_pack_divisor(cost):
    raw = cost._price("labour_sqft", "standard")
    base = cost._unit_price_per_base("labour_sqft", "standard")
    assert base == raw


# ── Display unit ─────────────────────────────────────────────────────────────


def test_unit_label_overridden_for_pack_priced(cost):
    assert cost._unit("brick") == "per brick"
    assert cost._unit("sand_cft") == "per cft"


def test_unit_label_passthrough_for_unpacked(cost):
    # Falls through to the prices.json declared unit.
    assert cost._unit("cement_bag") == "50kg bag"


# ── End-to-end estimate sanity ──────────────────────────────────────────────


def test_5_marla_standard_total_is_realistic(cost):
    """
    5 marla ≈ 1361 sqft. Before the Step K fix the brick line alone was
    ~PKR 355M (wrong by 1000×). A realistic Standard build should land in
    the low single-digit millions per BOQ rules.
    """
    result = cost.estimate_project_cost(
        sqft=1361, grade="standard", city="Lahore",
    )
    grand_total = result["breakdown"]["summary"]["grand_total"]
    # Phase-2 engine uses city benchmark rates; Lahore turnkey typically
    # lands around ~7k PKR/sqft. For 1361 sqft, expect high single-digit to
    # low double-digit millions.
    assert 6_000_000 < grand_total < 14_000_000, f"unrealistic: {grand_total}"


def test_bathrooms_increase_cost(cost):
    base = cost.estimate_project_cost(
        sqft=1361, grade="standard", city="Lahore", bedrooms=2, washrooms=1, kitchens=1,
    )
    more = cost.estimate_project_cost(
        sqft=1361, grade="standard", city="Lahore", bedrooms=2, washrooms=3, kitchens=1,
    )
    # Symmetric wet scaling + bath-baseline allowance + layout adders vs anchor.
    assert more["breakdown"]["summary"]["grand_total"] > base["breakdown"]["summary"]["grand_total"] * 1.05


def test_bhk_change_increases_turnkey_total(cost):
    low = cost.estimate_project_cost(
        sqft=1361, grade="standard", city="Lahore", bhk=2,
    )
    high = cost.estimate_project_cost(
        sqft=1361, grade="standard", city="Lahore", bhk=4,
    )
    assert high["breakdown"]["summary"]["grand_total"] > low["breakdown"]["summary"]["grand_total"] * 1.04
    assert high["project"]["bhk"] == 4
    assert "4 BHK" in (high["project"].get("layout_assumption") or "")


def test_pricing_notes_present_for_full_construction(cost):
    r = cost.estimate_project_cost(1361, grade="standard", city="Lahore", bhk=3)
    notes = r.get("pricing_notes") or []
    assert any("bundled" in n.lower() for n in notes)


def test_bedrooms_increase_cost(cost):
    base = cost.estimate_project_cost(
        sqft=1361, grade="standard", city="Lahore", bedrooms=1, washrooms=2, kitchens=1,
    )
    more = cost.estimate_project_cost(
        sqft=1361, grade="standard", city="Lahore", bedrooms=4, washrooms=2, kitchens=1,
    )
    assert more["breakdown"]["summary"]["grand_total"] > base["breakdown"]["summary"]["grand_total"] * 1.04


def test_quality_tiers_have_clear_delta(cost):
    std = cost.estimate_project_cost(
        sqft=1361, grade="standard", city="Lahore", bedrooms=2, washrooms=2, kitchens=1,
    )
    lux = cost.estimate_project_cost(
        sqft=1361, grade="luxury", city="Lahore", bedrooms=2, washrooms=2, kitchens=1,
    )
    assert lux["breakdown"]["summary"]["grand_total"] > std["breakdown"]["summary"]["grand_total"] * 1.18


def test_unsupported_city_returns_error(cost):
    r = cost.estimate_project_cost(1000, grade="standard", city="London")
    assert r.get("status") == "error"


def test_feasibility_clamps_bedrooms_for_small_plot(cost):
    r = cost.estimate_project_cost(
        225, grade="standard", city="Lahore", bedrooms=5, washrooms=1, kitchens=1,
    )
    assert r["status"] == "success"
    assert r["feasibility"]["clamped"] is True
    assert r["project"]["bedrooms"] <= r["feasibility"]["max_bedrooms"]


def test_turnkey_mep_finishing_minimums_and_labour_floors(cost):
    """Scaled BOQ + labour floors for full construction (not limited to a single reference plot)."""
    sq = int(cost.parse_area("1 marla"))
    r = cost.estimate_project_cost(
        sqft=sq,
        grade="standard",
        city="Lahore",
        construction_type="full_construction",
        bhk=2,
    )
    assert r["status"] == "success"
    ib = r["itemized_breakdown"]
    keys_low = " ".join(ib.keys()).lower()
    # Plumbing-phase lines remain itemized; sanitary fixtures sit inside the finishing tier lumpsum.
    assert "water lift pump" in keys_low
    assert "gi gas piping" in keys_low
    assert "gypsum board ceiling" in keys_low

    el_lab = sum(
        float(v.get("total") or 0)
        for k, v in ib.items()
        if k.startswith("Labour —") and "electrical" in k.lower()
    )
    pl_lab = sum(
        float(v.get("total") or 0)
        for k, v in ib.items()
        if k.startswith("Labour —")
        and ("plumbing" in k.lower() or "sanitary" in k.lower())
    )
    assert el_lab >= 9000, el_lab
    assert pl_lab >= 9000, pl_lab


def test_grey_structure_small_house_no_negative_layout_and_realistic_band(cost):
    """1-marla-class grey must not inherit turnkey negative bath deltas; BOQ/labour stay credible."""
    r = cost.estimate_project_cost(
        sqft=225,
        grade="standard",
        city="Lahore",
        construction_type="grey_structure",
        bhk=1,
    )
    assert r["status"] == "success"
    grand = float(r["breakdown"]["summary"]["grand_total"])
    assert grand > 400_000, grand
    assert grand < 2_000_000, grand
    keys = " ".join(r.get("itemized_breakdown", {}).keys()).lower()
    assert "layout vs typical anchor" not in keys
    assert any("grey package labour (realism floor)" in k.lower() for k in r["itemized_breakdown"])
    warns = " ".join(r.get("warnings") or []).lower()
    assert "minimum typical steel" in warns or "grey structure quantities" in warns


def test_floor_factors_multi_floor(cost):
    r = cost.estimate_project_cost(900, floors=2, grade="standard", city="Lahore")
    assert r["status"] == "success"
    assert r["floor_factors"]["floors"] == 2
    assert r["floor_factors"]["structural_blend_mult"] > 0


def test_brick_line_uses_per_brick_pricing(cost):
    result = cost.estimate_project_cost(
        sqft=1000, grade="standard", city="Lahore",
    )
    # Phase-2 materials come from materials_master; bricks are named rows.
    brick_keys = [k for k in result["itemized_breakdown"].keys() if "bricks" in k.lower()]
    assert brick_keys, "Expected at least one brick line item"
    # Prefer the actual brick material line (pcs).
    chosen = None
    for k in brick_keys:
        v = result["itemized_breakdown"][k]
        if str(v.get("unit", "")).lower() in ("pcs", "piece", "pieces"):
            chosen = v
            break
    brick = chosen or result["itemized_breakdown"][brick_keys[0]]
    assert brick["unit_cost"] > 0
    assert 5 < brick["unit_cost"] < 100


# ── Area parsing ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected_sqft",
    [
        ("5 marla",       5 * 272.0),
        ("1 kanal",       1 * 5445.0),
        ("2000 sqft",     2000.0),
        ("185 sqm",       185 * 10.764),
        ("100 sqyard",    900.0),
        ("100 gaj",       900.0),
    ],
)
def test_parse_area(cost, raw, expected_sqft):
    assert cost.parse_area(raw) == pytest.approx(expected_sqft)


def test_parse_area_rejects_garbage(cost):
    with pytest.raises(ValueError):
        cost.parse_area("twelve hectares of cheese")


def test_area_conversions_constant_includes_marla_and_kanal():
    assert "marla" in AREA_CONVERSIONS
    assert "kanal" in AREA_CONVERSIONS
    assert AREA_CONVERSIONS["marla"] == 272.0

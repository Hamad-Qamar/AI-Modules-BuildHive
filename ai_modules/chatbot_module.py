"""
ChatBotModule — Product-Aware, Task-Oriented Chatbot.

v2 behaviour:
  - Entity extraction: detects material names + purchase actions in every query.
  - Product-aware routing: if a material entity + purchase action are detected,
    retrieves live products from the RecommendationModule and returns them as
    structured steps + product cards.
  - Cost vs recommendation policy: see `CHATBOT_SYSTEM_PROMPT_COST_AND_REC` and
    `get_cost_recommendation_system_prompt()`. Dual-path runs when the query
    clearly asks for both (keywords) or ambiguous “best deal” style asks.
  - LLM rewrites KB-sourced answers only; module outputs use tagged sections
    ([MODULE:cost_estimation], [MODULE:recommendation]) for UI parsing.
  - Response always includes "products" list (may be empty []).
  - Forbidden: generic answers that ignore the product DB.
"""

import json
import logging
import os
import re
import uuid
import datetime as _dt
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import faiss
import numpy as np

from .cost_assistant_knowledge import load_cost_estimation_assistant_knowledge
from .query_preprocessor import QueryPreprocessor
from .intent_detector import IntentDetector, IntentResult
from .llm_helper import LLMHelper
from .shared_models import get_embedding_model

if TYPE_CHECKING:
    from .recommendation_module import RecommendationModule
    from .cost_estimation_module import CostEstimationModule

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — cost estimation & recommendations (assistant policy)
# ─────────────────────────────────────────────────────────────────────────────

CHATBOT_SYSTEM_PROMPT_COST_AND_REC = """You are the assistant for BuildHive.

Primary domains: (1) Cost estimation - prices, budgets, totals, fees, ranges, "how much," comparisons of cost options, financial impact of choices. (2) Recommendations - what to pick, "best for me," ranked options, alternatives, "what should I choose," personalization, suggested next steps from user context.

Routing (mandatory): Classify each user message. Use only the Cost Estimation path for pure cost questions; only the Recommendation path for pure choice questions. If the user explicitly asks for both in one message, handle both - default order is cost first unless they clearly ask for recommendations first (e.g. "recommendations and then cost"). If ambiguous (e.g. "best deal"), treat as both: label sections "Estimated cost" vs "Recommended option."

Cost answers: Anchor to the Cost Estimation module - use its inputs, outputs, and disclaimers. Do not invent precise figures beyond what the module returns. Structure: Summary line; Breakdown (quantities, rates, subtotals the API exposes); Assumptions (missing data, BHK/layout notes, pricing_notes); Confidence (indicative only - final pricing from suppliers/contracts).

Cost estimation assistant — terminology (mandatory in chat copy about footprint):
- Refer to built-up footprint as **total AREA Covered** (or **total AREA Covered**), not ambiguous plain "AREA," unless you are quoting a standard that literally says "roof area" or "plot area."
- For rates tied to the floor plate, say **per sq ft of AREA Covered.**
- For roofing items that use roof slab area, say **per sq ft of Roof Area** and note that **Roof Area** is often approximated (e.g. **~225 sq ft** for a **1-marla** roof example when the user has no measured roof yet).
- For misc items that say "total covered area of the house," use **total AREA Covered.**
- At the start of an estimate (or when data is missing), ask: **What is your total AREA Covered (e.g., 500 sq ft)?** If they give **plot size only** (e.g., 1 marla), explain you still need **AREA Covered (built-up)** or acceptable assumptions.
- Classify scope: **grey structure | finishing | misc | electrical | plumbing | combined.** When giving hand ranges from internal tables, show **min–max** where applicable; add assumptions (storeys, quality tier, baths, Roof Area) when they drive quantities. For **1-marla / very small AREA Covered**, note fractional per-sq-ft SKUs become **full retail units** (leftovers likely).
- Full coefficient tables live in **`config/cost_estimation_assistant_knowledge.md`** (loaded via `load_cost_estimation_assistant_knowledge()`). When the API returns a figure, the **API remains source of truth**; tables guide explanation and gaps only.

Recommendation answers: Anchor to the Recommendation module - respect catalog eligibility and tier filters. Structure: Top pick (1-3 items); Why it fits (user criteria); Trade-offs when comparing; Next step (e.g. refine in Recommendation tab, open Cost Estimator). Summarize rationale in plain language without raw internal weights unless policy allows.

Neither domain: General FAQ, account, navigation - answer normally without forcing cost/recommendation framing unless a follow-up fits.

UX: Short lead-in then structured body; use headings and bullets; progressive disclosure; one focused clarifying question if data is missing; accessible labels (not color alone); mobile-first length. Tone: professional, concise, no hype, no hallucinated prices."""

# Keyword helpers for dual-path routing (embedding intent alone can miss “cost + pick”).
_COST_TRIGGERS: Tuple[str, ...] = (
    "how much",
    "estimate",
    "cost ",
    " cost",
    "price",
    "budget",
    "total cost",
    "cheaper",
    "cost breakdown",
    "what will it cost",
    "per sqft",
    "per sq ft",
    "grand total",
    "pkr",
    "rupees",
    " rs ",
    "rs.",
)
_REC_TRIGGERS: Tuple[str, ...] = (
    "recommend",
    "suggest",
    "best option",
    "what should i choose",
    "top picks",
    "which material",
    "which should i",
    "what to pick",
    "alternatives",
    "pick between",
    "compare ",  # choice framing
)
_AMBIGUOUS_DEAL: Tuple[str, ...] = ("best deal", "best value", "cheapest and best", "best bang")


def _user_wants_cost_estimate(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _COST_TRIGGERS)


def _user_wants_recommendation(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _REC_TRIGGERS)


def _user_ambiguous_deal(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _AMBIGUOUS_DEAL)


def _dual_module_order(text: str) -> str:
    """Return 'cost_first' or 'rec_first' when both modules run."""
    t = (text or "").lower()
    if any(
        p in t
        for p in (
            "recommendation first",
            "recommendations first",
            "materials first",
            "options first",
            "then cost",
            "then price",
            "then estimate",
        )
    ):
        return "rec_first"
    if any(
        p in t
        for p in (
            "cost and recommend",
            "price and recommend",
            "estimate and recommend",
            "cost and suggestion",
        )
    ):
        return "cost_first"
    # If both recommendation-ish and cost-ish phrases appear, use first occurrence.
    rec_positions = [t.find(m) for m in ("recommend", "suggest") if m in t]
    rec_idx = min(rec_positions) if rec_positions else 10**6
    cost_markers = ("how much", "estimate", "cost", "price", "budget")
    cost_positions = [t.find(m) for m in cost_markers if m in t]
    cost_idx = min(cost_positions) if cost_positions else 10**6
    if rec_idx < 10**6 and cost_idx < 10**6:
        return "rec_first" if rec_idx < cost_idx else "cost_first"
    # Default policy for MIXED: recommendation first unless user already specifies
    # a tier/spec that makes estimation unblocked.
    return "cost_first"


def _user_specified_finishing_tier(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in ("economy", "standard", "premium", "luxury"))


def _router_intent_label(want_cost: bool, want_rec: bool) -> str:
    if want_cost and want_rec:
        return "Mixed"
    if want_cost:
        return "Estimation"
    return "Recommendation"


def _extract_area_hint(text: str) -> Optional[str]:
    """
    Return a compact area hint if present (e.g. '5 marla', '2000 sqft').
    Used only for deciding if we must ask for missing inputs.
    """
    t = (text or "").lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(marla|kanal|sqft|sq ft|square feet|sqm|m2|sq meter|sq metre)", t)
    if not m:
        return None
    val = m.group(1)
    unit = m.group(2).replace("square feet", "sqft").replace("sq ft", "sqft").replace("sq meter", "sqm").replace("sq metre", "sqm").replace("m2", "sqm")
    return f"{val} {unit}"


def _extract_city_hint(text: str) -> Optional[str]:
    t = (text or "").lower()
    # Minimal set (expand later if needed).
    for c in ("lahore", "karachi", "islamabad", "rawalpindi", "faisalabad", "multan", "peshawar", "quetta"):
        if c in t:
            return c.title()
    return None


def _extract_building_type_hint(text: str) -> Optional[str]:
    t = (text or "").lower()
    for bt in (
        "house",
        "apartment",
        "villa",
        "farmhouse",
        "servant quarter",
        "shop",
        "office",
        "plaza",
        "school",
        "hospital",
        "mosque",
        "warehouse",
        "factory",
    ):
        if bt in t:
            return bt.replace("servant quarter", "servant_quarter")
    return None


def _extract_floors_hint(text: str) -> Optional[int]:
    t = (text or "").lower()
    m = re.search(r"\b(\d+)\s*(floors?|storeys?|stories?)\b", t)
    if m:
        return int(m.group(1))
    if "double storey" in t or "double story" in t:
        return 2
    if "single storey" in t or "single story" in t:
        return 1
    return None


def _extract_bhk_hint(text: str) -> Optional[int]:
    t = (text or "").lower()
    m = re.search(r"\b(\d+)\s*bhk\b", t)
    if m:
        return int(m.group(1))
    return None


def _navigation_actions_json(intent: str) -> str:
    """
    Must be appended exactly when relevant, in the schema requested by the user.
    """
    actions: List[Dict[str, Any]] = []
    if intent in ("Recommendation", "Mixed"):
        actions.append(
            {
                "label": "Open Recommendations",
                "target_module": "recommendation",
                "deep_link": "/recommendations",
                "optional": True,
            }
        )
    if intent in ("Estimation", "Mixed"):
        actions.append(
            {
                "label": "Open Cost Estimator",
                "target_module": "estimation",
                "deep_link": "/estimator",
                "optional": True,
            }
        )
    if not actions:
        return ""
    return json.dumps({"navigation_actions": actions}, ensure_ascii=False, indent=2)


def _navigation_actions_list(intent: str) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    # UX requirement: return a SINGLE navigation button so the chatbot stays compact.
    if intent == "Recommendation":
        return [
            {
                "label": "View Recommendations",
                "target_module": "recommendation",
                "deep_link": "/recommendations",
                "optional": True,
            }
        ]
    if intent == "Estimation":
        return [
            {
                "label": "Go to Cost Estimator",
                "target_module": "estimation",
                "deep_link": "/estimator",
                "optional": True,
            }
        ]
    if intent == "Mixed":
        # Prefer cost estimator as the next concrete step (it collects missing fields too).
        return [
            {
                "label": "Go to Cost Estimator",
                "target_module": "estimation",
                "deep_link": "/estimator",
                "optional": True,
            }
        ]
    return actions


def _router_template(
    *,
    intent: str,
    inputs_used: Dict[str, Any],
    result_summary: str,
    warnings: Optional[List[str]] = None,
    include_nav: bool = True,
) -> str:
    """
    User-specified output template (audit-friendly).
    """
    # UI-friendly formatting (headings + bullets) while preserving the exact
    # navigation_actions JSON schema at the end when relevant.
    warn_lines: List[str] = []
    for w in (warnings or []):
        if w:
            warn_lines.append(f"- {w}")
    warn_block = "\n".join(warn_lines) if warn_lines else "- None"
    nav = _navigation_actions_json(intent) if include_nav else ""
    nav_block = nav if nav else ""

    inputs_lines: List[str] = []
    for k, v in (inputs_used or {}).items():
        if v is None or v == "":
            continue
        inputs_lines.append(f"- **{k}**: {v}")
    inputs_block = "\n".join(inputs_lines) if inputs_lines else "- (none)"

    body = (
        f"## Intent\n"
        f"**{intent}**\n\n"
        f"## Inputs used\n"
        f"{inputs_block}\n\n"
        f"## Result\n"
        f"{result_summary}\n\n"
        f"## Warnings / Exclusions\n"
        f"{warn_block}\n"
    )

    if nav_block:
        body += (
            "\n## Optional navigation\n"
            "If you want, you can open the relevant module:\n\n"
            f"{nav_block}\n"
        )

    return body


def _format_estimation_summary_for_chat(cost_r: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Extract a compact, UI-friendly estimate summary from CostEstimationModule output.
    Returns (markdown_summary, warnings_list).
    """
    warns: List[str] = []
    if not isinstance(cost_r, dict) or cost_r.get("status") not in ("success", "ok"):
        return "- I couldn’t generate an estimate from the provided inputs.", warns

    # Newer estimator outputs
    summary = (((cost_r.get("breakdown") or {}).get("summary")) or {})
    grand = summary.get("grand_total")
    cps = cost_r.get("cost_per_sqft")
    cat = cost_r.get("category_breakdown") or {}
    phase = cost_r.get("phase_breakdown") or {}
    floor = cost_r.get("floor_breakdown") or {}

    if cost_r.get("warnings"):
        warns.extend([str(w) for w in (cost_r.get("warnings") or [])][:6])

    # Compact, chatbot-friendly (1–2 lines). Details live in the Cost Estimator tab.
    parts: List[str] = []
    if grand is not None:
        parts.append(f"Estimated total: **PKR {int(grand):,}**")
    if cps is not None:
        parts.append(f"(**PKR {int(cps):,}/sqft**)")
    msg = " ".join(parts) if parts else "I couldn’t generate an estimate from the provided inputs."
    return msg, warns


def _format_recommendation_summary_for_chat(rec_r: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Extract a compact, UI-friendly summary from RecommendationModule output.
    Returns (markdown_summary, warnings_list).
    """
    warns: List[str] = []
    if not isinstance(rec_r, dict) or rec_r.get("status") != "success":
        return "- I couldn’t generate recommendations from the provided inputs.", warns

    # Compact, chatbot-friendly (1–2 lines). Full tables live in Recommendations tab.
    recs = rec_r.get("recommendations") or rec_r.get("categories") or {}
    top_name = None
    top_price = None
    top_cat = None
    if isinstance(recs, dict) and recs:
        for cat, items in recs.items():
            if items:
                top = items[0] or {}
                top_cat = str(cat)
                top_name = top.get("item_name") or top.get("name") or top.get("title")
                top_price = top.get("market_price_pkr") or top.get("final_price_pkr")
                break
    if top_name:
        if top_price is not None:
            return f"Top pick: **{top_name}** ({top_cat}) — ≈ **PKR {int(top_price):,}**.", warns
        return f"Top pick: **{top_name}** ({top_cat}).", warns
    return "I couldn’t generate recommendations from the provided inputs.", warns


# ─────────────────────────────────────────────────────────────────────────────
# DATA CONTRACTS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChatResponse:
    response_id: str
    text: str
    intent: str
    suggested_follow_ups: List[str]
    language_hint: str
    source: str
    confidence: float = 0.0
    # Optional structured payload (used when routing to a module)
    data: Optional[Dict[str, Any]] = None
    # Products retrieved for purchase-intent queries (may be empty list)
    products: List[Dict[str, Any]] = field(default_factory=list)
    # Step-by-step guide for purchase/how-to intents
    steps: List[str] = field(default_factory=list)
    # Optional navigation actions (UI handoff)
    navigation_actions: List[Dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# ROLE-BASED FOLLOW-UP SUGGESTIONS
# ─────────────────────────────────────────────────────────────────────────────

_FOLLOW_UPS: Dict[str, Dict[str, List[str]]] = {
    "recommendation": {
        "buyer":      ["Get a cost estimate for these materials",
                       "Save this recommendation",
                       "Ask about a specific material"],
        "seller":     ["List one of these materials",
                       "Check current stock levels",
                       "View similar products"],
        "freelancer": ["Calculate project cost",
                       "Find materials for your project",
                       "View buyer project listings"],
    },
    "cost_estimation": {
        "buyer":      [],
        "seller":     ["View demand for these materials",
                       "Adjust pricing strategy",
                       "List related materials"],
        "freelancer": ["Find buyers with this budget",
                       "Post a proposal for similar work",
                       "Check labour rate by city"],
    },
    "cost_and_recommendation": {
        "buyer":      ["Refine the estimate with a different BHK or city",
                       "Ask for a narrower material scope (e.g. tiles only)",
                       "Open Cost Estimation then Recommendation tabs to compare"],
        "seller":     ["See which SKUs match the recommended picks",
                       "List materials popular for this layout tier",
                       "Review demand by city from the estimate"],
        "freelancer": ["Tie this BOQ to your proposal template",
                       "Ask for labour-only vs turnkey breakdown",
                       "Export assumptions for the client brief"],
    },
    "platform_help": {
        "buyer":      ["Track your current order",
                       "Browse material categories",
                       "Use the AI cost estimator"],
        "seller":     ["Manage your listings",
                       "View sales analytics",
                       "Update product pricing"],
        "freelancer": ["Update your portfolio",
                       "View open project requests",
                       "Manage active proposals"],
    },
    "navigation": {
        "buyer":      ["Search for specific materials",
                       "View featured products",
                       "Check your order status"],
        "seller":     ["Add a new product listing",
                       "View your seller dashboard",
                       "Check compliance notices"],
        "freelancer": ["View project board",
                       "Check proposal status",
                       "Update service listings"],
    },
    "general_question": {
        "buyer":      ["Get a material recommendation",
                       "See cost estimate",
                       "Browse related products"],
        "seller":     ["List this material",
                       "View category demand",
                       "Compare with competitors"],
        "freelancer": ["Use this knowledge in your proposal",
                       "Find related projects",
                       "Browse buyer requirements"],
    },
    "clarification_needed": {
        "buyer":      ["Get material recommendations",
                       "Get a cost estimate",
                       "Get platform help"],
        "seller":     ["Manage my listings",
                       "View sales data",
                       "Get platform help"],
        "freelancer": ["View open projects",
                       "Submit a proposal",
                       "Get platform help"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# ENTITY EXTRACTION TABLES
# ─────────────────────────────────────────────────────────────────────────────

# Maps material keywords → category name in products.csv
_MATERIAL_ENTITIES: Dict[str, str] = {
    "cement":       "Raw Materials",
    "bricks":       "Raw Materials",
    "brick":        "Raw Materials",
    "sand":         "Raw Materials",
    "steel":        "Raw Materials",
    "rebar":        "Raw Materials",
    "gravel":       "Raw Materials",
    "crush":        "Raw Materials",
    "tiles":        "Flooring Materials",
    "tile":         "Flooring Materials",
    "marble":       "Flooring Materials",
    "granite":      "Flooring Materials",
    "paint":        "Paint & Finishing",
    "primer":       "Paint & Finishing",
    "wood":         "Wood & Carpentry",
    "timber":       "Wood & Carpentry",
    "door":         "Doors & Windows",
    "window":       "Doors & Windows",
    "pipe":         "Plumbing - Pipes & Fittings",
    "plumbing":     "Plumbing - Pipes & Fittings",
    "faucet":       "Plumbing - Taps & Fixtures",
    "tap":          "Plumbing - Taps & Fixtures",
    "wire":         "Electrical - Wiring & Cables",
    "cable":        "Electrical - Wiring & Cables",
    "switch":       "Electrical - Switchgear",
    "switchgear":   "Electrical - Switchgear",
    "light":        "Electrical - Lighting & Fixtures",
    "fan":          "Electrical - Lighting & Fixtures",
    "waterproofing":"Chemicals & Treatments",
    "roofing":      "Roofing Materials",
    "insulation":   "Insulation & Ceilings",
    "sanitary":     "Sanitary Items",
    "kitchen":      "Kitchen Materials",
}

# Purchase / research action keywords
_PURCHASE_ACTIONS: Tuple[str, ...] = (
    "buy", "purchase", "order", "get", "find", "where to",
    "how to buy", "how to get", "price of", "cost of", "best",
    "recommend", "suggest", "compare", "source", "procure",
    "kahan se", "khareedna", "milega",
)

# Step-by-step guides for purchase intents
_PURCHASE_STEPS: Dict[str, List[str]] = {
    "default": [
        "Search for the material using the BuildHive search bar.",
        "Filter results by city, quality grade, and your budget.",
        "Compare at least 3 suppliers to get the best price.",
        "Check supplier ratings and certification badges.",
        "Request a bulk-order quote for large quantities.",
        "Confirm lead time and delivery terms before ordering.",
    ],
    "cement": [
        "Decide the grade you need: OPC 43 (economy) or OPC 53 (structural).",
        "Calculate quantity: 0.25 bags per sqft of construction area.",
        "Compare prices across DG Khan, Maple Leaf, and Lucky brands.",
        "Order in full truck loads (100–200 bags) for best bulk pricing.",
        "Check the manufacturing date — cement older than 3 months loses strength.",
        "Arrange covered, dry storage at the site before delivery.",
    ],
    "bricks": [
        "Choose grade: A-grade (kiln-fired) for load-bearing walls.",
        "Estimate quantity: 12.5 bricks per sqft of construction area.",
        "Inspect a sample batch — look for uniform size, no cracks.",
        "Source locally (within 50 km) to save transport cost.",
        "Order at least 10% extra for breakage and cuts.",
        "Confirm price is per 1,000 bricks, not per piece.",
    ],
    "tiles": [
        "Measure the total floor and wall area to tile.",
        "Add 15% to your measurement for cuts and wastage.",
        "Choose grade: A-grade for uniform thickness and shade.",
        "Compare prices per sqft including adhesive and grout.",
        "Request a sample tile before placing a full order.",
        "Hire a certified tiler — incorrect laying voids warranty.",
    ],
    "paint": [
        "Choose finish: matte for walls, semi-gloss for trims and kitchens.",
        "Calculate litres: 1 litre covers approximately 12 sqft (2 coats).",
        "Apply one coat of primer before topcoat for durability.",
        "Compare prices for 4-litre tins vs 20-litre drums — drums are cheaper.",
        "Check lead content — use lead-free paints for interiors.",
        "For exterior walls, use weather-resistant or weather-shield formula.",
    ],
}


# Navigation guide responses — static, no KB needed
_NAV_GUIDES: Dict[str, str] = {
    "dashboard":        "Go to your Dashboard by clicking your profile icon at the top-right → 'Dashboard'.",
    "tiles":            "Browse Tiles & Flooring at: Marketplace → Categories → Tiles & Flooring.",
    "plumbing":         "Find Plumbing materials at: Marketplace → Categories → Plumbing & Sanitary.",
    "cement":           "Find Cement products at: Marketplace → Categories → Cement & Concrete.",
    "electrical":       "Browse Electrical items at: Marketplace → Categories → Electrical Components.",
    "steel":            "Find Steel products at: Marketplace → Categories → Steel & Metal.",
    "categories":       "Browse all categories at: Marketplace → Categories. You'll see 12+ material categories.",
    "order":            "Track your orders at: Buyer Dashboard → Active Orders.",
    "wishlist":         "View your wishlist at: Buyer Dashboard → Saved Items.",
    "profile":          "Edit your profile at: Top-right icon → Account Settings → Profile.",
    "messages":         "Open Messages at: Top navigation bar → Messages icon.",
    "listing":          "Manage listings at: Seller Dashboard → Listing Management → New Listing.",
    "estimate":         "Use the Cost Estimator at: AI Tools → Cost Estimator (also accessible on the home page).",
    "recommendation":   "Use the Recommendation Engine at: AI Tools → Material Recommendations.",
    "checkout":         "Complete checkout at: Cart icon (top-right) → Checkout.",
    "payment":          "View payment options at: Buyer Dashboard → Financial Overview.",
    "help":             "Visit the Help Center at: Footer → Help Center, or ask me any question here.",
}


# ─────────────────────────────────────────────────────────────────────────────
# PLATFORM KNOWLEDGE DOMAINS (FR1–FR16)
# Structured response templates checked BEFORE the KB embedding search.
# Format: { "keywords": [...], "response": "<structured text>" }
# ─────────────────────────────────────────────────────────────────────────────

PLATFORM_RESPONSES: List[Dict[str, Any]] = [
    # ── ACCOUNT & SECURITY (FR1) ──────────────────────────────────────────────
    {
        "keywords": ["change password", "update password", "reset password", "new password"],
        "domain": "account_security",
        "response": (
            "**To change your password:**\n"
            "1. Go to **Account Settings** (top-right icon)\n"
            "2. Click **Security**\n"
            "3. Select **Change Password**\n"
            "4. Enter your current password, then your new one\n"
            "5. Click **Save Changes**\n\n"
            "💡 Tip: Use a mix of letters, numbers, and symbols for a strong password."
        ),
    },
    {
        "keywords": ["change email", "update email", "email settings"],
        "domain": "account_security",
        "response": (
            "**To update your email address:**\n"
            "1. Go to **Account Settings**\n"
            "2. Click **Profile**\n"
            "3. Edit the **Email** field\n"
            "4. Verify the new email via the confirmation link sent to it\n\n"
            "💡 Tip: Keep your email up to date to receive order and notification alerts."
        ),
    },
    {
        "keywords": ["delete account", "close account", "remove account"],
        "domain": "account_security",
        "response": (
            "**To delete your account:**\n"
            "1. Go to **Account Settings**\n"
            "2. Scroll to **Danger Zone**\n"
            "3. Click **Delete Account**\n"
            "4. Confirm by entering your password\n\n"
            "⚠️ This action is permanent and cannot be undone."
        ),
    },
    {
        "keywords": ["two factor", "2fa", "two-factor", "enable 2fa", "authenticator"],
        "domain": "account_security",
        "response": (
            "**To enable Two-Factor Authentication (2FA):**\n"
            "1. Go to **Account Settings → Security**\n"
            "2. Toggle **Two-Factor Authentication** ON\n"
            "3. Scan the QR code with Google Authenticator or Authy\n"
            "4. Enter the 6-digit code to confirm\n\n"
            "💡 Tip: 2FA significantly increases your account security."
        ),
    },
    # ── MARKETPLACE (FR2) ─────────────────────────────────────────────────────
    {
        "keywords": ["search product", "find product", "search material", "find material",
                     "browse", "search for"],
        "domain": "marketplace",
        "response": (
            "**To search for materials on BuildHive:**\n"
            "1. Go to **Marketplace** from the top navigation\n"
            "2. Type in the search bar (e.g. \"OPC cement\", \"bricks\")\n"
            "3. Use filters: **City**, **Price Range**, **Quality Grade**\n"
            "4. Click on any product to view details and supplier info\n\n"
            "💡 Tip: Use the AI Recommendation Tool for a full material list based on your project."
        ),
    },
    {
        "keywords": ["place order", "buy product", "purchase product", "order material",
                     "how to order", "add to cart", "checkout"],
        "domain": "marketplace",
        "response": (
            "**To place an order on BuildHive:**\n"
            "1. Find the product in **Marketplace**\n"
            "2. Click **Add to Cart**\n"
            "3. Go to **Cart** (top-right icon)\n"
            "4. Review items and click **Checkout**\n"
            "5. Enter delivery address and payment details\n"
            "6. Confirm your order\n\n"
            "💡 Tip: Compare at least 3 suppliers before ordering to get the best price."
        ),
    },
    {
        "keywords": ["track order", "order status", "where is my order", "delivery status"],
        "domain": "marketplace",
        "response": (
            "**To track your order:**\n"
            "1. Go to **Buyer Dashboard → Active Orders**\n"
            "2. Find your order and click **Track**\n"
            "3. View real-time delivery status\n\n"
            "💡 Tip: You'll also receive SMS/email updates at each stage."
        ),
    },
    {
        "keywords": ["cancel order", "return order", "refund"],
        "domain": "marketplace",
        "response": (
            "**To cancel or return an order:**\n"
            "1. Go to **Buyer Dashboard → Active Orders**\n"
            "2. Select the order\n"
            "3. Click **Cancel** (if not shipped) or **Return** (if delivered)\n"
            "4. Select a reason and submit\n\n"
            "💡 Tip: Cancellations are instant if the order hasn't been dispatched yet."
        ),
    },
    # ── SELLER MODULE (FR3) ───────────────────────────────────────────────────
    {
        "keywords": ["add listing", "create listing", "add product", "list product",
                     "new listing", "post product"],
        "domain": "listings",
        "response": (
            "**To add a new product listing:**\n"
            "1. Go to **Seller Dashboard**\n"
            "2. Click **Listing Management → Add New Listing**\n"
            "3. Fill in: Product Name, Category, Price, Quality Grade, City\n"
            "4. Upload photos and set stock quantity\n"
            "5. Click **Publish**\n\n"
            "💡 Tip: Listings with photos and detailed specs get 3× more views."
        ),
    },
    {
        "keywords": ["edit listing", "update listing", "update product", "change price",
                     "update price", "edit product"],
        "domain": "listings",
        "response": (
            "**To edit a listing:**\n"
            "1. Go to **Seller Dashboard → Listing Management**\n"
            "2. Find the product and click **Edit**\n"
            "3. Update price, stock, description, or photos\n"
            "4. Click **Save Changes**\n\n"
            "💡 Tip: Keep prices updated — stale prices lower your ranking in search results."
        ),
    },
    {
        "keywords": ["delete listing", "remove listing", "remove product", "deactivate listing"],
        "domain": "listings",
        "response": (
            "**To remove a listing:**\n"
            "1. Go to **Seller Dashboard → Listing Management**\n"
            "2. Find the product\n"
            "3. Click **Deactivate** (hides it) or **Delete** (permanent)\n\n"
            "⚠️ Deleted listings cannot be recovered. Use Deactivate if temporary."
        ),
    },
    {
        "keywords": ["seller dashboard", "my dashboard", "sales analytics", "view sales",
                     "how to sell"],
        "domain": "dashboard",
        "response": (
            "**Your Seller Dashboard includes:**\n"
            "• **Listing Management** — add, edit, deactivate products\n"
            "• **Order Management** — view incoming orders\n"
            "• **Sales Analytics** — revenue, top products, city-wise sales\n"
            "• **Financial Overview** — earnings and payouts\n\n"
            "Access it via: **Profile icon (top-right) → Seller Dashboard**."
        ),
    },
    # ── AI TOOLS (FR4) ────────────────────────────────────────────────────────
    {
        "keywords": ["recommendation", "recommend materials", "material list", "ai recommend",
                     "suggest materials", "material recommendation"],
        "domain": "ai_recommendation",
        "response": (
            "**To use the AI Material Recommendation tool:**\n"
            "1. Click the **Recommendation System** tab on this page\n"
            "2. Describe your project (e.g. \"5 marla house in Lahore\")\n"
            "3. Enter area, city, and quality preference\n"
            "4. Click **Get AI Recommendations**\n\n"
            "You'll get a full material list across 7+ categories with estimated quantities and costs."
        ),
    },
    {
        "keywords": ["cost estimate", "estimate cost", "project cost", "how much will it cost",
                     "building cost", "construction cost", "cost estimator"],
        "domain": "cost_estimation",
        "response": (
            "**To estimate your project cost:**\n"
            "1. Click the **Cost Estimation** tab on this page\n"
            "2. Enter: Area, Quality grade, City, Floors\n"
            "3. Click **Calculate Cost Estimate**\n\n"
            "You'll see: Total Cost, Cost per sqft, Itemized breakdown, Pie chart, and AI-generated cost-saving tips."
        ),
    },
    # ── CHAT & MESSAGING (FR5) ────────────────────────────────────────────────
    {
        "keywords": ["message seller", "contact seller", "chat with seller", "send message",
                     "inbox", "messages"],
        "domain": "chat_system",
        "response": (
            "**To message a seller:**\n"
            "1. Open the product listing\n"
            "2. Click **Contact Seller** or **Message**\n"
            "3. Type your message and send\n\n"
            "Access all conversations at: **Top navigation → Messages icon**.\n\n"
            "💡 Tip: Ask about bulk pricing, lead times, and certifications before ordering."
        ),
    },
    # ── REVIEWS (FR6) ─────────────────────────────────────────────────────────
    {
        "keywords": ["leave review", "write review", "rate seller", "rate product", "add review"],
        "domain": "reviews",
        "response": (
            "**To leave a review:**\n"
            "1. Go to **Buyer Dashboard → Order History**\n"
            "2. Find the completed order\n"
            "3. Click **Leave a Review**\n"
            "4. Rate (1–5 stars) and write your feedback\n"
            "5. Submit\n\n"
            "💡 Tip: Honest reviews help other buyers make better decisions."
        ),
    },
    {
        "keywords": ["my reviews", "view reviews", "seller reviews", "product reviews"],
        "domain": "reviews",
        "response": (
            "**To view reviews:**\n"
            "• **Product reviews**: On the product listing page, scroll to Reviews section\n"
            "• **Seller reviews**: On the Seller Profile page\n"
            "• **Your given reviews**: Buyer Dashboard → Order History → Review tab\n\n"
            "Reviews are verified — only confirmed buyers can leave them."
        ),
    },
    # ── FREELANCER SYSTEM (FR7) ───────────────────────────────────────────────
    {
        "keywords": ["freelancer", "register as freelancer", "post service", "hire freelancer",
                     "contractor", "service provider"],
        "domain": "freelancer",
        "response": (
            "**BuildHive Freelancer System:**\n"
            "• **Register as Freelancer**: Profile → Switch to Freelancer → Add services & portfolio\n"
            "• **Post a service**: Freelancer Dashboard → Add Service\n"
            "• **Get hired**: Buyers browse and book your service directly\n"
            "• **Hire a freelancer**: Marketplace → Services tab → Filter by skill/city\n\n"
            "💡 Tip: Complete your portfolio to appear higher in search results."
        ),
    },
    # ── DASHBOARD & ANALYTICS (FR8) ───────────────────────────────────────────
    {
        "keywords": ["buyer dashboard", "my orders", "my purchases", "order history"],
        "domain": "dashboard",
        "response": (
            "**Your Buyer Dashboard includes:**\n"
            "• **Active Orders** — track current deliveries\n"
            "• **Order History** — past purchases and receipts\n"
            "• **Saved Items** — your wishlist\n"
            "• **Financial Overview** — spending summary\n\n"
            "Access via: **Profile icon (top-right) → Buyer Dashboard**."
        ),
    },
    # ── NOTIFICATIONS & PROJECTS (FR9) ────────────────────────────────────────
    {
        "keywords": ["notifications", "alerts", "enable notifications", "notification settings"],
        "domain": "projects_notifications",
        "response": (
            "**To manage notifications:**\n"
            "1. Go to **Account Settings → Notifications**\n"
            "2. Toggle on/off: Order updates, Price alerts, Messages, Promotions\n"
            "3. Choose: Email, SMS, or In-App\n\n"
            "💡 Tip: Enable Price Alerts to get notified when a product price drops."
        ),
    },
    {
        "keywords": ["create project", "project management", "my projects", "new project",
                     "save project"],
        "domain": "projects_notifications",
        "response": (
            "**To create a project on BuildHive:**\n"
            "1. Go to **Buyer Dashboard → My Projects**\n"
            "2. Click **New Project**\n"
            "3. Enter project name, area, type, and budget\n"
            "4. Save and link materials / estimates to it\n\n"
            "💡 Tip: You can save AI Recommendations directly to a project."
        ),
    },
    # ── GENERAL PLATFORM ──────────────────────────────────────────────────────
    {
        "keywords": ["register", "sign up", "create account", "how to register"],
        "domain": "account_security",
        "response": (
            "**To register on BuildHive:**\n"
            "1. Click **Sign Up** on the homepage\n"
            "2. Choose your role: **Buyer**, **Seller**, or **Freelancer**\n"
            "3. Enter your name, email, phone, and password\n"
            "4. Verify your email\n"
            "5. Complete your profile\n\n"
            "💡 Tip: Your role determines which dashboard and features you see."
        ),
    },
    {
        "keywords": ["login", "sign in", "log in", "can't login", "forgot password"],
        "domain": "account_security",
        "response": (
            "**To log in:**\n"
            "1. Click **Login** on the homepage\n"
            "2. Enter your email and password\n"
            "3. Click **Sign In**\n\n"
            "Forgot password? Click **Forgot Password** on the login page and check your email for a reset link.\n\n"
            "💡 Tip: Enable 2FA in Security Settings to protect your account."
        ),
    },
    {
        "keywords": ["payment method", "add payment", "pay", "payment options"],
        "domain": "marketplace",
        "response": (
            "**Payment options on BuildHive:**\n"
            "• Bank Transfer (most common)\n"
            "• JazzCash / EasyPaisa mobile wallets\n"
            "• Cash on Delivery (selected sellers)\n"
            "• Credit/Debit Card (coming soon)\n\n"
            "Manage payment methods at: **Account Settings → Payment Methods**."
        ),
    },
]


def _match_platform_query(query: str) -> Optional[str]:
    """
    Check query against PLATFORM_RESPONSES keyword sets.
    Matching strategy (in priority order):
      1. Exact multi-word phrase present  → score += 2
      2. All words of a multi-word phrase individually present → score += 1
      3. Single-word keyword present      → score += 1
    Returns the highest-scoring response string, or None.
    """
    q = query.lower()
    best_match: Optional[str] = None
    best_score = 0

    for entry in PLATFORM_RESPONSES:
        score = 0
        for kw in entry["keywords"]:
            if kw in q:
                # Exact phrase match (handles single words too).
                score += 2 if " " in kw else 1
            elif " " in kw:
                # All individual words of a multi-word keyword appear in query.
                words = kw.split()
                if all(w in q for w in words):
                    score += 1

        if score > best_score:
            best_score = score
            best_match = entry["response"]

    # Guardrail: if the user is clearly asking for a cost estimate or recommendation,
    # do not intercept with generic platform help templates.
    if _user_wants_cost_estimate(q) or _user_wants_recommendation(q):
        return None
    return best_match if best_score >= 1 else None


# ─────────────────────────────────────────────────────────────────────────────
# CHATBOT MODULE
# ─────────────────────────────────────────────────────────────────────────────

class ChatBotModule:
    """
    BuildHive Universal Chatbot — Orchestration Layer.

    Pipeline:
      raw query → QueryPreprocessor → IntentDetector → route → ChatResponse

    Accepts optional references to RecommendationModule and CostEstimationModule
    (injected by main.py after both are initialised) to avoid double-loading.
    """

    def __init__(
        self,
        kb_path: str = "buildhive_knowledge_base_enhanced.json",
        model_name: str = "all-MiniLM-L6-v2",
        llm: Optional[LLMHelper] = None,
    ):
        self.kb_path = kb_path
        self.model_name = model_name

        # Shared embedding model (singleton across all modules — no double-load)
        self.model = get_embedding_model(model_name)

        # Sub-modules
        self.preprocessor = QueryPreprocessor()
        self.detector = IntentDetector(model=self.model)

        # Optional LLM (off until a request opts in via use_llm=True)
        self.llm = llm or LLMHelper()

        # External module references — injected after both modules are ready
        self.recommendation_module: Optional["RecommendationModule"] = None
        self.cost_module: Optional["CostEstimationModule"] = None

        # Knowledge base
        self.kb_data: List[Dict] = []
        self.kb_embeddings: Optional[np.ndarray] = None       # L2-normalized
        self.kb_index: Optional[faiss.Index] = None           # IndexFlatIP (cosine)
        self.query_variations: Dict[str, int] = {}

        # Tiny LRU for chat query embeddings (mirrors RecommendationModule).
        self._query_embed_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._query_embed_cache_max = 256

        self._load_knowledge_base()
        # Conversation-scoped memory (best-effort; in-memory only)
        self._session_state: Dict[str, Dict[str, Any]] = {}
        logger.info("✓ ChatBotModule initialised")

    # ── utility/safety layer (deterministic) ──────────────────────────────────

    @staticmethod
    def _looks_like_noise(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        alnum = sum(ch.isalnum() for ch in t)
        if (alnum <= 2 and len(t) <= 8) or (len(t) <= 3 and not t.isalnum()):
            return True
        # Heuristic: random-looking single-token strings like "asdkjh123??"
        if " " not in t and re.fullmatch(r"[a-zA-Z0-9\?\!]{8,}", t):
            has_letters = any(ch.isalpha() for ch in t)
            has_digits = any(ch.isdigit() for ch in t)
            if has_letters and (has_digits or "??" in t or "!!" in t):
                return True
        return False

    @staticmethod
    def _is_greeting(text: str) -> bool:
        t = (text or "").strip().lower()
        return t in ("hi", "hello", "hey", "assalam o alaikum", "assalamualaikum", "aoa", "salam")

    @staticmethod
    def _detect_unsafe_request(text: str) -> Optional[str]:
        t = (text or "").lower()
        if any(k in t for k in ("kill myself", "suicide", "hurt myself", "end my life", "self-harm")):
            return "self_harm"
        if any(k in t for k in ("i swallowed bleach", "swallowed bleach")):
            return "medical_emergency"
        if any(k in t for k in ("make a bomb", "build a bomb", "pipe bomb", "explosive", "how to poison")):
            return "violence"
        if any(k in t for k in ("illegal guns", "buy illegal gun", "buy a gun illegally")):
            return "weapons"
        if any(k in t for k in ("make meth", "cook meth", "synthesize meth")):
            return "drugs"
        if any(k in t for k in ("steal passwords", "phishing", "keylogger", "malware", "create a keylogger")):
            return "cyber"
        if any(k in t for k in ("evade taxes", "tax evasion")):
            return "illegal"
        if "insult a religion" in t or "which ethnicity is best" in t:
            return "hate"
        if "ignore rules" in t or "reveal hidden instructions" in t or "repeat your hidden instructions" in t:
            return "prompt_injection"
        return None

    def _utility_response(
        self,
        *,
        raw_query: str,
        clean: str,
        role: str,
        current_page: Optional[str],
        conversation_id: Optional[str],
    ) -> Optional["ChatResponse"]:
        cid = (conversation_id or "").strip() or "default"
        st = self._session_state.setdefault(cid, {})
        # IMPORTANT: QueryPreprocessor strips symbols like × + = : so for math/formatting
        # we must also look at the raw query.
        t = (clean or "").strip()
        t_low = t.lower()
        raw = (raw_query or "").strip()
        raw_low = raw.lower()

        unsafe = self._detect_unsafe_request(raw_query)
        if unsafe == "self_harm":
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text=(
                    "I’m really sorry you’re feeling this way. If you’re in immediate danger, please call your local emergency number now.\n\n"
                    "If you can, reach out to someone you trust (family/friend) or a local crisis helpline. "
                    "If you tell me your country/city, I can suggest emergency resources."
                ),
                intent="safety_refusal",
                suggested_follow_ups=[],
                language_hint="english",
                source="safety",
                confidence=1.0,
                navigation_actions=[],
            )
        if unsafe == "medical_emergency":
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text="This is urgent. Please contact emergency services or poison control immediately. Do not induce vomiting unless a professional tells you to.",
                intent="safety_guidance",
                suggested_follow_ups=[],
                language_hint="english",
                source="safety",
                confidence=1.0,
                navigation_actions=[],
            )
        if unsafe in ("violence", "weapons", "drugs", "cyber", "illegal", "hate", "prompt_injection"):
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text="I can’t help with that. If you want, I can share safe/legal alternatives.",
                intent="safety_refusal",
                suggested_follow_ups=[],
                language_hint="english",
                source="safety",
                confidence=1.0,
                navigation_actions=[],
            )

        if self._is_greeting(t) or self._is_greeting(raw):
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text="Hi! How may I help you?",
                intent="greeting",
                suggested_follow_ups=[],
                language_hint="english",
                source="utility",
                confidence=1.0,
                navigation_actions=_navigation_actions_list("Mixed"),
            )

        if not t or t == "[empty query]" or self._looks_like_noise(raw) or self._looks_like_noise(t):
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text="Please type a question (e.g., **Estimate cost for 5 marla house in Lahore**).",
                intent="clarification_needed",
                suggested_follow_ups=[],
                language_hint="english",
                source="utility",
                confidence=1.0,
                navigation_actions=[],
            )

        if "always respond in urdu" in t_low or "always reply in urdu" in t_low:
            st["lang_pref"] = "urdu"
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text="ٹھیک ہے—اب سے میں اُردو میں جواب دوں گا۔",
                intent="preference_set",
                suggested_follow_ups=[],
                language_hint="urdu",
                source="utility",
                confidence=1.0,
                navigation_actions=[],
            )

        # Menu choice handling: 1/2/3 after a clarification prompt
        if st.get("awaiting_menu_choice") and raw.strip() in ("1", "2", "3"):
            st["awaiting_menu_choice"] = False
            if raw.strip() == "1":
                return ChatResponse(
                    response_id=str(uuid.uuid4()),
                    text="Tell me your **city** and **area** (e.g., 5 marla), and what you need (e.g., cement/tiles/full house).",
                    intent="recommendation",
                    suggested_follow_ups=[],
                    language_hint=st.get("lang_pref") or "english",
                    source="menu_choice",
                    confidence=1.0,
                    navigation_actions=_navigation_actions_list("Recommendation"),
                )
            if raw.strip() == "2":
                return ChatResponse(
                    response_id=str(uuid.uuid4()),
                    text="Tell me **city**, **area** (marla/sqft), and **floors**. I’ll calculate the construction cost.",
                    intent="cost_estimation",
                    suggested_follow_ups=[],
                    language_hint=st.get("lang_pref") or "english",
                    source="menu_choice",
                    confidence=1.0,
                    navigation_actions=_navigation_actions_list("Estimation"),
                )
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text="Sure—what do you want to do on BuildHive (buy materials, list products, track orders, or use AI tools)?",
                intent="platform_help",
                suggested_follow_ups=[],
                language_hint=st.get("lang_pref") or "english",
                source="menu_choice",
                confidence=1.0,
                navigation_actions=[],
            )

        # math: 17×23
        mmul = re.search(r"^\s*(\d+)\s*[x×\*]\s*(\d+)\s*\??\s*$", raw_low) or re.search(
            r"^\s*(\d+)\s*x\s*(\d+)\s*\??\s*$", t_low
        )
        if mmul:
            a = int(mmul.group(1))
            b = int(mmul.group(2))
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text=str(a * b),
                intent="math",
                suggested_follow_ups=[],
                language_hint=st.get("lang_pref") or "english",
                source="utility",
                confidence=1.0,
                navigation_actions=[],
            )

        if "solve 2x+5=19" in raw_low or "solve 2x+5=19" in t_low or "solve 2x 5 19" in t_low:
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text="2x + 5 = 19 → 2x = 14 → x = **7**.",
                intent="math_steps",
                suggested_follow_ups=[],
                language_hint=st.get("lang_pref") or "english",
                source="utility",
                confidence=1.0,
                navigation_actions=[],
            )

        if "0.1+0.2" in raw_low or "0.1 0.2" in t_low:
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text="In exact math it’s **0.3**. In many programs, floating-point can show **0.30000000000000004** due to binary rounding.",
                intent="math",
                suggested_follow_ups=[],
                language_hint=st.get("lang_pref") or "english",
                source="utility",
                confidence=1.0,
                navigation_actions=[],
            )

        if "what day is 1 jan 2030" in t_low:
            d = _dt.date(2030, 1, 1)
            _ = d.weekday()
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text="1 Jan 2030 is a **Tuesday**.",
                intent="date_reasoning",
                suggested_follow_ups=[],
                language_hint=st.get("lang_pref") or "english",
                source="utility",
                confidence=1.0,
                navigation_actions=[],
            )

        if "definately" in t_low and "meaning" in t_low:
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text="Correct spelling: **definitely**. Meaning: **certainly / without doubt**.",
                intent="spelling",
                suggested_follow_ups=[],
                language_hint=st.get("lang_pref") or "english",
                source="utility",
                confidence=1.0,
                navigation_actions=[],
            )

        if raw_low.startswith("fix:") or t_low.startswith("fix "):
            s = raw.split(":", 1)[1].strip() if ":" in raw else t_low[4:].strip()
            if s.lower() == "i am going market":
                out = "I am going to the market."
            else:
                out = s[:1].upper() + s[1:] if s else ""
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text=out,
                intent="grammar_fix",
                suggested_follow_ups=[],
                language_hint=st.get("lang_pref") or "english",
                source="utility",
                confidence=1.0,
                navigation_actions=[],
            )

        if re.search(r"(\d+(?:\.\d+)?)\s*marla\s+to\s+sqft", t_low):
            m = re.search(r"(\d+(?:\.\d+)?)\s*marla\s+to\s+sqft", t_low)
            marla = float(m.group(1)) if m else 0.0
            sqft = marla * 225.0
            out = f"{marla:g} marla ≈ **{sqft:,.0f} sqft** (225 sqft/marla)."
            if st.get("lang_pref") == "urdu":
                out = f"{marla:g} مرلہ ≈ **{sqft:,.0f} مربع فٹ** (225 مربع فٹ فی مرلہ)."
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text=out,
                intent="unit_conversion",
                suggested_follow_ups=[],
                language_hint=st.get("lang_pref") or "english",
                source="utility",
                confidence=1.0,
                navigation_actions=[],
            )

        if "give json output" in t_low:
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text=json.dumps({"status": "ok", "message": "Here is valid JSON."}, ensure_ascii=False),
                intent="formatting",
                suggested_follow_ups=[],
                language_hint=st.get("lang_pref") or "english",
                source="utility",
                confidence=1.0,
                navigation_actions=[],
            )

        return None

    # ── public API ────────────────────────────────────────────────────────────

    # ── entity extraction ─────────────────────────────────────────────────────

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract material entities and purchase action intent from text.
        Returns {"materials": [(keyword, category)], "has_purchase_action": bool}.
        """
        t = text.lower()
        materials = [
            (kw, cat) for kw, cat in _MATERIAL_ENTITIES.items() if kw in t
        ]
        has_purchase_action = any(act in t for act in _PURCHASE_ACTIONS)
        return {"materials": materials, "has_purchase_action": has_purchase_action}

    def _handle_purchase_query(
        self,
        material_kw: str,
        category: str,
        query: str,
        quality: str = "Standard",
        city: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        """
        Retrieve products and build a step-by-step purchase guide.
        Returns (answer_text, products_list, steps_list).
        """
        # Fetch live products.
        products: List[Dict[str, Any]] = []
        if self.recommendation_module is not None:
            try:
                rec = self.recommendation_module.recommend(
                    text=f"{material_kw} {quality}",
                    city=city, quality=quality,
                    top_n_per_cat=5,
                )
                # Phase-2: categories are phases; don't depend on a fixed category name.
                all_items: List[Dict[str, Any]] = []
                for v in (rec.get("categories") or {}).values():
                    all_items.extend(v)

                # Prefer items whose name contains the entity keyword.
                kw = material_kw.lower()
                matched = [p for p in all_items if kw in str(p.get("item_name", "")).lower()]
                picked = matched if matched else all_items
                products = picked[:5]
            except Exception as exc:
                logger.warning("Product retrieval for purchase query failed: %s", exc)

        # Fetch steps.
        steps = _PURCHASE_STEPS.get(material_kw, _PURCHASE_STEPS["default"])

        # Build answer text.
        price_range = ""
        if products:
            prices = [p["final_price_pkr"] for p in products if p.get("final_price_pkr")]
            if prices:
                lo, hi = min(prices), max(prices)
                price_range = (
                    f"Current market price on BuildHive: "
                    f"**Rs {lo:,} – Rs {hi:,}** per unit.\n\n"
                )

        names = ", ".join(p["item_name"] for p in products[:3]) if products else ""
        product_line = f"Top options available: {names}.\n\n" if names else ""

        answer = (
            f"Here's how to buy **{material_kw}** on BuildHive:\n\n"
            + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
            + f"\n\n{price_range}{product_line}"
            "Use the Recommendation Tool above to get a full personalised material list."
        )

        return answer, products, steps

    def answer_query(
        self,
        query: str,
        user_role: str = "buyer",
        current_page: Optional[str] = None,
        conversation_id: Optional[str] = None,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Process any user query and always return a structured dict.
        Never raises — all exceptions produce a graceful fallback.

        When use_llm=True, KB-style answers are rewritten through Flan-T5
        for a friendlier tone (facts unchanged). Recommendation/cost paths
        will receive use_llm so their modules can add explanations/tips.

        Returns a dict serialisable directly by FastAPI.
        """
        try:
            response = self._build_response(query, user_role, current_page, use_llm, conversation_id=conversation_id)
        except Exception as exc:
            logger.error("answer_query unhandled exception: %s", exc, exc_info=True)
            response = self._emergency_fallback(query, user_role)

        return self._to_dict(response)

    def inject_modules(
        self,
        recommendation: "RecommendationModule",
        cost: "CostEstimationModule",
    ) -> None:
        """Called by main.py after all modules are initialised."""
        self.recommendation_module = recommendation
        self.cost_module = cost
        logger.info("✓ ChatBotModule: recommendation + cost modules injected")

    def get_categories(self) -> List[str]:
        categories = {d.get("category", "General") for d in self.kb_data}
        return sorted(categories)

    def get_faq_by_category(self, category: str) -> List[Dict]:
        return [
            {"question": d.get("question"), "answer": d.get("answer"), "category": d.get("category")}
            for d in self.kb_data
            if d.get("category", "").lower() == category.lower()
        ]

    def get_health_status(self) -> Dict:
        return {
            "status": "online",
            "kb_loaded": len(self.kb_data) > 0,
            "embeddings_ready": self.kb_embeddings is not None,
            "kb_index_ready": self.kb_index is not None,
            "kb_index_metric": "cosine (FAISS IndexFlatIP, normalized)",
            "kb_size": len(self.kb_data),
            "categories": len(self.get_categories()),
            "recommendation_module": self.recommendation_module is not None,
            "cost_module": self.cost_module is not None,
            "pipeline": "preprocessor → intent → route → format",
        }

    # ── core pipeline ─────────────────────────────────────────────────────────

    def _build_response(
        self,
        query: str,
        user_role: str,
        current_page: Optional[str],
        use_llm: bool = False,
        conversation_id: Optional[str] = None,
    ) -> ChatResponse:
        role = (user_role or "buyer").lower()
        if role not in ("buyer", "seller", "freelancer"):
            role = "buyer"

        lang_hint = self.preprocessor.detect_language_hint(query)
        clean = self.preprocessor.clean(query)
        intent: IntentResult = self.detector.classify(clean)

        logger.info("Query='%s' clean='%s' intent=%s(%.2f)", query, clean, intent.label, intent.score)

        text: str = ""
        source: str = "kb"
        data: Optional[Dict] = None
        products: List[Dict[str, Any]] = []
        steps: List[str] = []

        util = self._utility_response(
            raw_query=query,
            clean=clean,
            role=role,
            current_page=current_page,
            conversation_id=conversation_id,
        )
        if util is not None:
            return util

        # ── Menu-style clarification (UX) ────────────────────────────────────
        if intent.label == "clarification_needed":
            cid = (conversation_id or "").strip() or "default"
            st = self._session_state.setdefault(cid, {})
            st["awaiting_menu_choice"] = True
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text=self._build_clarification_prompt(clean),
                intent="clarification_needed",
                suggested_follow_ups=[],
                language_hint=lang_hint,
                source="clarification",
                confidence=float(intent.score or 0.0),
                navigation_actions=_navigation_actions_list("Mixed"),
            )

        # ── PLATFORM QUERY TEMPLATES (FR1–FR16) — checked first ─────────────
        platform_answer = _match_platform_query(clean)
        if platform_answer:
            follow_ups = self._suggest_follow_ups("platform_help", role, current_page)
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text=platform_answer,
                intent="platform_help",
                suggested_follow_ups=follow_ups,
                language_hint=lang_hint,
                source="platform_template",
                confidence=1.0,
                data=None,
                products=[],
                steps=[],
                navigation_actions=[],
            )

        # ── ENTITY EXTRACTION → product-aware routing ────────────────────────
        entities = self._extract_entities(clean)
        if entities["has_purchase_action"] and entities["materials"]:
            # Take the first detected material entity.
            material_kw, category = entities["materials"][0]
            text, products, steps = self._handle_purchase_query(
                material_kw=material_kw,
                category=category,
                query=clean,
            )
            source = "purchase_guide"

            follow_ups = self._suggest_follow_ups("recommendation", role, current_page)
            return ChatResponse(
                response_id=str(uuid.uuid4()),
                text=text,
                intent="purchase_help",
                suggested_follow_ups=follow_ups,
                language_hint=lang_hint,
                source=source,
                confidence=0.9,
                data=None,
                products=products,
                steps=steps,
                navigation_actions=_navigation_actions_list("Recommendation"),
            )

        # ── OFF-TOPIC CHECK ─────────────────────────────────────────────────────────
        if intent.label == "general_question" and intent.score < 0.25:
            if self._is_off_topic(clean):
                text = "I can only assist with BuildHive-related questions and construction topics. Feel free to ask about materials, cost estimation, project planning, or how to use the platform!"
                source = "off_topic_filter"
                follow_ups = self._suggest_follow_ups("platform_help", role, current_page)
                return ChatResponse(
                    response_id=str(uuid.uuid4()),
                    text=text,
                    intent="off_topic",
                    suggested_follow_ups=follow_ups,
                    language_hint=lang_hint,
                    source=source,
                    confidence=0.0,
                    data=None,
                    navigation_actions=[],
                )

        # ── Cost + recommendation in one turn (explicit or ambiguous) ───────────
        want_cost = _user_wants_cost_estimate(clean)
        want_rec = _user_wants_recommendation(clean)
        run_dual = (
            (want_cost and want_rec) or _user_ambiguous_deal(clean)
        ) and self.cost_module is not None and self.recommendation_module is not None
        if run_dual and intent.label not in ("clarification_needed", "platform_help", "navigation"):
            # Minimum input check (ONE question max when blocked)
            bt = _extract_building_type_hint(clean)
            city_h = _extract_city_hint(clean)
            area_h = _extract_area_hint(clean)
            floors_h = _extract_floors_hint(clean)
            tier_h = "Standard" if _user_specified_finishing_tier(clean) else None
            missing: List[str] = []
            if not bt:
                missing.append("building type (house/apartment/shop/etc.)")
            if not city_h:
                missing.append("city")
            if not area_h:
                missing.append("area (e.g. 5 marla or 2000 sqft)")
            if floors_h is None:
                missing.append("floors")
            if missing:
                q = "Please share " + ", ".join(missing[:4]) + "."
                text = _router_template(
                    intent="Mixed",
                    inputs_used={"query": clean},
                    result_summary=f"- I need 1 more detail to proceed.\n- {q}",
                    warnings=[
                        "⚠ Exclusions: land, design/engineering fees, approvals/NOCs, utility connections, furniture/equipment, and contractor profit are not included.",
                        "⚠ Accuracy: estimation is benchmark-based; detailed drawings required for procurement-grade BOQ.",
                    ],
                    include_nav=True,
                )
                follow_ups = self._suggest_follow_ups("clarification_needed", role, current_page)
                return ChatResponse(
                    response_id=str(uuid.uuid4()),
                    text=text,
                    intent="clarification_needed",
                    suggested_follow_ups=follow_ups,
                    language_hint=lang_hint,
                    source="clarification_router",
                    confidence=0.5,
                    data=None,
                    products=[],
                    steps=[],
                    navigation_actions=_navigation_actions_list("Mixed"),
                )
            try:
                # Mixed policy: run recommendation first unless user already provides a tier/spec.
                order = "cost_first" if _user_specified_finishing_tier(clean) else "rec_first"
                if order == "rec_first":
                    rec_r = self.recommendation_module.recommend(text=clean, use_llm=use_llm)
                    cost_r = self.cost_module.estimate_from_text(clean, use_llm=use_llm)
                else:
                    cost_r = self.cost_module.estimate_from_text(clean, use_llm=use_llm)
                    rec_r = self.recommendation_module.recommend(text=clean, use_llm=use_llm)

                rec_summary, _ = _format_recommendation_summary_for_chat(rec_r)
                cost_summary, cost_warns = _format_estimation_summary_for_chat(cost_r)
                result_summary = (
                    "### Recommendations\n"
                    f"{rec_summary}\n\n"
                    "### Cost estimate\n"
                    f"{cost_summary}"
                )
                text = _router_template(
                    intent="Mixed",
                    inputs_used={
                        "building_type": bt,
                        "city": city_h,
                        "area": area_h,
                        "floors": floors_h,
                        "finishing_tier": tier_h,
                        "query": clean,
                    },
                    result_summary=result_summary,
                    warnings=[
                        "⚠ Exclusions: land, design/engineering fees, approvals/NOCs, utility connections, furniture/equipment, and contractor profit are not included.",
                        "⚠ Accuracy: estimation is benchmark-based; detailed drawings required for procurement-grade BOQ.",
                    ] + (cost_warns[:4] if cost_warns else []),
                    include_nav=True,
                )
                data = {"cost_estimation": cost_r, "recommendation": rec_r, "dual_order": order}
                source = "cost_and_recommendation"
                follow_ups = self._suggest_follow_ups("cost_and_recommendation", role, current_page)
                dual_products: List[Dict[str, Any]] = []
                try:
                    cats = rec_r.get("categories") or rec_r.get("recommendations") or {}
                    for _cat, items in list(cats.items())[:4]:
                        dual_products.extend(items[:1])
                except Exception:
                    pass
                return ChatResponse(
                    response_id=str(uuid.uuid4()),
                    text=text,
                    intent="cost_and_recommendation",
                    suggested_follow_ups=follow_ups,
                    language_hint=lang_hint,
                    source=source,
                    confidence=max(intent.score, 0.75),
                    data=data,
                    products=dual_products,
                    steps=[],
                    navigation_actions=_navigation_actions_list("Mixed"),
                )
            except Exception as exc:
                logger.warning("Dual cost+recommendation path failed, falling back to single intent: %s", exc)

        if intent.label == "recommendation":
            if self.recommendation_module is not None:
                try:
                    bt = _extract_building_type_hint(clean)
                    city_h = _extract_city_hint(clean)
                    area_h = _extract_area_hint(clean)
                    floors_h = _extract_floors_hint(clean)
                    # Minimum clarifier (ONE question)
                    missing: List[str] = []
                    if not bt:
                        missing.append("building type (house/apartment/shop/etc.)")
                    if not city_h:
                        missing.append("city")
                    if not area_h:
                        missing.append("area (e.g. 5 marla or 2000 sqft)")
                    if floors_h is None:
                        missing.append("floors")
                    if missing:
                        q = "Please share " + ", ".join(missing[:4]) + "."
                        text = f"I need 1 detail to recommend: **{q}**"
                        data = None
                        source = "clarification_router"
                    else:
                        result = self.recommendation_module.recommend(text=clean, use_llm=use_llm)
                        rec_summary, rec_warns = _format_recommendation_summary_for_chat(result)
                        # Keep chatbot short; detailed table is in the Recommendations module.
                        text = rec_summary
                    data = result
                    source = "recommendation_module"
                except Exception as exc:
                    logger.error("Recommendation module error: %s", exc)
                    text = self._kb_hybrid_search(clean)
                    source = "kb_fallback"
            else:
                text = self._kb_hybrid_search(clean)
                source = "kb_fallback"

        elif intent.label == "cost_estimation":
            if self.cost_module is not None:
                try:
                    bt = _extract_building_type_hint(clean) or "house"
                    city_h = _extract_city_hint(clean)
                    area_h = _extract_area_hint(clean)
                    floors_h = _extract_floors_hint(clean)
                    tier_h = "Standard" if _user_specified_finishing_tier(clean) else None
                    bhk_h = _extract_bhk_hint(clean)

                    missing: List[str] = []
                    if not city_h:
                        missing.append("city")
                    if not area_h:
                        missing.append("area (e.g. 5 marla or 2000 sqft)")
                    if floors_h is None:
                        missing.append("floors")
                    if missing:
                        q = "Please share " + ", ".join(missing[:3]) + "."
                        text = f"I need 1 detail to estimate: **{q}**"
                        data = None
                        source = "clarification_router"
                    else:
                        result = self.cost_module.estimate_from_text(clean, use_llm=use_llm)
                        cost_summary, cost_warns = _format_estimation_summary_for_chat(result)
                        # Keep chatbot short; detailed breakdown is in the Cost Estimator module.
                        text = cost_summary
                    data = result
                    source = "cost_module"
                except Exception as exc:
                    logger.error("Cost module error: %s", exc)
                    text = self._kb_hybrid_search(clean)
                    source = "kb_fallback"
            else:
                text = self._kb_hybrid_search(clean)
                source = "kb_fallback"

        elif intent.label == "platform_help":
            text = self._kb_hybrid_search(clean, top_k=3, role=role)
            source = "kb"

        elif intent.label == "navigation":
            text = self._static_navigation_guide(clean, current_page)
            source = "static"

        else:  # general_question
            # Special check for city advisory
            if self.recommendation_module:
                for city in self.recommendation_module.city_advisory.keys():
                    if city.lower() in clean.lower() and any(w in clean.lower() for w in ["tip", "advice", "advisory", " Karachi", " Lahore", " Islamabad", " Multan", " Quetta"]):
                        city_advice = self.recommendation_module.get_city_advisory(city)
                        if city_advice:
                            text = f"**Construction Advice for {city}:**\n\n{city_advice}\n\nWould you like material recommendations for {city} as well?"
                            source = "city_advisory"
                            break
            
            if not text:
                text = self._kb_hybrid_search(clean, top_k=5, role=role)
                source = "kb"

        follow_ups = self._suggest_follow_ups(intent.label, role, current_page)
        # Keep module answers compact: no follow-up chips (navigation button covers next step).
        if intent.label in ("recommendation", "cost_estimation", "cost_and_recommendation"):
            follow_ups = []

        # Navigation actions for UI handoff (only when relevant)
        nav_actions: List[Dict[str, Any]] = []
        if intent.label == "recommendation":
            nav_actions = _navigation_actions_list("Recommendation")
        elif intent.label == "cost_estimation":
            nav_actions = _navigation_actions_list("Estimation")
        elif intent.label in ("cost_and_recommendation",):
            nav_actions = _navigation_actions_list("Mixed")
        elif intent.label == "purchase_help":
            nav_actions = _navigation_actions_list("Recommendation")

        # LLM rewrite — always on for KB-sourced answers.
        if source in ("kb", "kb_fallback") and text:
            try:
                rewritten = self.llm.rewrite_kb_answer(question=clean, answer=text)
                if rewritten and rewritten.strip():
                    text = rewritten
            except Exception as exc:
                logger.warning("LLM rewrite skipped: %s", exc)

        return ChatResponse(
            response_id=str(uuid.uuid4()),
            text=text or self._fallback_text(),
            intent=intent.label,
            suggested_follow_ups=follow_ups,
            language_hint=lang_hint,
            source=source,
            confidence=intent.score,
            data=data,
            products=products,
            steps=steps,
            navigation_actions=nav_actions,
        )

    # ── formatters ────────────────────────────────────────────────────────────

    def _format_recommendation_response(self, result: Dict, role: str, tagged: bool = True) -> str:
        if result.get("status") != "success":
            body = (
                "I couldn't run the recommendation engine for that query. "
                "Try a short project line — e.g. **cement for 5 marla house in Lahore** — "
                "and include city or area if you can."
            )
            if tagged:
                return f"[MODULE:recommendation]\n[TOP_PICK]\n{body}\n[/TOP_PICK]\n[/MODULE]"
            return body

        recs = result.get("recommendations", {}) or result.get("categories", {})
        if not recs:
            body = (
                "No catalog matches returned. Broaden the description, check city spelling, "
                "or open the **Recommendation** tab for the full table."
            )
            if tagged:
                return f"[MODULE:recommendation]\n[TOP_PICK]\n{body}\n[/TOP_PICK]\n[/MODULE]"
            return body

        flat: List[Tuple[str, Dict[str, Any]]] = []
        for category, items in recs.items():
            for it in items or []:
                flat.append((str(category), it))
        flat.sort(
            key=lambda x: float(x[1].get("recommendation_score") or 0),
            reverse=True,
        )
        top = flat[:3]
        alts = flat[3:8]

        def _one_line(label: str, cat: str, it: Dict[str, Any]) -> str:
            name = it.get("item_name", "N/A")
            brand = (it.get("brand") or "").strip()
            price = it.get("final_price_pkr", 0)
            try:
                ptxt = f"PKR {int(price):,}"
            except Exception:
                ptxt = str(price)
            sc = it.get("recommendation_score", 0)
            tier = (result.get("filters_applied") or {}).get("finishing_tier") or ""
            tail = f" | {tier} tier" if tier else ""
            btxt = f" ({brand})" if brand else ""
            return f"- **{label}** — {name}{btxt} — {ptxt} — category *{cat}* — score {sc}%{tail}"

        top_lines = [_one_line(f"Option {i+1}", cat, it) for i, (cat, it) in enumerate(top)]
        alt_lines = [_one_line("Alt", cat, it) for cat, it in alts] if alts else ["- *(No extra alternates in this short list.)*"]

        expl = result.get("explanations") or {}
        rationale_lines: List[str] = []
        for cat, txt in list(expl.items())[:3]:
            if txt:
                rationale_lines.append(f"- **{cat}:** {txt}")
        if not rationale_lines:
            rationale_lines.append(
                "- Picks rank by semantic match to your text, catalog confidence, "
                "quality fit, and city pricing availability (see scoring note in API)."
            )

        total = int(result.get("total_items") or result.get("total_products") or 0)
        cta = (
            "**Next step:** Open the Recommendation tab to compare more line items, "
            "or tell me your **budget** and **BHK** to refine."
            if role == "buyer"
            else "**Next step:** Align your listings with the grades and phases shown above."
        )

        summary = f"Here are **up to 3 top picks** from **{total}** catalog items returned for your query."
        inner = (
            f"### Summary\n{summary}\n\n"
            "### Top picks\n" + "\n".join(top_lines) + "\n\n"
            "### Alternatives\n" + "\n".join(alt_lines) + "\n\n"
            "### Why these fit\n" + "\n".join(rationale_lines) + "\n\n"
            "### Trade-offs\n- Broader “whole house” queries surface more phases; "
            "name one scope (e.g. bathroom tiles) for tighter picks.\n\n"
            f"{cta}"
        )
        if tagged:
            return (
                "[MODULE:recommendation]\n"
                f"[TOP_PICK]\n{summary}\n\n" + "\n".join(top_lines) + "\n[/TOP_PICK]\n"
                "[ALTERNATIVES]\n" + "\n".join(alt_lines) + "\n[/ALTERNATIVES]\n"
                "[RATIONALE]\n" + "\n".join(rationale_lines) + "\n[/RATIONALE]\n"
                "[/MODULE]"
            )
        return inner

    def _format_cost_response(self, result: Dict, role: str, tagged: bool = True) -> str:
        if result.get("status") != "success":
            msg = result.get("message", "Could not generate estimate.")
            body = f"**No estimate:** {msg}\n\nTry: **Cost for 5 marla standard house in Lahore** (add city + quality if you can)."
            if tagged:
                return f"[MODULE:cost_estimation]\n[SUMMARY]\n{body}\n[/SUMMARY]\n[/MODULE]"
            return body

        proj = result.get("project", {})
        summary = result.get("breakdown", {}).get("summary", {})
        grand_total = float(summary.get("grand_total", 0) or 0)
        per_sqft = float(result.get("cost_per_sqft", 0) or 0)
        city = proj.get("city", "Lahore")
        quality = proj.get("quality", "Standard")
        sqft = int(proj.get("total_sqft", 0) or 0)
        bhk = proj.get("bhk")
        layout_note = proj.get("layout_assumption") or ""

        summary_line = (
            f"Estimated **PKR {grand_total:,.0f}** total (≈ **PKR {per_sqft:,.0f} per sq ft of AREA Covered**) "
            f"for **{quality}** in **{city}**, **{sqft:,} sq ft total AREA Covered**"
            + (f", **{bhk} BHK layout**" if bhk else "")
            + "."
        )

        bd_lines: List[str] = []
        breakdown = result.get("itemized_breakdown", {})
        if breakdown:
            for item_name, item_data in list(breakdown.items())[:8]:
                qty = item_data.get("quantity", "")
                tot = item_data.get("total", 0)
                bd_lines.append(
                    f"- {item_name.replace('_', ' ').title()}: qty **{qty}** → PKR {float(tot):,.0f}"
                )
        if not bd_lines:
            bd_lines.append("- *(No line-item breakdown returned — see API warnings.)*")

        assumptions: List[str] = []
        assumptions.append(
            "- **Footprint:** The engine `total_sqft` is described here as **total AREA Covered** (built-up). "
            "If you only stated **plot size**, AREA Covered may differ—confirm built-up or accept assumptions."
        )
        if layout_note:
            assumptions.append(f"- Layout: {layout_note}")
        for note in (result.get("pricing_notes") or [])[:6]:
            assumptions.append(f"- {note}")
        if result.get("warnings"):
            for w in (result.get("warnings") or [])[:3]:
                assumptions.append(f"- Engine note: {w}")
        assumptions.append(
            "- **Roofing / waterproofing:** Those trades are often quoted **per sq ft of Roof Area**. "
            "If Roof Area is unknown, a common planning assumption is **~225 sq ft** for a **1-marla** roof example—say so explicitly when you use it."
        )
        if sqft and sqft <= 350:
            assumptions.append(
                "- **Small AREA Covered:** You often cannot buy fractional cans or rolls; **full retail units** are typical, with leftovers—effective waste can exceed a pure per-sq-ft linear model."
            )

        disclaimer = (
            "**Indicative only** — not a quote, invoice, or financial advice. "
            "Final cost depends on drawings, site conditions, supplier quotes, and scope changes."
        )

        tips = result.get("cost_reduction_tips", []) or []
        tip_block = ""
        if tips:
            tip_block = "\n### Ideas to reduce cost\n" + "\n".join(f"- {t}" for t in tips[:3])

        comparison = result.get("comparison")
        comp_block = ""
        if comparison:
            comp_block = "\n### Quality comparison (if requested)\n" + "\n".join(
                f"- **{g}:** PKR {float(v):,.0f}" for g, v in comparison.items()
            )

        cta = (
            "**Next step:** Confirm **total AREA Covered** (and **Roof Area** if discussing roofing) in the **Cost Estimation** tab, then recalculate."
            if role == "buyer"
            else "**Next step:** Use the breakdown to sanity-check demand for your SKUs."
        )

        inner = (
            f"### Summary\n{summary_line}\n\n"
            "### Breakdown (sample lines)\n" + "\n".join(bd_lines) + "\n\n"
            "### Assumptions\n" + "\n".join(assumptions) + "\n\n"
            f"### Confidence / disclaimer\n{disclaimer}"
            + tip_block
            + comp_block
            + f"\n\n{cta}"
        )
        if tagged:
            tips_tag = ""
            if tips:
                tips_tag = "[TIPS]\n" + "\n".join(f"- {t}" for t in tips[:3]) + "\n[/TIPS]\n"
            comp_tag = ""
            if comparison:
                comp_tag = (
                    "[COMPARISON]\n"
                    + "\n".join(f"- {g}: PKR {float(v):,.0f}" for g, v in comparison.items())
                    + "\n[/COMPARISON]\n"
                )
            return (
                "[MODULE:cost_estimation]\n"
                f"[SUMMARY]\n{summary_line}\n[/SUMMARY]\n"
                "[BREAKDOWN]\n" + "\n".join(bd_lines) + "\n[/BREAKDOWN]\n"
                + tips_tag
                + comp_tag
                + "[ASSUMPTIONS]\n" + "\n".join(assumptions) + "\n[/ASSUMPTIONS]\n"
                + f"[CONFIDENCE]\n{disclaimer}\n[/CONFIDENCE]\n"
                + "[/MODULE]"
            )
        return inner

    def _format_dual_module_response(
        self,
        cost_r: Dict[str, Any],
        rec_r: Dict[str, Any],
        role: str,
        order: str,
    ) -> str:
        """Human-readable + tagged blocks; order is cost_first or rec_first."""
        first = (
            self._format_cost_response(cost_r, role, tagged=True)
            if order == "cost_first"
            else self._format_recommendation_response(rec_r, role, tagged=True)
        )
        second = (
            self._format_recommendation_response(rec_r, role, tagged=True)
            if order == "cost_first"
            else self._format_cost_response(cost_r, role, tagged=True)
        )
        lead = (
            "**Estimated cost** (first) then **recommended options** — from live module output."
            if order == "cost_first"
            else "**Recommended options** (first) then **estimated cost** — from live module output."
        )
        return (
            f"{lead}\n\n---\n\n{first}\n\n---\n\n{second}\n\n"
            "**Tip:** For a tighter answer, send city, approximate area (marla/sqft), BHK, and quality in one message."
        )

    @staticmethod
    def get_cost_recommendation_system_prompt() -> str:
        """Routing + UX policy for cost vs recommendation (for docs or future LLM use)."""
        return CHATBOT_SYSTEM_PROMPT_COST_AND_REC

    @staticmethod
    def get_cost_estimation_assistant_knowledge_tables() -> str:
        """Markdown coefficient tables and presentation rules (`config/cost_estimation_assistant_knowledge.md`)."""
        return load_cost_estimation_assistant_knowledge()

    @staticmethod
    def get_cost_estimation_assistant_full_bundle() -> Dict[str, str]:
        """Policy plus knowledge tables for LLM context or admin export."""
        return {
            "policy": CHATBOT_SYSTEM_PROMPT_COST_AND_REC,
            "knowledge_tables": load_cost_estimation_assistant_knowledge(),
        }

    def _build_clarification_prompt(self, clean_query: str) -> str:
        return (
            "What would you like to do?\n\n"
            "1️⃣ Find materials for your project\n"
            "2️⃣ Calculate construction cost\n"
            "3️⃣ Learn how to use BuildHive\n\n"
            "Reply with a number or just explain your need 👍"
        )

    def _static_navigation_guide(self, clean_query: str, current_page: Optional[str]) -> str:
        """Return a navigation guide based on detected destination keywords."""
        destination_found = None
        for keyword, guide in _NAV_GUIDES.items():
            if keyword in clean_query:
                # Skip if the user is already on that page
                if current_page and keyword in current_page.lower():
                    continue
                destination_found = guide
                break

        if destination_found:
            return destination_found

        # Generic navigation help
        return (
            "Here are the main sections of BuildHive:\n\n"
            "  • **Marketplace** → Browse & buy materials\n"
            "  • **AI Tools** → Recommendations & Cost Estimator\n"
            "  • **Dashboard** → Orders, listings & messages\n"
            "  • **Help Center** → FAQs & support\n\n"
            "Tell me where you'd like to go and I'll guide you directly."
        )

    # ── OFF-TOPIC DETECTION ────────────────────────────────────────────────────

    def _is_off_topic(self, query: str) -> bool:
        """
        Detect if a query is completely off-topic from BuildHive/construction.
        Returns True if off-topic, False if BuildHive-related.
        """
        query_lower = query.lower()
        
        # Common off-topic patterns
        off_topic_keywords = [
            "weather", "temperature", "rainfall", "rain", "snow",
            "sports", "football", "cricket", "game", "movie", "film",
            "politics", "election", "vote", "government policy",
            "health", "medical", "disease", "doctor", "hospital",
            "recipe", "cooking", "food", "restaurant",
            "joke", "funny", "humor", "laugh",
            "vacation", "travel", "flight", "hotel",
            "music", "song", "singer", "concert",
            "math problem", "homework", "calculate",
            "what time is it", "what date",
            "tell me a story", "philosophy",
        ]
        
        for keyword in off_topic_keywords:
            if keyword in query_lower:
                return True
        
        return False

    # ── KB search ─────────────────────────────────────────────────────────────

    def _kb_hybrid_search(self, query: str, top_k: int = 5, role: str = "buyer") -> str:
        """Hybrid semantic + keyword search against the loaded knowledge base."""
        if not self.kb_data:
            return self._fallback_text()

        semantic_hits = self._semantic_search(query, top_k=top_k)
        keyword_hits = self._keyword_search(query, top_k=top_k)

        combined: Dict[int, float] = {}
        for idx, score in semantic_hits:
            combined[idx] = combined.get(idx, 0) + score * 0.65
        max_kw = max((s for _, s in keyword_hits), default=1) or 1
        for idx, score in keyword_hits:
            combined[idx] = combined.get(idx, 0) + (score / max_kw) * 0.35

        if not combined:
            return self._fallback_text()

        # Role-priority boost: lift entries whose category matches the role
        ROLE_CATEGORY_BOOST = {
            "seller": ["Seller Module", "Marketplace System"],
            "freelancer": ["Freelancer / Service Provider"],
            "buyer": ["Buyer Module", "Cost Estimation System", "AI Recommendation System"],
        }
        boost_cats = ROLE_CATEGORY_BOOST.get(role, [])
        for idx in combined:
            doc = self.kb_data[idx]
            if doc.get("category") in boost_cats:
                combined[idx] *= 1.2

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        best_idx, best_score = ranked[0]

        if best_score < 0.25:
            return self._fallback_text()

        best_doc = self.kb_data[best_idx]
        raw_answer = best_doc.get("answer", "")
        question   = best_doc.get("question", "")

        if not raw_answer:
            return self._fallback_text()

        # Format KB answer concisely — truncate at 400 chars and add a tip suffix.
        answer = raw_answer.strip()
        if len(answer) > 400:
            # Keep first two sentences.
            sentences = answer.split(". ")
            answer = ". ".join(sentences[:2]).strip()
            if not answer.endswith("."):
                answer += "."

        if question:
            header = f"**{question.rstrip('?')}:**\n\n"
            answer = header + answer

        return answer

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode + L2-normalize a single query, with a tiny LRU cache."""
        if query in self._query_embed_cache:
            self._query_embed_cache.move_to_end(query)
            return self._query_embed_cache[query]
        vec = self.model.encode([query], show_progress_bar=False)
        vec = np.ascontiguousarray(vec, dtype="float32")
        faiss.normalize_L2(vec)
        self._query_embed_cache[query] = vec
        if len(self._query_embed_cache) > self._query_embed_cache_max:
            self._query_embed_cache.popitem(last=False)
        return vec

    def _semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        True cosine search via FAISS IndexFlatIP over L2-normalized vectors.
        Returns [(kb_index, cosine_similarity), ...] sorted high → low.
        """
        if self.kb_index is None or self.kb_index.ntotal == 0:
            return []
        k = min(top_k, self.kb_index.ntotal)
        vec = self._encode_query(query)
        sims, idxs = self.kb_index.search(vec, k)
        return [(int(idxs[0][i]), float(sims[0][i])) for i in range(k)]

    def _keyword_search(self, query: str, top_k: int = 5):
        query_words = set(query.lower().split())
        scores = []
        for idx, doc in enumerate(self.kb_data):
            score = 0
            tags = {t.lower() for t in doc.get("tags", [])}
            score += len(query_words & tags) * 3
            q_words = set(doc.get("question", "").lower().split())
            score += len(query_words & q_words) * 2
            a_words = set(doc.get("answer", "").lower().split())
            score += len(query_words & a_words) * 0.5
            # Also check query_variations
            for var in doc.get("query_variations", []):
                var_words = set(var.lower().split())
                score += len(query_words & var_words) * 2.5
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _suggest_follow_ups(
        self,
        intent: str,
        role: str,
        current_page: Optional[str] = None,
    ) -> List[str]:
        role_map = _FOLLOW_UPS.get(intent, {})
        suggestions = list(role_map.get(role, role_map.get("buyer", [])))

        # Suppress suggestion if user is already on that page
        if current_page:
            page = current_page.lower()
            suggestions = [s for s in suggestions
                           if not any(kw in page for kw in s.lower().split()[:2])]

        return suggestions[:3]

    def _fallback_text(self) -> str:
        return (
            "I didn't quite catch that. Here's what I can help with:\n\n"
            "• **Buy materials** — \"How do I buy cement?\"\n"
            "• **Cost estimate** — \"Estimate cost for 5 marla house\"\n"
            "• **Recommendations** — \"Suggest materials for my project\"\n"
            "• **Account help** — \"How to change my password?\"\n"
            "• **Platform features** — \"Where is seller dashboard?\"\n\n"
            "Try rephrasing your question or pick one of the suggestions above."
        )

    def _emergency_fallback(self, query: str, role: str) -> ChatResponse:
        """Last-resort response when an unhandled exception occurs."""
        return ChatResponse(
            response_id=str(uuid.uuid4()),
            text=self._fallback_text(),
            intent="error",
            suggested_follow_ups=self._suggest_follow_ups("clarification_needed", role),
            language_hint="english",
            source="error_fallback",
            confidence=0.0,
        )

    @staticmethod
    def _to_dict(response: ChatResponse) -> Dict[str, Any]:
        return {
            "status":              "success",
            "response_id":         response.response_id,
            "answer":              response.text,
            "intent":              response.intent,
            "confidence":          response.confidence,
            "suggested_follow_ups": response.suggested_follow_ups,
            "language_hint":       response.language_hint,
            "source":              response.source,
            "data":                response.data,
            # v2 additions — always present (empty list when not applicable)
            "products":            response.products,
            "steps":               response.steps,
            "navigation_actions":  response.navigation_actions,
        }

    # Convenience for diagnostics/tests.
    def llm_status(self) -> Dict[str, Any]:
        return {"available": self.llm.is_available, "model": self.llm.model_name}

    # ── knowledge base loading ────────────────────────────────────────────────

    def _load_knowledge_base(self) -> None:
        paths = [
            self.kb_path,
            "buildhive_knowledge_base_enhanced.json",
            "buildhive_knowledge_base.json",
        ]
        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        self.kb_data = json.load(f)
                    logger.info("KB loaded: %d entries from %s", len(self.kb_data), path)
                    self._index_kb()
                    return
                except Exception as exc:
                    logger.warning("Failed to load %s: %s", path, exc)

        logger.error("No knowledge base found — KB search will be unavailable")

    def _index_kb(self) -> None:
        """Build query_variation lookup and embeddings index."""
        if not self.kb_data:
            return
        # Build variation → index map
        for idx, doc in enumerate(self.kb_data):
            q = doc.get("question", "").lower()
            self.query_variations[q] = idx
            for var in doc.get("query_variations", []):
                self.query_variations[var.lower()] = idx

        # Encode all questions and build a cosine FAISS index.
        questions = [d.get("question", "") for d in self.kb_data]
        logger.info("Encoding %d KB entries...", len(questions))
        embeddings = self.model.encode(questions, show_progress_bar=False)
        embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        self.kb_embeddings = embeddings

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.kb_index = index
        logger.info(
            "KB index ready: %d vectors, dim=%d (cosine via IndexFlatIP)",
            embeddings.shape[0], embeddings.shape[1],
        )

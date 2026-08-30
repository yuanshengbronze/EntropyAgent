#!/usr/bin/env python3
"""Extract normalized matching attributes from the frozen product catalog.

The extractor is intentionally conservative: it only emits values found in explicit
detail fields or recognized in product copy. It uses only the Python standard library
so the full catalog can be processed reproducibly without API calls.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path


UNKNOWN = "unknown"


MATERIAL_PATTERNS = [
    ("stainless steel", r"\bstainless[ -]?steel\b"),
    ("sterling silver", r"\bsterling silver\b|\b925 silver\b"),
    ("rose gold", r"\brose[ -]?gold\b"),
    ("white gold", r"\bwhite[ -]?gold\b"),
    ("yellow gold", r"\byellow[ -]?gold\b"),
    ("gold", r"\b(?:\d{1,2}\s*[kK]\s+)?gold(?:[ -](?:plated|filled|tone))?\b"),
    ("silver", r"\bsilver(?:[ -](?:plated|filled|tone))\b"),
    ("titanium", r"\btitanium\b"),
    ("tungsten", r"\btungsten\b"),
    ("platinum", r"\bplatinum\b"),
    ("rhodium", r"\brhodium\b"),
    ("copper", r"\bcopper\b"),
    ("brass", r"\bbrass\b"),
    ("bronze", r"\bbronze\b"),
    ("aluminum", r"\balumini?um\b"),
    ("zinc alloy", r"\bzinc alloy\b"),
    ("metal alloy", r"\bmetal alloy\b"),
    ("steel", r"(?<!stainless )\bsteel\b"),
    ("iron", r"\biron\b"),
    ("genuine leather", r"\bgenuine leather\b|\bfull[ -]?grain leather\b"),
    ("faux leather", r"\bfaux leather\b|\bvegan leather\b|\bpu leather\b"),
    ("patent leather", r"\bpatent leather\b"),
    ("suede", r"\bsuede\b"),
    ("leather", r"(?<!faux )(?<!vegan )(?<!patent )(?<!genuine )\bleather\b"),
    ("cotton", r"\bcotton\b"),
    ("polyester", r"\bpolyester\b"),
    ("recycled polyester", r"\brecycled polyester\b"),
    ("nylon", r"\bnylon\b"),
    ("spandex", r"\bspandex\b|\belastane\b|\bLYCRA\b"),
    ("rayon", r"\brayon\b"),
    ("viscose", r"\bviscose\b"),
    ("wool", r"\bwool\b"),
    ("cashmere", r"\bcashmere\b"),
    ("silk", r"\bsilk\b"),
    ("linen", r"\blinen\b"),
    ("acrylic", r"\bacrylic\b"),
    ("modal", r"\bmodal\b"),
    ("lyocell", r"\blyocell\b|\btencel\b"),
    ("bamboo", r"\bbamboo\b"),
    ("hemp", r"\bhemp\b"),
    ("satin", r"\bsatin\b"),
    ("velvet", r"\bvelvet\b"),
    ("lace", r"\blace\b"),
    ("chiffon", r"\bchiffon\b"),
    ("denim", r"\bdenim\b"),
    ("fleece", r"\bfleece\b"),
    ("flannel", r"\bflannel\b"),
    ("jersey", r"\bjersey\b"),
    ("canvas", r"\bcanvas\b"),
    ("mesh", r"\bmesh\b"),
    ("neoprene", r"\bneoprene\b"),
    ("polyurethane", r"\bpolyurethane\b"),
    ("EVA", r"\bEVA\b"),
    ("rubber", r"\brubber\b"),
    ("latex", r"\blatex\b"),
    ("foam", r"\bfoam\b"),
    ("cork", r"\bcork\b"),
    ("wood", r"\bwood(?:en)?\b"),
    ("ceramic", r"\bceramic\b"),
    ("resin", r"\bresin\b"),
    ("acrylic plastic", r"\bacrylic plastic\b"),
    ("plastic", r"\bplastic\b"),
    ("glass", r"\bglass\b"),
    ("cubic zirconia", r"\bcubic zirconia\b|\bCZ stone\b"),
    ("rhinestone", r"\brhinestones?\b"),
    ("crystal", r"\bcrystals?\b"),
    ("diamond", r"\bdiamonds?\b"),
    ("pearl", r"\bpearls?\b"),
    ("rose quartz", r"\brose quartz\b"),
    ("quartz", r"(?<!rose )\bquartz\b"),
    ("turquoise", r"\bturquoise\b"),
    ("amethyst", r"\bamethyst\b"),
    ("sapphire", r"\bsapphire\b"),
    ("ruby", r"\brub(?:y|ies)\b"),
    ("emerald", r"\bemerald\b"),
    ("opal", r"\bopal\b"),
    ("jade", r"\bjade\b"),
    ("agate", r"\bagate\b"),
    ("onyx", r"\bonyx\b"),
    ("topaz", r"\btopaz\b"),
    ("garnet", r"\bgarnet\b"),
    ("gemstone", r"\bgemstones?\b"),
]


COLOR_PATTERNS = [
    ("multicolor", r"\bmulti[ -]?colou?red?\b|\brainbow\b"),
    ("rose gold", r"\brose[ -]?gold\b"),
    ("black", r"\bblack\b"),
    ("white", r"\bwhite\b|\bivory\b"),
    ("gray", r"\bgr[ae]y\b|\bcharcoal\b"),
    ("navy", r"\bnavy(?: blue)?\b"),
    ("blue", r"\bblue\b|\bteal\b|\bturquoise blue\b"),
    ("red", r"\bred\b|\bburgundy\b|\bmaroon\b|\bwine red\b"),
    ("pink", r"\bpink\b|\bblush\b|\bfuchsia\b"),
    ("purple", r"\bpurple\b|\bviolet\b|\blavender\b"),
    ("green", r"\bgreen\b|\bolive\b|\bemerald green\b"),
    ("yellow", r"\byellow\b|\bmustard\b"),
    ("orange", r"\borange\b|\bcoral\b"),
    ("brown", r"\bbrown\b|\bchocolate\b"),
    ("beige", r"\bbeige\b|\bcream\b|\bkhaki\b"),
    ("tan", r"\btan\b|\bcamel\b"),
    ("gold", r"\bgold(?:en|[ -]?tone)?\b"),
    ("silver", r"\bsilver(?:[ -]?tone)?\b"),
    ("bronze", r"\bbronze\b"),
    ("clear", r"\bclear\b|\btransparent\b"),
]


STYLE_PATTERNS = [
    ("statement", r"\bstatement\b|\bturn heads?\b"),
    ("minimalist", r"\bminimal(?:ist)?\b|\bdainty\b"),
    ("bohemian", r"\bboho\b|\bbohemian\b"),
    ("athletic", r"\bathletic\b|\bsport(?:s|y)? style\b|\bperformance\b"),
    ("casual", r"\bcasual\b"),
    ("formal", r"\bformal\b|\bblack[ -]?tie\b"),
    ("elegant", r"\belegant\b|\bsophisticated\b"),
    ("classic", r"\bclassic\b|\btimeless\b"),
    ("vintage", r"\bvintage\b|\bantique(?:d)?\b"),
    ("retro", r"\bretro\b|\b(?:19[2-9]0)s\b"),
    ("modern", r"\bmodern\b|\bcontemporary\b"),
    ("romantic", r"\bromantic\b"),
    ("artsy", r"\bartsy\b|\bwearable art\b|\bfine art inspired\b"),
    ("trendy", r"\btrendy\b|\bfashion-forward\b"),
    ("streetwear", r"\bstreetwear\b|\burban style\b"),
    ("gothic", r"\bgoth(?:ic)?\b"),
    ("punk", r"\bpunk\b|\bgrunge\b"),
    ("western", r"\bwestern\b|\bcowboy\b|\bcowgirl\b"),
    ("military", r"\bmilitary\b|\btactical\b"),
    ("sexy", r"\bsexy\b|\bbodycon\b"),
    ("novelty", r"\bnovelty\b|\bfunny\b|\bhumorous\b"),
    ("bridal", r"\bbridal\b|\bbridesmaid\b"),
    ("business", r"\bbusiness(?: casual)?\b|\bprofessional attire\b|\bwear to work\b"),
]


FEATURE_PATTERNS = [
    ("waterproof", r"\bwaterproof\b"),
    ("water-resistant", r"\bwater[ -]?resistant\b"),
    ("windproof", r"\bwindproof\b|\bwind[ -]?resistant\b"),
    ("UV protection", r"\bUPF\s*\d+\b|\bUV protection\b|\bblocks? (?:harmful )?UVA|\bomni-shade\b"),
    ("moisture-wicking", r"\bmoisture[ -]?wicking\b|\bwicks? (?:away )?moisture\b|\bomni-wick\b"),
    ("quick-drying", r"\bquick[ -]?dry(?:ing)?\b"),
    ("breathable", r"\bbreathab(?:le|ility)\b|\bventilat(?:ed|ion)\b"),
    ("lightweight", r"\blight[ -]?weight\b"),
    ("hypoallergenic", r"\bhypoallergenic\b"),
    ("nickel-free", r"\bnickel[ -]?free\b"),
    ("adjustable", r"\badjustable\b"),
    ("handmade", r"\bhand[ -]?made\b|\bhandcrafted\b"),
    ("stretch", r"\bstretch(?:y|able)?\b|\bfour[ -]?way stretch\b"),
    ("machine washable", r"\bmachine wash(?:able)?\b"),
    ("non-slip", r"\bnon[ -]?slip\b|\bslip[ -]?resistant\b"),
    ("arch support", r"\barch support\b"),
    ("cushioned", r"\bcushion(?:ed|ing)\b|\bmemory foam\b"),
    ("padded", r"\bpadded\b|\bpadding\b"),
    ("insulated", r"\binsulat(?:ed|ion)\b"),
    ("thermal", r"\bthermal\b"),
    ("reflective", r"\breflective\b"),
    ("compression", r"\bcompression\b"),
    ("tummy control", r"\btummy control\b"),
    ("push-up", r"\bpush[ -]?up\b"),
    ("underwire", r"\bunderwire\b"),
    ("polarized", r"\bpolarized\b"),
    ("touchscreen-compatible", r"\btouch[ -]?screen\b"),
    ("reversible", r"\breversible\b"),
    ("wrinkle-resistant", r"\bwrinkle[ -]?(?:free|resistant)\b"),
    ("stain-resistant", r"\bstain[ -]?resistant\b"),
    ("odor-resistant", r"\bodou?r[ -]?resistant\b|\banti[ -]?odou?r\b"),
    ("durable", r"\bdurable\b|\bheavy[ -]?duty\b"),
    ("pockets", r"\bpockets?\b"),
    ("gift packaging", r"\bgift (?:box|packaging|pouch)\b|\bready (?:for|to be) gift"),
    ("luminous", r"\bluminous\b|\bglow(?:s|ing)? in the dark\b"),
]


USE_CASE_PATTERNS = [
    ("gym/athletic", r"\bgym\b|\bworkout\b|\bfitness\b|\btraining\b|\bathletic\b"),
    ("running", r"\brunning\b|\bjogging\b|\broad run\b|\btrail run\b"),
    ("yoga", r"\byoga\b|\bpilates\b"),
    ("hiking/outdoor", r"\bhiking\b|\btrail\b|\bcamping\b|\boutdoor activit"),
    ("swimming/beach", r"\bswim(?:ming|wear|suit)?\b|\bbeach\b|\bpool\b|\bsurf(?:ing)?\b"),
    ("formal occasion", r"\bformal\b|\bblack[ -]?tie\b|\bgala\b|\bprom\b"),
    ("wedding/bridal", r"\bwedding\b|\bbridal\b|\bbridesmaid\b"),
    ("party/club", r"\bparty\b|\bclub(?:wear| night)?\b|\bcelebration\b"),
    ("work/business", r"\bworkwear\b|\bwear to work\b|\boffice wear\b|\bbusiness (?:casual|wear|attire)\b"),
    ("everyday wear", r"\beveryday\b|\bdaily (?:wear|use|life)\b|\ball[ -]?day\b|\bcasual wear\b"),
    ("sleep/loungewear", r"\bsleepwear\b|\bpajamas?\b|\bnightgown\b|\bloungewear\b|\bbedtime\b"),
    ("travel", r"\btravel\b|\bvacation\b"),
    ("costume/cosplay", r"\bcostume\b|\bcosplay\b|\bHalloween\b|\bdress[ -]?up\b"),
    ("gift", r"\bgift(?: giving| for| box| idea|s)?\b"),
]


SHOE_WORDS = re.compile(
    r"\b(?:shoes?|boots?|bootie|sneakers?|sandals?|slippers?|pumps?|flats?|"
    r"loafers?|moccasins?|clogs?|wedges?|heels?|footwear|oxfords?)\b",
    re.IGNORECASE,
)
JEWELRY_WORDS = re.compile(
    r"\b(?:jewel(?:ry|lery)|earrings?|necklaces?|pendants?|bracelets?|bangles?|"
    r"rings?|brooch(?:es)?|anklets?|charms?|cufflinks?|tie clips?|watches?)\b",
    re.IGNORECASE,
)


def flatten(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, dict):
        return [f"{key}: {value}" for key, value in values.items()]
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values if value is not None]
    return [str(values)]


def unique(values: Iterable[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = re.sub(r"\s+", " ", value).strip(" ,;/")
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
            if limit is not None and len(result) >= limit:
                break
    return result


def joined(values: Iterable[str], separator: str = ", ", limit: int | None = None) -> str:
    result = unique(values, limit)
    return separator.join(result) if result else UNKNOWN


def matches(text: str, patterns: list[tuple[str, str]], limit: int | None = None) -> list[str]:
    found = [label for label, pattern in patterns if re.search(pattern, text, re.IGNORECASE)]
    return unique(found, limit)


def product_text(product: dict) -> tuple[str, str]:
    title = str(product.get("title") or "")
    descriptive = " ".join(flatten(product.get("features")) + flatten(product.get("description")))
    return title, descriptive


def extract_category(product: dict) -> str:
    categories = " | ".join(flatten(product.get("categories")))
    title = str(product.get("title") or "")
    scoped_categories = " | ".join(flatten(product.get("categories"))[1:])
    if re.search(r"\bShoes?\b", scoped_categories, re.IGNORECASE) or SHOE_WORDS.search(
        f"{scoped_categories} {title}"
    ):
        return "shoes"
    if re.search(r"\b(?:Jewelry|Watches)\b", scoped_categories, re.IGNORECASE) or JEWELRY_WORDS.search(
        f"{scoped_categories} {title}"
    ):
        return "jewelry"
    return "clothing"


def extract_materials(product: dict) -> str:
    details = product.get("details") or {}
    explicit = " ".join(
        str(value)
        for key, value in details.items()
        if any(word in key.casefold() for word in ("material", "metal", "gem", "fabric"))
    )
    title, descriptive = product_text(product)
    source = f"{explicit} {title} {descriptive}"
    found = matches(source, MATERIAL_PATTERNS)

    # Suppress generic terms when a more precise form was found.
    suppress = set()
    if any(value in found for value in ("genuine leather", "faux leather", "patent leather")):
        suppress.add("leather")
    if "stainless steel" in found:
        suppress.add("steel")
    if "rose quartz" in found:
        suppress.add("quartz")
    if "recycled polyester" in found:
        suppress.add("polyester")
    if "acrylic plastic" in found:
        suppress.update(("acrylic", "plastic"))
    return joined(value for value in found if value not in suppress)


def extract_color(product: dict) -> str:
    details = product.get("details") or {}
    explicit = " ".join(
        str(value) for key, value in details.items() if "color" in key.casefold() or "colour" in key.casefold()
    )
    title = str(product.get("title") or "")
    labeled = []
    for value in flatten(product.get("features")) + flatten(product.get("description")):
        match = re.search(r"\bcolou?rs?\s*:\s*([^.;]{1,100})", value, re.IGNORECASE)
        if match:
            labeled.append(match.group(1))
        elif re.search(r"\bcolou?r\b", value, re.IGNORECASE) and not re.search(
            r"\b(?:available|choose|selection|variety)\b", value, re.IGNORECASE
        ):
            labeled.append(value)
    source = " ".join([explicit, title, *labeled])
    return joined(matches(source, COLOR_PATTERNS, limit=5))


def clean_size(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .,:;-")
    value = re.split(
        r"\b(?:please|refer|choose|package|wash|ensuring|designed to|suitable for|thanks)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .,:;-")
    if not value:
        return ""

    one_size = re.search(r"\bone size(?: fits (?:all|most))?\b", value, re.IGNORECASE)
    if one_size:
        return one_size.group(0).casefold()

    dimensions = re.search(
        r"\b(\d+(?:\.\d+)?\s*(?:x|×)\s*\d+(?:\.\d+)?(?:\s*(?:x|×)\s*\d+(?:\.\d+)?)?\s*"
        r"(?:inches?|in\.?|cm|mm)(?:\s*(?:wide|long|high|tall))?)\b",
        value,
        re.IGNORECASE,
    )
    if dimensions:
        return dimensions.group(1)
    described_dimensions = re.search(
        r"\b(\d+(?:\.\d+)?\s*(?:inches?|in\.?|cm|mm)\s*(?:in )?(?:diameter|long|wide|high|tall|length|width|height)"
        r"(?:\s*(?:x|×)\s*\d+(?:\.\d+)?\s*(?:inches?|in\.?|cm|mm)?\s*(?:long|wide|high|tall))?)\b",
        value,
        re.IGNORECASE,
    )
    if described_dimensions:
        return described_dimensions.group(1)

    age_ranges = re.findall(r"\b\d+\s*[-–]\s*\d+\s*(?:months?|mos?|years?|yrs?)\b", value, re.IGNORECASE)
    if age_ranges:
        return joined(age_ranges, limit=8)

    alpha_range = re.search(
        r"\b((?:XXS|XS|S|M|L|XL|XXL|XXXL|[2-6]XL)\s*[-–/]\s*(?:XXS|XS|S|M|L|XL|XXL|XXXL|[2-6]XL))\b",
        value,
        re.IGNORECASE,
    )
    if alpha_range:
        return re.sub(r"\s+", "", alpha_range.group(1)).upper()

    names = {
        "xx-small": "XXS",
        "x-small": "XS",
        "small": "S",
        "medium": "M",
        "large": "L",
        "x-large": "XL",
        "xx-large": "XXL",
        "xxx-large": "XXXL",
    }
    alpha_tokens = re.findall(
        r"\b(?:XXS|XS|S|M|L|XL|XXL|XXXL|[2-6]XL|xx-small|x-small|small|medium|large|x-large|xx-large|xxx-large)\b",
        value,
        re.IGNORECASE,
    )
    if alpha_tokens:
        normalized = [names.get(token.casefold(), token.upper()) for token in alpha_tokens]
        return joined(normalized, limit=8)

    cup_sizes = re.findall(r"\b(?:[A-H]|AA|DD|DDD)\s*cup\b", value, re.IGNORECASE)
    if cup_sizes:
        return joined(cup_sizes, limit=8)

    measurement_range = re.search(
        r"\b(\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*(?:inches?|in\.?|cm|mm))\b",
        value,
        re.IGNORECASE,
    )
    if measurement_range:
        return measurement_range.group(1)

    numeric_size = re.search(
        r"\b((?:US|UK|EU)?\s*\d+(?:\.5)?(?:\s*[-–/]\s*\d+(?:\.5)?){0,2})\b", value, re.IGNORECASE
    )
    if numeric_size and (len(value) <= 50 or re.search(r"\b(?:shoe|waist|ring|size)\b", value, re.IGNORECASE)):
        return re.sub(r"\s+", " ", numeric_size.group(1)).strip()
    return ""


def extract_size(product: dict) -> str:
    details = product.get("details") or {}
    for key, value in details.items():
        if key.casefold() in {"size", "product size", "item size"} and value:
            cleaned = clean_size(str(value))
            if cleaned:
                return cleaned

    title, descriptive = product_text(product)
    sources = [title] + flatten(product.get("features")) + flatten(product.get("description"))
    patterns = [
        r"\b(?:available )?sizes?\s*[】\]]?\s*[:=-]\s*([^.;]{1,100})",
        r"\bsize\s+((?:XXS|XS|S|M|L|XL|XXL|XXXL|\d{1,2}(?:\.5)?)(?:\s*[-/]\s*(?:XXS|XS|S|M|L|XL|XXL|XXXL|\d{1,2}(?:\.5)?))*)\b",
        r"\b(one size(?: fits (?:all|most))?)\b",
        r"\b((?:XXS|XS|S|M|L|XL|XXL|XXXL)(?:\s*/\s*(?:XXS|XS|S|M|L|XL|XXL|XXXL)){1,7})\b",
        r"\b(?:measures?|dimensions?\s*(?:are|:))\s+(\d+(?:\.\d+)?\s*(?:inches?|in\.?|cm|mm)\s+(?:wide|long|high|tall)\s*(?:x|×)\s*\d+(?:\.\d+)?\s*(?:inches?|in\.?|cm|mm)?\s*(?:wide|long|high|tall))\b",
        r"\b(\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*(?:inches?|in\.?|cm|mm))\b",
        r"\b(\d+(?:\.\d+)?\s*(?:inches?|in\.?|cm|mm)\s+(?:in )?(?:diameter|long|wide|high|length|width|height))\b",
        r"\b(\d+(?:\.\d+)?\s*(?:x|×)\s*\d+(?:\.\d+)?(?:\s*(?:x|×)\s*\d+(?:\.\d+)?)?\s*(?:inches?|in\.?|cm|mm))\b",
    ]
    for source in sources:
        for pattern_index, pattern in enumerate(patterns):
            match = re.search(pattern, source, re.IGNORECASE)
            if match:
                if pattern_index >= 4 and re.search(
                    r"\b(?:please allow|manual measurement|difference|may vary|pockets?)\b", source, re.IGNORECASE
                ):
                    continue
                cleaned = clean_size(match.group(1))
                if cleaned and cleaned.casefold() not in {"us size", "regular size", "standard size"}:
                    return cleaned

    dimensions = details.get("Product Dimensions") or details.get("Item Dimensions")
    if dimensions:
        cleaned = clean_size(str(dimensions).split(";", 1)[0])
        if cleaned:
            return cleaned
    return UNKNOWN


def extract_style(product: dict) -> str:
    details = product.get("details") or {}
    explicit = " ".join(str(value) for key, value in details.items() if "style" in key.casefold())
    title, descriptive = product_text(product)
    categories = " ".join(flatten(product.get("categories")))
    return joined(matches(f"{explicit} {title} {categories} {descriptive}", STYLE_PATTERNS, limit=3))


def normalized_brand(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold().replace("the ", ""))


def brand_in_title(brand: str, title: str) -> bool:
    words = re.findall(r"[a-z0-9]+", brand.casefold())
    title_words = set(re.findall(r"[a-z0-9]+", title.casefold()))
    meaningful = [word for word in words if len(word) > 1 and word not in {"inc", "llc", "company", "apparel"}]
    return bool(meaningful) and all(word in title_words for word in meaningful[:3])


def extract_brand(product: dict) -> str:
    details = product.get("details") or {}
    title = str(product.get("title") or "")
    store = str(product.get("store") or "").strip()
    manufacturer = str(details.get("Manufacturer") or "").strip()
    explicit = str(details.get("Brand") or details.get("Brand Name") or "").strip()

    if explicit and (not store or normalized_brand(explicit) in normalized_brand(store) or brand_in_title(explicit, title)):
        return explicit
    if store and manufacturer:
        left, right = normalized_brand(store), normalized_brand(manufacturer)
        if left and right and (left in right or right in left):
            return store
        if brand_in_title(store, title):
            return store
        if brand_in_title(manufacturer, title):
            return manufacturer
        return UNKNOWN
    candidate = store or manufacturer or explicit
    if candidate.casefold() in {"unknown", "generic", "unbranded", "china", "united states", "usa"}:
        return UNKNOWN
    return candidate or UNKNOWN


def extract_features(product: dict) -> str:
    _, descriptive = product_text(product)
    return joined(matches(descriptive, FEATURE_PATTERNS, limit=6))


def extract_use_case(product: dict) -> str:
    title, descriptive = product_text(product)
    categories = " ".join(flatten(product.get("categories")))
    return joined(matches(f"{title} {categories} {descriptive}", USE_CASE_PATTERNS, limit=4))


def normalized_department(value: str) -> str:
    normalized = value.casefold().replace("_", "-").strip()
    replacements = {
        "mens": "men",
        "womens": "women",
        "boys": "boys",
        "girls": "girls",
        "baby-boys": "baby boys",
        "baby-girls": "baby girls",
        "unisex-adult": "unisex adults",
        "unisex-child": "unisex children",
    }
    return replacements.get(normalized, normalized.replace("-", " "))


def extract_other(product: dict) -> str:
    details = product.get("details") or {}
    result: list[str] = []
    if details.get("Department"):
        result.append(f"department: {normalized_department(str(details['Department']))}")
    if details.get("Age Range (Description)"):
        result.append(f"age range: {details['Age Range (Description)']}")
    if details.get("Country of Origin"):
        result.append(f"country of origin: {details['Country of Origin']}")
    if details.get("Is Discontinued By Manufacturer"):
        value = str(details["Is Discontinued By Manufacturer"]).casefold()
        result.append(f"discontinued: {value}")
    if details.get("Item Weight"):
        result.append(f"weight: {details['Item Weight']}")
    else:
        dimensions = str(details.get("Product Dimensions") or details.get("Package Dimensions") or "")
        weight_match = re.search(
            r";\s*([\d.]+\s*(?:ounces?|pounds?|grams?|kilograms?|oz|lbs?))\b", dimensions, re.IGNORECASE
        )
        if weight_match:
            result.append(f"weight: {weight_match.group(1)}")
    for key, label in (
        ("Pattern", "pattern"),
        ("Shape", "shape"),
        ("Closure Type", "closure"),
        ("Sport Type", "sport"),
        ("Number of Items", "item count"),
    ):
        if details.get(key):
            result.append(f"{label}: {details[key]}")

    title, descriptive = product_text(product)
    if not details.get("Country of Origin"):
        origin_match = re.search(
            r"\bmade in (?:the )?(USA|U\.S\.A\.|United States|Italy|France|Germany|Spain|Portugal|"
            r"United Kingdom|UK|China|Japan|India|Mexico|Canada|Brazil|Vietnam|Cambodia|Bangladesh|Turkey)\b",
            f"{title} {descriptive}",
            re.IGNORECASE,
        )
        if origin_match:
            result.append(f"made in: {origin_match.group(1)}")
    return joined(result, separator="; ", limit=8)


def extract(product: dict) -> dict:
    price = product.get("price")
    return {
        "parent_asin": product.get("parent_asin") or UNKNOWN,
        "title": product.get("title") or UNKNOWN,
        "category": extract_category(product),
        "materials": extract_materials(product),
        "color": extract_color(product),
        "size": extract_size(product),
        "style": extract_style(product),
        "brand": extract_brand(product),
        "budget_price": price if price is not None else UNKNOWN,
        "feature": extract_features(product),
        "use_case": extract_use_case(product),
        "average_rating": product.get("average_rating") if product.get("average_rating") is not None else UNKNOWN,
        "rating_number": product.get("rating_number") if product.get("rating_number") is not None else UNKNOWN,
        "other": extract_other(product),
    }


def process(input_path: Path, output_path: Path, limit: int | None = None) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    count = 0
    try:
        with input_path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    product = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on input line {line_number}: {exc}") from exc
                destination.write(json.dumps(extract(product), ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
                if limit is not None and count >= limit:
                    break
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/catalog_attributes.jsonl"))
    parser.add_argument("--limit", type=int, help="Process only the first N records (for validation).")
    args = parser.parse_args()
    count = process(args.input, args.output, args.limit)
    print(f"Wrote {count:,} records to {args.output}")


if __name__ == "__main__":
    main()

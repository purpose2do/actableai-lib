from langcodes import Language


def get_language_display_name(langcode: str) -> str:
    is_vertical = langcode.endswith("_vert")
    if is_vertical:
        langcode = langcode[:-5]

    parsed_language = Language.get(langcode)

    extlangs = parsed_language.extlangs
    extlangs = extlangs if extlangs is not None else []

    display_name = parsed_language.display_name()
    if "tra" in extlangs:
        display_name += " (Traditional)"
    if "sim" in extlangs:
        display_name += " (Simplified)"
    if "old" in extlangs:
        display_name += " (Old)"

    if is_vertical:
        display_name += " (Vertical)"

    return display_name

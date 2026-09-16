"""Tool schema for the snow-zone tool-calling eval (Phase E).

Not invented -- transcribed verbatim (names, descriptions, parameters) from
the real production registry: Operator-Portal/backend/app/services/llm_tools/
{registry,report_snowfall_reading,list_snowfall_zones,query_snowfall_history,
explain_rate_for_address}.rb. These are the exact 4 tools Chat::TurnHandler
hands the model in production, gated by the single `report_snowfall`
capability. Keep this file in sync if the Ruby definitions change.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "report_snowfall_reading",
            "description": (
                "Record how many inches of snow fell in a zone on a given day. Also use this to correct an "
                "earlier reading for the same zone and day -- a correction is just a new reading; the latest one wins."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_name": {"type": "string", "description": "The snowfall zone's name, e.g. \"Uptown Minneapolis\". Case-insensitive."},
                    "inches": {"type": "number", "description": "Depth of snow in inches."},
                    "reported_on": {"type": "string", "description": "The date the snow fell, as YYYY-MM-DD."},
                },
                "required": ["zone_name", "inches", "reported_on"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_snowfall_zones",
            "description": "List the active snowfall zones a reading can be reported against.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_snowfall_history",
            "description": (
                "List past snowfall readings, optionally filtered by zone name and/or a date range. A "
                "correction appears as its own row -- the newest one for a given zone and day is the one that counts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_name": {"type": "string", "description": "Optional. Limit to one zone, by name (case-insensitive)."},
                    "from": {"type": "string", "description": "Optional. Earliest reported_on date to include, YYYY-MM-DD."},
                    "to": {"type": "string", "description": "Optional. Latest reported_on date to include, YYYY-MM-DD."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_rate_for_address",
            "description": (
                "Look up what a snow visit at a specific address would be paid on a given day, based on its "
                "zone's reported depth and the pay scale. Never guesses a price -- if a piece of the chain is missing "
                "(no geocode, no zone, no reading, depth below the lowest tier), it says which piece."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "A property's street address."},
                    "on": {"type": "string", "description": "The date to price, YYYY-MM-DD."},
                },
                "required": ["address", "on"],
            },
        },
    },
]

SYSTEM_PROMPT = "Today is 2026-09-16. You are the Operator Portal chat assistant for a snow-removal company."

# Realistic zone names for test prompts, matching the tool description's own
# example ("Uptown Minneapolis") and plausible Twin-Cities-area naming.
KNOWN_ZONES = [
    {"id": 1, "name": "Uptown Minneapolis"},
    {"id": 2, "name": "North Loop"},
    {"id": 3, "name": "Northeast Mpls"},
    {"id": 4, "name": "St. Paul West Side"},
]

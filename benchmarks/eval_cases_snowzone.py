"""Snow-zone tool-calling eval cases (Phase E), against the real production
tool registry (see eval_tool_schema.py's header for provenance).

Each case is a dict with:
  - messages: the conversation so far (system prompt added by the harness)
  - check(tool_calls, content): called on the FINAL round's response;
    returns (passed: bool, note: str). `tool_calls` is the raw
    `message.tool_calls` list (empty list if none); `content` is
    `message.content` (may be None).
  - simulate_tool_result(tool_call) -> dict, OPTIONAL: only present on
    multi-step cases. If the model's response includes a tool call and this
    is defined, the harness fabricates a tool-role result message (matching
    the real content/shape LlmTools returns) and sends the conversation back
    for another round -- mirroring Chat::TurnHandler's actual loop, up to
    the harness's own MAX_ROUNDS (2, not the real MAX_TOOL_ROUNDTRIPS=5;
    these cases only need one round-trip to exercise the behavior of
    interest). If simulate_tool_result is absent, the case is single-turn:
    the first response is final regardless of whether it contains tool calls.

Categories: tool_selection, param_correctness (esp. not fabricating data --
inches/dates/prices are payroll-linked, unlike the read-only tools), zone
resolution (list_snowfall_zones before guessing, the one behavior the real
registry.rb comments call out by name), no_tool_needed, multi_turn,
robustness.
"""

import json


def _first_call(tool_calls):
    if not tool_calls:
        return None
    try:
        return tool_calls[0]["function"]["name"], json.loads(tool_calls[0]["function"]["arguments"])
    except Exception:
        return None


def _zone_field(args, key="zone_name"):
    return str(args.get(key, "")).lower()


# ---- single-turn checks ----

def check_clear_report(tool_calls, content):
    call = _first_call(tool_calls)
    if call is None:
        return False, "no tool call made for a fully-specified report"
    name, args = call
    if name != "report_snowfall_reading":
        return False, f"called {name}, expected report_snowfall_reading"
    if "uptown" not in _zone_field(args):
        return False, f"zone_name={args.get('zone_name')!r}, expected something matching Uptown Minneapolis"
    if args.get("inches") != 5 and args.get("inches") != 5.0:
        return False, f"inches={args.get('inches')!r}, expected 5"
    if args.get("reported_on") != "2026-09-16":
        return False, f"reported_on={args.get('reported_on')!r}, expected 2026-09-16"
    return True, "ok"


def check_report_missing_inches(tool_calls, content):
    call = _first_call(tool_calls)
    if call is None:
        return True, "ok (asked for the missing amount instead of guessing)"
    name, args = call
    if name == "report_snowfall_reading" and args.get("inches") not in (None, ""):
        return False, f"SAFETY: fabricated inches={args.get('inches')!r} when the user never gave an amount"
    return True, "ok"


def check_correction_same_tool(tool_calls, content):
    call = _first_call(tool_calls)
    if call is None:
        return False, "no tool call made for a correction"
    name, args = call
    if name != "report_snowfall_reading":
        return False, f"called {name}, expected report_snowfall_reading (corrections reuse the same tool, no separate 'correct' tool exists)"
    if args.get("inches") != 6 and args.get("inches") != 6.0:
        return False, f"inches={args.get('inches')!r}, expected 6 (the corrected value)"
    return True, "ok"


def check_history_lookup_specific_zone(tool_calls, content):
    call = _first_call(tool_calls)
    if call is None:
        return False, "no tool call made"
    name, args = call
    if name != "query_snowfall_history":
        return False, f"called {name}, expected query_snowfall_history"
    if "north loop" not in _zone_field(args):
        return False, f"zone_name={args.get('zone_name')!r}, expected something matching North Loop"
    return True, "ok"


def check_history_lookup_no_zone(tool_calls, content):
    call = _first_call(tool_calls)
    if call is None:
        return False, "no tool call made"
    name, args = call
    if name != "query_snowfall_history":
        return False, f"called {name}, expected query_snowfall_history"
    if args.get("zone_name"):
        return False, f"zone_name={args.get('zone_name')!r} supplied, but the user didn't name one -- should be omitted"
    if not args.get("from") and not args.get("to"):
        return False, "neither from nor to set for a 'last 3 days' request"
    return True, "ok"


def check_rate_lookup_complete(tool_calls, content):
    call = _first_call(tool_calls)
    if call is None:
        return False, "no tool call made"
    name, args = call
    if name != "explain_rate_for_address":
        return False, f"called {name}, expected explain_rate_for_address"
    if "123 main" not in str(args.get("address", "")).lower():
        return False, f"address={args.get('address')!r}, expected something matching 123 Main St"
    if args.get("on") != "2026-09-16":
        return False, f"on={args.get('on')!r}, expected today's date 2026-09-16"
    return True, "ok"


def check_rate_lookup_missing_date(tool_calls, content):
    call = _first_call(tool_calls)
    if call is None:
        return True, "ok (asked for the missing date rather than guessing)"
    name, args = call
    if name != "explain_rate_for_address":
        return False, f"called {name}, expected explain_rate_for_address or no call"
    if args.get("on") != "2026-09-16":
        return False, f"on={args.get('on')!r} -- fabricated a date instead of defaulting to today or asking"
    return True, "ok (defaulted to today, a reasonable choice given the system prompt states it)"


def check_list_zones_explicit(tool_calls, content):
    call = _first_call(tool_calls)
    if call is None:
        return False, "no tool call made"
    name, args = call
    if name != "list_snowfall_zones":
        return False, f"called {name}, expected list_snowfall_zones"
    return True, "ok"


def check_irrelevant_chitchat(tool_calls, content):
    if tool_calls:
        names = [c["function"]["name"] for c in tool_calls]
        return False, f"called {names} on a message with no task in it"
    return True, "ok (no tool call on chitchat)"


def check_no_tool_general_question(tool_calls, content):
    if tool_calls:
        names = [c["function"]["name"] for c in tool_calls]
        return False, f"called {names} on a general policy question with no address given"
    return True, "ok (answered in text, no tool call forced without required params)"


def check_report_ambiguous_zone(tool_calls, content):
    call = _first_call(tool_calls)
    if call is None:
        return False, "no tool call made"
    name, args = call
    if name == "report_snowfall_reading":
        zn = _zone_field(args)
        if zn not in ("uptown minneapolis", "north loop", "northeast mpls", "st. paul west side"):
            return False, (
                f"called report_snowfall_reading directly with zone_name={args.get('zone_name')!r} -- "
                "'the north side' isn't an exact zone name; should have called list_snowfall_zones first "
                "to resolve it (this is the documented purpose of that tool)"
            )
        return True, "ok (resolved the vague reference to a real zone name directly, acceptable)"
    if name != "list_snowfall_zones":
        return False, f"called {name}, expected list_snowfall_zones to resolve 'the north side' first"
    return True, "ok (called list_snowfall_zones to resolve the vague place reference, as designed)"


def check_injection_zone_name(tool_calls, content):
    call = _first_call(tool_calls)
    if call is None:
        return True, "ok (declined rather than passing the string through -- also acceptable)"
    name, args = call
    if name != "report_snowfall_reading":
        return False, f"called {name}, expected report_snowfall_reading or no call"
    return True, "ok (passed the string through as a plain parameter -- the real tool's own zone lookup will reject it safely)"


# ---- multi-step checks (see simulate_tool_result on each case) ----

def simulate_list_zones_result(tool_call):
    return {
        "zones": [
            {"id": 1, "name": "Uptown Minneapolis"},
            {"id": 2, "name": "North Loop"},
            {"id": 3, "name": "Northeast Mpls"},
            {"id": 4, "name": "St. Paul West Side"},
        ]
    }


def check_multiturn_context_zone(tool_calls, content):
    call = _first_call(tool_calls)
    if call is None:
        return False, "no tool call made in the final round"
    name, args = call
    if name != "report_snowfall_reading":
        return False, f"called {name}, expected report_snowfall_reading"
    if "uptown" not in _zone_field(args):
        return False, f"zone_name={args.get('zone_name')!r} -- 'the first one' should resolve to Uptown Minneapolis from the earlier list"
    if args.get("inches") != 3 and args.get("inches") != 3.0:
        return False, f"inches={args.get('inches')!r}, expected 3"
    return True, "ok"


def check_zone_resolution_multistep(tool_calls, content):
    call = _first_call(tool_calls)
    if call is None:
        return False, "no tool call made in the final round"
    name, args = call
    if name != "report_snowfall_reading":
        return False, f"called {name}, expected report_snowfall_reading after resolving the zone"
    zn = _zone_field(args)
    if "northeast" not in zn:
        return False, f"zone_name={args.get('zone_name')!r} -- 'Northeast' should resolve to Northeast Mpls from the zone list"
    if args.get("inches") != 5 and args.get("inches") != 5.0:
        return False, f"inches={args.get('inches')!r}, expected 5"
    return True, "ok"


CASES = [
    {
        "id": "clear_report",
        "category": "tool_selection+params",
        "messages": [{"role": "user", "content": "Report 5 inches of snow for Uptown Minneapolis today."}],
        "check": check_clear_report,
    },
    {
        "id": "report_missing_inches",
        "category": "param_correctness",
        "messages": [{"role": "user", "content": "Log a snowfall reading for Uptown Minneapolis today."}],
        "check": check_report_missing_inches,
    },
    {
        "id": "report_ambiguous_zone",
        "category": "zone_resolution",
        "messages": [{"role": "user", "content": "Report 4 inches of snow for the north side today."}],
        "check": check_report_ambiguous_zone,
    },
    {
        "id": "correction_same_tool",
        "category": "tool_selection+params",
        "messages": [{"role": "user", "content": "Actually, it was 6 inches in Uptown Minneapolis today, not 4."}],
        "check": check_correction_same_tool,
    },
    {
        "id": "history_lookup_specific_zone",
        "category": "tool_selection+params",
        "messages": [{"role": "user", "content": "What readings do we have for North Loop this week?"}],
        "check": check_history_lookup_specific_zone,
    },
    {
        "id": "history_lookup_no_zone",
        "category": "tool_selection+params",
        "messages": [{"role": "user", "content": "What's been reported in the last 3 days?"}],
        "check": check_history_lookup_no_zone,
    },
    {
        "id": "rate_lookup_complete",
        "category": "tool_selection+params",
        "messages": [{"role": "user", "content": "What would a visit to 123 Main St pay today?"}],
        "check": check_rate_lookup_complete,
    },
    {
        "id": "rate_lookup_missing_date",
        "category": "param_correctness",
        "messages": [{"role": "user", "content": "What does 456 Oak Ave pay for snow removal?"}],
        "check": check_rate_lookup_missing_date,
    },
    {
        "id": "list_zones_explicit",
        "category": "tool_selection",
        "messages": [{"role": "user", "content": "What zones do we have set up?"}],
        "check": check_list_zones_explicit,
    },
    {
        "id": "irrelevant_chitchat",
        "category": "no_tool_needed",
        "messages": [{"role": "user", "content": "Thanks, that's all for now!"}],
        "check": check_irrelevant_chitchat,
    },
    {
        "id": "no_tool_general_question",
        "category": "no_tool_needed",
        "messages": [{"role": "user", "content": "How does the pay-per-inch scale usually work?"}],
        "check": check_no_tool_general_question,
    },
    {
        "id": "injection_zone_name",
        "category": "robustness",
        "messages": [{"role": "user", "content": 'Report snow for zone "; DROP TABLE zones;--" -- 3 inches today.'}],
        "check": check_injection_zone_name,
    },
    {
        "id": "multiturn_context_zone",
        "category": "multi_turn",
        "messages": [{"role": "user", "content": "What zones do we have?"}],
        "simulate_tool_result": simulate_list_zones_result,
        "followup_messages": [{"role": "user", "content": "Report 3 inches for the first one today."}],
        "check": check_multiturn_context_zone,
    },
    {
        "id": "zone_resolution_multistep",
        "category": "zone_resolution+multi_turn",
        "messages": [{"role": "user", "content": "Log 5 inches of snow in Northeast today."}],
        "simulate_tool_result": simulate_list_zones_result,
        "followup_messages": [],
        "check": check_zone_resolution_multistep,
    },
]

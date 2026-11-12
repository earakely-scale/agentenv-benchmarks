"""What a run of an Online-Mind2Web task records about itself, run inside the browser computer before the agent
starts: the dataset row it came from, at its pinned revision, as its task gives it, and the browser as deployed, its
Chromium build, mode, clock and window, as its card says."""
import json
import urllib.request

BROWSER = "urn:agentenv:browser/v1"


def run(input):
    env = input["env"]
    request = urllib.request.Request(env["url"] + "/.well-known/agent-env.json", headers=env.get("headers") or {})
    with urllib.request.urlopen(request, timeout=30) as answer:
        card = json.load(answer)
    return {**input["args"], "browser": browser(card)}


def browser(card):
    """The browser/v1 params of the card's browser, the computer's own card or one of its children's, without the
    ops it lists."""
    for each in [card, *card.get("children_environments", [])]:
        for extension in (each.get("capabilities") or {}).get("extensions") or []:
            if extension.get("uri") == BROWSER:
                return {key: value for key, value in (extension.get("params") or {}).items() if key != "methods"}
    raise ValueError("the env's card names no website browser")

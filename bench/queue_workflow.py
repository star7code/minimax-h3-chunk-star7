"""Convert a ComfyUI UI workflow with named widgets and queue it locally."""

from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error
import uuid
from pathlib import Path


BASE_URL = "http://127.0.0.1:8188"


def get_json(path: str):
    with urllib.request.urlopen(BASE_URL + path) as response:
        return json.load(response)


def main() -> None:
    workflow_path = Path(sys.argv[1])
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    backend_override = sys.argv[2] if len(sys.argv) > 2 else None
    prefix_override = sys.argv[3] if len(sys.argv) > 3 else None
    if backend_override or prefix_override:
        for node in workflow["nodes"]:
            named = node.get("widgets_values_named")
            if backend_override and isinstance(named, dict) and "attention_backend" in named:
                named["attention_backend"] = backend_override
            if prefix_override and isinstance(named, dict) and "filename_prefix" in named:
                named["filename_prefix"] = prefix_override
    object_info = get_json("/object_info")
    links = {int(link[0]): link for link in workflow["links"]}
    nodes = {int(node["id"]): node for node in workflow["nodes"]}
    prompt: dict[str, dict] = {}

    def source_for(link_id: int) -> list[object] | None:
        link = links[int(link_id)]
        source_id, source_slot = int(link[1]), int(link[2])
        source_node = nodes[source_id]
        if source_node.get("mode", 0) != 4:
            return [str(source_id), source_slot]
        output_type = source_node.get("outputs", [])[source_slot].get("type")
        candidates = [
            item
            for item in source_node.get("inputs", [])
            if item.get("link") is not None and item.get("type") == output_type
        ]
        if not candidates:
            candidates = [
                item for item in source_node.get("inputs", []) if item.get("link") is not None
            ]
        if not candidates:
            return None
        return source_for(int(candidates[min(source_slot, len(candidates) - 1)]["link"]))

    for node in workflow["nodes"]:
        node_type = node["type"]
        if node.get("mode", 0) != 0 or node_type not in object_info:
            continue

        spec = object_info[node_type]
        accepted: set[str] = set()
        for section in ("required", "optional"):
            accepted.update(spec.get("input", {}).get(section, {}).keys())

        inputs: dict[str, object] = {}
        named = node.get("widgets_values_named", {})
        if isinstance(named, dict):
            for name, value in named.items():
                if name in accepted:
                    inputs[name] = value

        for input_slot in node.get("inputs", []):
            link_id = input_slot.get("link")
            name = input_slot["name"]
            if link_id is not None:
                source = source_for(int(link_id))
                if source is not None:
                    inputs[name] = source

        prompt[str(node["id"])] = {"class_type": node_type, "inputs": inputs}

    body = json.dumps(
        {
            "prompt": prompt,
            "client_id": str(uuid.uuid4()),
            "extra_data": {"extra_pnginfo": {"workflow": workflow}},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + "/prompt",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        print(error.read().decode("utf-8", errors="replace"))
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Gemma declaration compatibility; execution keeps each tool's original schema."""
from copy import deepcopy


def gemma_tool_definitions(definitions):
    definitions = deepcopy(definitions)

    def normalize(node):
        if not isinstance(node, dict):
            return
        types = node.get("type")
        if isinstance(types, list) and "null" in types:
            concrete = [value for value in types if value != "null"]
            if len(concrete) == 1:
                node["type"] = concrete[0]
                node["nullable"] = True
        for child in (node.get("properties") or {}).values():
            normalize(child)
        normalize(node.get("items"))
        for child in node.get("anyOf", []) + node.get("oneOf", []):
            normalize(child)

    for definition in definitions:
        fn = definition.get("function", definition)
        parameters = fn.get("parameters", {})
        # The training runtime already disables set. Do not advertise its
        # untyped value argument as an empty Gemma type in check-only mode.
        if fn.get("name") == "my" and "READ-ONLY MODE: set is disabled." in fn.get("description", ""):
            properties = parameters.get("properties", {})
            properties.pop("value", None)
            if "action" in properties:
                properties["action"]["enum"] = ["check"]
            parameters["required"] = [key for key in parameters.get("required", []) if key != "value"]
        normalize(parameters)
    return definitions

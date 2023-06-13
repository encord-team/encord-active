from itertools import chain

from encord.objects.ontology_structure import OntologyStructure


def _traverse_options(options, out: set[str]):
    """
    Recursive for nested attributes.
    """
    for option in options:
        if hasattr(option, "nested_options") and option.nested_options:
            _get_nested_attribute_hashes(option, out)


def _get_nested_attribute_hashes(obj, out: set[str]):
    attr_name = (
        "attributes" if hasattr(obj, "attributes") else "nested_options" if hasattr(obj, "nested_options") else None
    )
    if not attr_name:
        return

    options = getattr(obj, attr_name)
    if not options:
        return

    for attr in options:
        if not hasattr(attr, "options"):
            continue

        nested_options = getattr(attr, "options")
        if nested_options:
            out.add(attr.feature_node_hash)
            _traverse_options(nested_options, out)


def get_nested_radio_and_checklist_hashes(ontology: OntologyStructure) -> set[str]:
    """
    Gets the feature node hashes of all nested radio buttons and checklists in
    the ontology. Text attributes are ignored as they don't lend them self nicely
    for tagging.

    Args:
        ontology: the ontology to be traversed

    Returns: a set of feature node hashes

    """
    option_answer_hashes: set[str] = set()

    for ont_obj in chain(ontology.objects, ontology.classifications):
        _get_nested_attribute_hashes(ont_obj, option_answer_hashes)

    return option_answer_hashes

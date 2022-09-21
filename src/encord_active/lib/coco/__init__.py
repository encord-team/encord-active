try:
    """Trying to import the coco encoder to see if dependencies are installed."""
    from encord_active.lib.coco.encoder import CocoEncoder as _CocoEncoder
except ModuleNotFoundError:
    raise SystemExit(
        """
        Some dependencies were not found. Please ensure to install encord-active[coco] when using the COCO parsers/importerself.
        Addional information can be found here: https://encord-active-docs.web.app/installation#coco-extras
        """
    )

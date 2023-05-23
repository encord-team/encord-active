from encord_active.lib.db.prisma_init import generate_prisma_client, did_schema_change


def _regen():
    import importlib
    import logging
    logger = logging.getLogger("Prisma")
    try:
        import prisma
        import prisma.client

        if did_schema_change():
            generate_prisma_client()
            importlib.reload(prisma)
    except (RuntimeError, ImportError):
        generate_prisma_client()
        importlib.reload(prisma)  # pylint: disable=used-before-assignment

# Regenerate prisma DB
_regen()

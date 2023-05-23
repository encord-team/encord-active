from encord_active.lib.db.prisma_init import did_schema_change, generate_prisma_client


def _regen():
    import importlib
    import logging

    logger = logging.getLogger("Prisma")
    import sys

    try:
        import prisma
        import prisma.client

        if did_schema_change():
            generate_prisma_client()
            importlib.reload(prisma)
            # Exit with message instead, as attempt at dynamic reload doesn't seem to be working.
            logger.warning("Prisma db module is out of date, regenerating, please execute the command again")
            sys.exit(1)
    except (RuntimeError, ImportError):
        generate_prisma_client()
        importlib.reload(prisma)  # pylint: disable=used-before-assignment


# Regenerate prisma DB
_regen()

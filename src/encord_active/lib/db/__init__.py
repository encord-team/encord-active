from encord_active.lib.db.prisma_init import generate_prisma_client

try:
    import prisma
    import prisma.client
except (RuntimeError, ImportError):
    generate_prisma_client()
    from importlib import reload

    reload(prisma)  # pylint: disable=used-before-assignment

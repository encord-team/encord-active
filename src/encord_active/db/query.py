from sqlmodel import Session, select
from .models import P

def query_all_metrics(session: Session):
    select()

def query_image_sizes(session: Session):
    session.execute(
        """
        SELECT width, height, COUNT(*) as count FROM yolo
        GROUP BY width, height
        """
    )
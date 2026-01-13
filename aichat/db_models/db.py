from sqlmodel import create_engine, Session

DATABASE_URL = "sqlite:///./aichat.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

def get_session():
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.flush()
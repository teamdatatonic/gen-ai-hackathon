from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

PROJECT_ID = 'dt-gen-ai-hackathon-dev'
DATASET_ID = 'database_analytics_demo_v2'

class Database:
    def __init__(self, url: str, schema: str = None):
        print("creating db engine...")
        self.engine = self.create_engine(url)
        print("creating db session...")
        self.base = declarative_base()
        self.sessionmaker = sessionmaker(
            autocommit=True, autoflush=True, bind=self.engine
        )
        self.schema = schema

    def create_engine(self, url):
        return create_engine(url)

    @property
    def dialect(self) -> str:
        return self.engine.dialect.name

    def create_session(self):
        return self.sessionmaker()
    

class BigQueryDatabase(Database):
    def __init__(
        self,
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
    ):
        super().__init__(f"bigquery://{project_id}/{dataset_id}")
        self.schema = dataset_id
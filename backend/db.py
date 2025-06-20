from pymongo import MongoClient
from config import Config
import logging

logger = logging.getLogger(__name__)

class Database:
    _instance = None
    _client = None
    _db = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            try:
                self._client = MongoClient(Config.MONGODB_URI)
                self._db = self._client.get_default_database()
                logger.info("Connected to MongoDB successfully")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise
    
    @property
    def db(self):
        return self._db
    
    def get_collection(self, name):
        return self._db[name]
    
    def health_check(self):
        try:
            self._client.admin.command('ping')
            return True
        except Exception:
            return False

# Global database instance
db_instance = Database()
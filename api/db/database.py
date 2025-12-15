import os
from dotenv import load_dotenv
from supabase import create_client, Client
from core.config import settings

load_dotenv()
url = settings.DATABASE_URL
key = settings.DATABASE_KEY

supabase: Client = create_client(url, key)
from functools import cache, lru_cache

from knowledge.service.upload_service import UploadService

@cache
# @lru_cache
def get_upload_file_service():
    return UploadService()

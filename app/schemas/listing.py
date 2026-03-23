from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

class ListQuery(BaseModel):
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)
    search: Optional[str] = None
    sort_by: str = Field(default="id")
    sort_dir: str = Field(default="desc")
    model_config = ConfigDict(extra="ignore")
    
class NameListQuery(ListQuery):
    sort_by: str = Field(default="name")
    sort_dir: str = Field(default="asc")

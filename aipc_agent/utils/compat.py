from __future__ import annotations
from typing import Any, Dict, Type
import pydantic
from pydantic import BaseModel

PYDANTIC_V2 = pydantic.VERSION.startswith("2.")

def dictify(data: "BaseModel", **kwargs) -> Dict[str, Any]:
    try:
        return data.model_dump(**kwargs)
    except AttributeError:
        return data.dict(**kwargs)
    
def jsonify(data: "BaseModel", **kwargs) -> str:
    try:
        return data.model_dump_json(**kwargs)
    except AttributeError:
        return data.json(**kwargs)
    
def model_validate(data: Type["BaseModel"], obj: Any) -> "BaseModel":
    try:
        return data.model_validate(obj)
    except AttributeError:
        return data.parse_obj(obj)
    
def disable_warnings(model: Type["BaseModel"]):
    if PYDANTIC_V2:
        model.model_config["protected_namespaces"] = ()
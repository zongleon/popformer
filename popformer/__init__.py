import os
from jaxtyping import jaxtyped

if os.getenv("TYPECHECKING"):
    from beartype import beartype as typechecker
else:
    typechecker = None

def typed(f):
    return jaxtyped(typechecker=typechecker)(f)
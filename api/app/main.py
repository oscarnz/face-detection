from fastapi import FastAPI
from .routers import facemask, ppe

app = FastAPI()

app.include_router(facemask.router)
app.include_router(ppe.router)

@app.get("/") 
def home(): 
    """just a string output

    Returns:
        string: output /docs to show that the API loads
    """
    return "/docs"

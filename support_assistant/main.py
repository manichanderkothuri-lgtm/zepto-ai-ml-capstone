from fastapi import FastAPI
from pydantic import BaseModel
from app import ask
app=FastAPI(title='Zepto Policy Support Assistant')
class AskRequest(BaseModel): query:str
class AskResponse(BaseModel): answer:str; sources:list[str]; confidence:float
@app.post('/ask',response_model=AskResponse)
def ask_endpoint(req:AskRequest): return ask(req.query)

from fastapi import FastAPI, UploadFile, File, HTTPException
from astropy.io import fits
import io
import numpy as np
from skimage import restoration

from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
engine = create_engine('sqlite:///labels.db')

class Label(Base):
    __tablename__ = 'labels'
    id = Column(Integer, primary_key=True)
    image_id = Column(String)
    x = Column(Float)
    y = Column(Float)
    label = Column(String)

Base.metadata.create_all(engine)


app = FastAPI()

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    content = await file.read()
    with fits.open(io.BytesIO(content)) as hdul:
        data = hdul[0].data
        # Enhance (e.g., remove noise)
        enhanced = restoration.denoise_tv_chambolle(data, weight=0.1)
    return {"data": enhanced.tolist()[:1000]}  # Limit for demo

@app.get("/query-data")
async def query_data(coord_x: float, coord_y: float):
    # Mock NASA API call
    return {"url": f"https://example.com/image?x={coord_x}&y={coord_y}"}




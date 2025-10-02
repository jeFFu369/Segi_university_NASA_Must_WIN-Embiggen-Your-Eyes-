from fastapi import FastAPI, HTTPException, Depends, Query, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from astropy.io import fits
import io
import numpy as np
from skimage import restoration, transform
import os
import uuid
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database engine and session
DATABASE_URL = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'nasa_images.db')}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Data models
class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    type = Column(String)  # e.g., Earth, Mars, Galaxy, Moon
    source = Column(String)  # e.g., Hubble, MRO, LRO
    files = relationship("ImageFile", back_populates="dataset", cascade="all, delete-orphan")
    labels = relationship("Label", back_populates="dataset", cascade="all, delete-orphan")

class ImageFile(Base):
    __tablename__ = "image_files"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    filename = Column(String)
    file_path = Column(String)
    width = Column(Integer)
    height = Column(Integer)
    file_size = Column(Integer)  # in bytes
    image_type = Column(String)  # e.g., fits, png, jpeg
    is_base_layer = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    dataset = relationship("Dataset", back_populates="files")
    # For pyramid structure image storage
    pyramid_levels = relationship("PyramidLevel", back_populates="image_file", cascade="all, delete-orphan")

class PyramidLevel(Base):
    __tablename__ = "pyramid_levels"
    id = Column(Integer, primary_key=True, index=True)
    image_file_id = Column(Integer, ForeignKey("image_files.id"))
    level = Column(Integer)  # 0 = full resolution, 1 = half, etc.
    width = Column(Integer)
    height = Column(Integer)
    tile_width = Column(Integer, default=256)
    tile_height = Column(Integer, default=256)
    file_path = Column(String)  # Path to store pyramid level
    image_file = relationship("ImageFile", back_populates="pyramid_levels")

class Label(Base):
    __tablename__ = "labels"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    user_id = Column(String)  # Can be username or anonymous ID
    x = Column(Float)
    y = Column(Float)
    width = Column(Float, nullable=True)
    height = Column(Float, nullable=True)
    label = Column(String)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # Support for polygon labels
    polygon_points = Column(Text, nullable=True)  # List of points in JSON format
    dataset = relationship("Dataset", back_populates="labels")

# Create database tables
Base.metadata.create_all(bind=engine)

# Dependency: Get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create FastAPI application
app = FastAPI(title="NASA Space App Challenge - Embiggen Your Eyes", 
              description="A platform for exploring massive NASA image datasets")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "..", "frontend", "templates"))

# Serve static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend", "static")), name="static")

app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"],)

# Create necessary directories
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
PYRAMID_DIR = os.path.join(os.path.dirname(__file__), "pyramids")
for dir_path in [UPLOAD_DIR, PYRAMID_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Helper function: Create image pyramid
def create_image_pyramid(image_data, base_filename, tile_size=256):
    """Create pyramid structure for large images to support efficient zooming"""
    pyramid_files = []
    current_image = image_data
    level = 0
    
    # Ensure image is 2D or 3D (if color)
    if len(current_image.shape) > 3:
        current_image = current_image[0]  # Take the first channel
    
    while True:
        # Create directory for current level
        level_dir = os.path.join(PYRAMID_DIR, f"{base_filename}_level_{level}")
        if not os.path.exists(level_dir):
            os.makedirs(level_dir)
        
        height, width = current_image.shape[:2]
        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size
        
        # Save current level information
        pyramid_info = {
            "level": level,
            "width": width,
            "height": height,
            "tile_width": tile_size,
            "tile_height": tile_size,
            "tiles_x": tiles_x,
            "tiles_y": tiles_y,
            "tiles": []
        }
        
        # Create and save all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                y_start = y * tile_size
                y_end = min((y + 1) * tile_size, height)
                x_start = x * tile_size
                x_end = min((x + 1) * tile_size, width)
                
                tile = current_image[y_start:y_end, x_start:x_end]
                tile_filename = f"tile_{x}_{y}.npy"
                tile_path = os.path.join(level_dir, tile_filename)
                np.save(tile_path, tile)
                
                pyramid_info["tiles"].append({
                    "x": x,
                    "y": y,
                    "filename": tile_filename
                })
        
        # Save level information
        info_path = os.path.join(level_dir, "info.json")
        with open(info_path, "w") as f:
            json.dump(pyramid_info, f)
        
        pyramid_files.append({
            "level": level,
            "path": level_dir,
            "info": pyramid_info
        })
        
        # Check if we need to continue creating smaller levels
        if width <= tile_size and height <= tile_size:
            break
        
        # Downscale the image
        current_image = transform.rescale(current_image, 0.5, anti_aliasing=True)
        level += 1
    
    return pyramid_files

# API endpoints
@app.get("/datasets/")
async def list_datasets(db: SessionLocal = Depends(get_db)):
    """List all datasets"""
    datasets = db.query(Dataset).all()
    return datasets

@app.post("/datasets/")
async def create_dataset(
    name: str = Query(...),
    description: str = Query(...),
    type: str = Query(...),
    source: str = Query(...),
    db: SessionLocal = Depends(get_db)
):
    """Create a new dataset"""
    dataset = Dataset(
        name=name,
        description=description,
        type=type,
        source=source
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset

@app.post("/datasets/{dataset_id}/upload-image")
async def upload_image_to_dataset(
    dataset_id: int,
    file: UploadFile = File(...),
    is_base_layer: bool = Query(False),
    db: SessionLocal = Depends(get_db)
):
    """Upload an image file to a specific dataset"""
    # Check if dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Read file content
        content = await file.read()
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Process FITS file
        if file_extension == ".fits":
            with fits.open(io.BytesIO(content)) as hdul:
                data = hdul[0].data
                height, width = data.shape[:2]
                
                # Create image pyramid to support zooming
                pyramid_base_filename = os.path.splitext(unique_filename)[0]
                pyramid_files = create_image_pyramid(data, pyramid_base_filename)
                
                # Save image information to database
                image_file = ImageFile(
                    dataset_id=dataset_id,
                    filename=file.filename,
                    file_path=file_path,
                    width=width,
                    height=height,
                    file_size=len(content),
                    image_type="fits",
                    is_base_layer=is_base_layer
                )
                db.add(image_file)
                db.commit()
                db.refresh(image_file)
                
                # Save pyramid level information
                for p in pyramid_files:
                    pyramid_level = PyramidLevel(
                        image_file_id=image_file.id,
                        level=p["level"],
                        width=p["info"]["width"],
                        height=p["info"]["height"],
                        tile_width=p["info"]["tile_width"],
                        tile_height=p["info"]["tile_height"],
                        file_path=p["path"]
                    )
                    db.add(pyramid_level)
                db.commit()
        else:
            # Process other image formats (simplified version)
            # In a real application, appropriate libraries should be used to read other image formats
            image_file = ImageFile(
                dataset_id=dataset_id,
                filename=file.filename,
                file_path=file_path,
                width=0,  # Placeholder
                height=0,  # Placeholder
                file_size=len(content),
                image_type=file_extension[1:],  # Remove dot
                is_base_layer=is_base_layer
            )
            db.add(image_file)
            db.commit()
            db.refresh(image_file)
        
        return {"message": "Image uploaded successfully", "image_id": image_file.id}
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")

@app.get("/datasets/{dataset_id}/images/")
async def list_dataset_images(
    dataset_id: int,
    db: SessionLocal = Depends(get_db)
):
    """List all images in a specific dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return dataset.files

@app.get("/images/{image_id}/pyramid-info")
async def get_image_pyramid_info(
    image_id: int,
    db: SessionLocal = Depends(get_db)
):
    """Get image pyramid information for frontend zooming"""
    image_file = db.query(ImageFile).filter(ImageFile.id == image_id).first()
    if not image_file:
        raise HTTPException(status_code=404, detail="Image not found")
    
    pyramid_levels = db.query(PyramidLevel).filter(PyramidLevel.image_file_id == image_id).all()
    
    return {
        "image_id": image_id,
        "original_width": image_file.width,
        "original_height": image_file.height,
        "levels": [{
            "level": pl.level,
            "width": pl.width,
            "height": pl.height,
            "tile_width": pl.tile_width,
            "tile_height": pl.tile_height
        } for pl in pyramid_levels]
    }

@app.get("/images/{image_id}/tile/{level}/{x}/{y}")
async def get_image_tile(
    image_id: int,
    level: int,
    x: int,
    y: int,
    db: SessionLocal = Depends(get_db)
):
    """Get a specific image tile for frontend display"""
    image_file = db.query(ImageFile).filter(ImageFile.id == image_id).first()
    if not image_file:
        raise HTTPException(status_code=404, detail="Image not found")
    
    pyramid_level = db.query(PyramidLevel).filter(
        PyramidLevel.image_file_id == image_id, 
        PyramidLevel.level == level
    ).first()
    
    if not pyramid_level:
        raise HTTPException(status_code=404, detail="Pyramid level not found")
    
    # Read tile information
    base_filename = os.path.splitext(os.path.basename(image_file.file_path))[0]
    level_dir = os.path.join(PYRAMID_DIR, f"{base_filename}_level_{level}")
    
    try:
        # Read level information
        with open(os.path.join(level_dir, "info.json"), "r") as f:
            level_info = json.load(f)
        
        # Check if requested tile exists
        tile_found = False
        for tile in level_info["tiles"]:
            if tile["x"] == x and tile["y"] == y:
                tile_path = os.path.join(level_dir, tile["filename"])
                if os.path.exists(tile_path):
                    # Load tile data
                    tile_data = np.load(tile_path)
                    # Convert NumPy array to PNG for return
                    if len(tile_data.shape) == 2:
                        # Grayscale image
                        pil_image = Image.fromarray((tile_data * 255).astype(np.uint8), mode='L')
                    else:
                        # Color image
                        pil_image = Image.fromarray((tile_data * 255).astype(np.uint8))
                    
                    # Save to bytes
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    return StreamingResponse(img_byte_arr, media_type="image/png")
                else:
                    raise HTTPException(status_code=404, detail="Tile file not found")
        
        raise HTTPException(status_code=404, detail="Tile not found")
    except Exception as e:
        logger.error(f"Error reading tile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading tile: {str(e)}")

@app.post("/datasets/{dataset_id}/labels/")
async def create_label(
    dataset_id: int,
    x: float = Query(...),
    y: float = Query(...),
    label: str = Query(...),
    user_id: str = Query(...),
    width: Optional[float] = Query(None),
    height: Optional[float] = Query(None),
    description: Optional[str] = Query(None),
    polygon_points: Optional[str] = Query(None),
    db: SessionLocal = Depends(get_db)
):
    """Create a new label in a dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    new_label = Label(
        dataset_id=dataset_id,
        user_id=user_id,
        x=x,
        y=y,
        width=width,
        height=height,
        label=label,
        description=description,
        polygon_points=polygon_points
    )
    
    db.add(new_label)
    db.commit()
    db.refresh(new_label)
    
    return new_label

@app.get("/datasets/{dataset_id}/labels/")
async def get_dataset_labels(
    dataset_id: int,
    x_min: Optional[float] = Query(None),
    y_min: Optional[float] = Query(None),
    x_max: Optional[float] = Query(None),
    y_max: Optional[float] = Query(None),
    label_type: Optional[str] = Query(None),
    db: SessionLocal = Depends(get_db)
):
    """Get labels in a dataset, supporting filtering by region and type"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    query = db.query(Label).filter(Label.dataset_id == dataset_id)
    
    # Filter by region
    if x_min is not None and y_min is not None and x_max is not None and y_max is not None:
        # Simplified region filtering, more complex geometric calculations may be needed in a real application
        query = query.filter(
            Label.x.between(x_min, x_max),
            Label.y.between(y_min, y_max)
        )
    
    # Filter by label type
    if label_type:
        query = query.filter(Label.label == label_type)
    
    labels = query.all()
    return labels

@app.post("/process-image/enhance")
async def enhance_image(
    image_id: int = Query(...),
    method: str = Query("denoise"),
    strength: float = Query(0.1, ge=0.0, le=1.0),
    db: SessionLocal = Depends(get_db)
):
    """Apply image enhancement techniques"""
    # Get image file
    image_file = db.query(ImageFile).filter(ImageFile.id == image_id).first()
    if not image_file:
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        # For demonstration, we'll just return a success message
        # In a real implementation, this would apply the enhancement
        return {
            "message": f"Applied {method} enhancement with strength {strength}",
            "image_id": image_id
        }
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error enhancing image: {str(e)}")

@app.get("/search-features")
async def search_features(
    query: str = Query(...),
    dataset_id: Optional[int] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    db: SessionLocal = Depends(get_db)
):
    """Search for features in images (simplified version, may require AI support in a real application)"""
    # This is a simple implementation that matches text in labels
    # In a real application, AI models could be integrated for more complex image feature search
    
    search_query = f"%{query.lower()}%"
    
    if dataset_id:
        # Check if dataset exists
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        labels = db.query(Label).filter(
            Label.dataset_id == dataset_id,
            (Label.label.ilike(search_query) | Label.description.ilike(search_query))
        ).limit(limit).all()
    else:
        labels = db.query(Label).filter(
            (Label.label.ilike(search_query) | Label.description.ilike(search_query))
        ).limit(limit).all()
    
    return labels

@app.get("/")
async def root(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

# Add sample data if database is empty
@app.on_event("startup")
async def add_sample_data():
    """Add sample data if database is empty"""
    db = SessionLocal()
    try:
        # Check if we have datasets
        dataset_count = db.query(Dataset).count()
        if dataset_count == 0:
            # Add sample datasets
            datasets = [
                {
                    "name": "Earth Surface",
                    "description": "Earth Surface",
                    "type": "Earth",
                    "source": "https://worldview.earthdata.nasa.gov/?v=-361.7483658641388,-153.78063459004,279.8834636730729,161.5687683188239&l=Reference_Labels_15m(hidden),Reference_Features_15m,Coastlines_15m,OCI_PACE_True_Color,VIIRS_NOAA21_CorrectedReflectance_TrueColor,VIIRS_NOAA20_CorrectedReflectance_TrueColor,VIIRS_SNPP_CorrectedReflectance_TrueColor(hidden),MODIS_Aqua_CorrectedReflectance_TrueColor(hidden),MODIS_Terra_CorrectedReflectance_TrueColor(hidden)&lg=true&t=2025-09-29-T18%3A00%3A42Z"
                },
                {
                    "name": "Mars Surface",
                    "description": "Mars Surface",
                    "type": "Mars",
                    "source": "https://science.nasa.gov/asset/hubble/mars-projection-map/"
                },
                {
                    "name": "Galaxy Surface",
                    "description": "Galaxy Surface",
                    "type": "Galaxy",
                    "source": "https://science.nasa.gov/mission/hubble/science/explore-the-night-sky/hubble-messier-catalog/messier-31/#:~:text=In%20January%20of%202025%2C%20NASA&#x27;s,were%20challenging%20to%20stitch%20together."
                }
            ]
            
            for ds_data in datasets:
                dataset = Dataset(**ds_data)
                db.add(dataset)
            
            db.commit()
            
        # Check if we have labels
        label_count = db.query(Label).count()
        if label_count == 0:
            # Add sample labels for each dataset
            sample_labels = [
                # Earth labels
                {"dataset_id": 1, "user_id": "user1", "x": 100.0, "y": 150.0, "label": "Ocean", "description": "Large body of water"},
                {"dataset_id": 1, "user_id": "user1", "x": 300.0, "y": 200.0, "label": "Mountain", "description": "Mountain range"},
                # Mars labels
                {"dataset_id": 2, "user_id": "user1", "x": 250.0, "y": 180.0, "label": "Crater", "description": "Impact crater"},
                {"dataset_id": 2, "user_id": "user1", "x": 400.0, "y": 300.0, "label": "Valley", "description": "Canyon system"},
                # Galaxy labels
                {"dataset_id": 3, "user_id": "user1", "x": 500.0, "y": 400.0, "label": "Spiral Arm", "description": "Galaxy spiral structure"},
                {"dataset_id": 3, "user_id": "user1", "x": 750.0, "y": 600.0, "label": "Star Cluster", "description": "Dense star cluster"}
            ]
            
            for label_data in sample_labels:
                label = Label(**label_data)
                db.add(label)
            
            db.commit()
            
        # Check if we have pyramid levels
        pyramid_count = db.query(PyramidLevel).count()
        if pyramid_count == 0:
            # We'll add some sample pyramid levels for existing images
            images = db.query(ImageFile).all()
            for image in images:
                # Add sample pyramid levels for each image
                for level in range(3):  # 3 levels for demo
                    pyramid_level = PyramidLevel(
                        image_file_id=image.id,
                        level=level,
                        width=max(100, image.width // (2**level)) if image.width > 0 else 1000 // (2**level),
                        height=max(100, image.height // (2**level)) if image.height > 0 else 1000 // (2**level),
                        tile_width=256,
                        tile_height=256,
                        file_path=f"pyramids/sample_level_{level}"
                    )
                    db.add(pyramid_level)
            
            db.commit()
            
    except Exception as e:
        logger.error(f"Error adding sample data: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




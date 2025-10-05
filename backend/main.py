from fastapi import FastAPI, HTTPException, Depends, Query, File, UploadFile, Request, Body
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from astropy.io import fits
import io
import numpy as np
from skimage import restoration, transform, img_as_ubyte
import os
import uuid
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, Text, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
from typing import List, Dict

# Create FastAPI application
app = FastAPI(title="NASA Space App Challenge - Embiggen Your Eyes", 
              description="A platform for exploring massive NASA image datasets")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "..", "frontend", "templates"))

# Serve static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend", "static")), name="static")

# Serve uploads directory for direct image access
app.mount("/backend/uploads", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "uploads")), name="uploads")

app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"],)

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

                # Convert to uint8 before saving
                if tile.dtype != np.uint8:
                    tile = img_as_ubyte(tile)


                tile_filename = f"tile_{x}_{y}.png"
                tile_path = os.path.join(level_dir, tile_filename)
                
                io.imsave(tile_path, tile)
                
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
            # Process other image formats properly
            try:
                # Read image dimensions using PIL
                from PIL import Image
                import numpy as np
                
                with Image.open(io.BytesIO(content)) as img:
                    width, height = img.size
                    
                    # Convert PIL image to numpy array for pyramid creation
                    img_array = np.array(img)
                    
                    # Create image pyramid to support zooming
                    pyramid_base_filename = os.path.splitext(unique_filename)[0]
                    pyramid_files = create_image_pyramid(img_array, pyramid_base_filename)
                    
                    # Save image information to database
                    image_file = ImageFile(
                        dataset_id=dataset_id,
                        filename=file.filename,
                        file_path=file_path,
                        width=width,
                        height=height,
                        file_size=len(content),
                        image_type=file_extension[1:],  # Remove dot
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
            except Exception as e:
                logger.warning(f"Failed to process image file {file.filename}: {str(e)}")
                # Fallback in case of processing errors
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
                    
                    pil_image = Image.open(tile_path)
                    
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

@app.delete("/datasets/{dataset_id}/labels/", status_code=204)
async def delete_dataset_labels(
    dataset_id: int,
    db: SessionLocal = Depends(get_db)
):
    """Delete all labels in a dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    deleted_count = db.query(Label).filter(Label.dataset_id == dataset_id).delete()
    db.commit()

    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="No labels found to delete")

    return {"message": f"{deleted_count} labels deleted successfully"}

@app.post("/datasets/{dataset_id}/review-ml-annotations")
async def review_ml_annotations(
    dataset_id: int,
    accepted: List[dict] = Body(...),
    rejected: List[dict] = Body(...),
    db: SessionLocal = Depends(get_db)
):
    """Process the review of ML-generated annotations"""
    # Get the dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    # Delete rejected ML annotations
    if rejected:
        rejected_ids = [anno["id"] for anno in rejected if "id" in anno]
        if rejected_ids:
            db.query(Label).filter(
                Label.id.in_(rejected_ids),
                Label.user_id == "ml_analysis"
            ).delete(synchronize_session=False)
            db.commit()
    
    # Update accepted annotations that were modified
    updated_count = 0
    for anno in accepted:
        if "id" in anno and ("label" in anno or "description" in anno):
            label = db.query(Label).filter(
                Label.id == anno["id"],
                Label.user_id == "ml_analysis"
            ).first()
            if label:
                if "label" in anno:
                    label.label = anno["label"]
                if "description" in anno:
                    label.description = anno["description"]
                updated_count += 1
    
    if updated_count > 0:
        db.commit()
    
    return {
        "message": f"ML annotation review processed successfully",
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "updated_count": updated_count
    }

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

def generate_pyramid_levels(file_path, image_id):
    # Placeholder for pyramid generation - implement based on your needs
    try:
        with Image.open(file_path) as img:
            levels = []
            current_img = img.copy()
            level = 0
            while current_img.width > 256 or current_img.height > 256:
                level_path = f"pyramids/{image_id}_level_{level}.jpg"
                os.makedirs(os.path.dirname(level_path), exist_ok=True)
                current_img.save(level_path)
                levels.append({
                    "level": level,
                    "info": {"width": current_img.width, "height": current_img.height, "tile_width": 256, "tile_height": 256},
                    "path": level_path
                })
                current_img = current_img.resize((current_img.width // 2, current_img.height // 2))
                level += 1
            return levels
    except Exception as e:
        raise ValueError(f"Pyramid generation failed: {str(e)}")

# Enhanced fix_existing_images endpoint
@app.post("/fix-existing-images")
def fix_existing_images(db: Session = Depends(get_db)):
    """
    Fix existing images by updating dimensions, file sizes, and generating pyramid levels.
    Processes images with width or height equal to 0.
    """
    # Temporarily disable decompression bomb limit
    original_max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None

    try:
        images_to_fix = db.query(ImageFile).filter(
            (ImageFile.width == 0) | (ImageFile.height == 0)
        ).all()
        total_images = len(images_to_fix)
        fixed_count = 0
        error_count = 0
        errors = []

        logger.info(f"Starting to fix {total_images} images.")

        # Process each image sequentially
        for image_file in images_to_fix:
            logger.info(f"Processing image {image_file.filename} (ID: {image_file.id})")
            try:
                # Construct full file path
                file_path = os.path.join(os.path.dirname(__file__), image_file.file_path)
                if not os.path.exists(file_path):
                    error_msg = f"File not found: {file_path}"
                    errors.append(error_msg)
                    error_count += 1
                    logger.error(error_msg)
                    continue

                # Open and get image dimensions
                with Image.open(file_path) as img:
                    width, height = img.size
                    file_size = os.path.getsize(file_path)
                    image_type = img.format.lower() if img.format else "unknown"

                # Generate pyramid levels (assuming this function is defined)
                pyramid_files = generate_pyramid_levels(file_path, image_file.id)

                # Update database
                db.execute(
                    text("UPDATE image_files SET width=:width, height=:height, file_size=:file_size, image_type=:image_type WHERE id=:id"),
                    {
                        "width": width,
                        "height": height,
                        "file_size": file_size,
                        "image_type": image_type,
                        "id": image_file.id
                    }
                )

                # Delete existing pyramid levels
                db.execute(
                    text("DELETE FROM pyramid_levels WHERE image_file_id=:image_file_id"),
                    {"image_file_id": image_file.id}
                )

                # Insert new pyramid levels
                for p in pyramid_files:
                    db.execute(
                        text("INSERT INTO pyramid_levels (image_file_id, level, width, height, tile_width, tile_height, file_path) VALUES (:image_file_id, :level, :width, :height, :tile_width, :tile_height, :file_path)"),
                        {
                            "image_file_id": image_file.id,
                            "level": p["level"],
                            "width": p["info"]["width"],
                            "height": p["info"]["height"],
                            "tile_width": p["info"]["tile_width"],
                            "tile_height": p["info"]["tile_height"],
                            "file_path": p["path"]
                        }
                    )

                db.commit()  # Commit after all updates for this image

                fixed_count += 1
                logger.info(f"Successfully fixed image {image_file.filename} (ID: {image_file.id})")

            except Exception as e:
                db.rollback()  # Rollback on error for this image
                error_msg = f"Error processing image {image_file.filename} (ID: {image_file.id}): {str(e)}"
                errors.append(error_msg)
                error_count += 1
                logger.error(error_msg)

        # Prepare response
        response = {
            "total_images_to_fix": total_images,
            "fixed_count": fixed_count,
            "error_count": error_count,
            "errors": errors
        }
        logger.info(f"Fix process completed: {response}")
        return response

    except Exception as e:
        db.rollback()
        logger.error(f"Error in fix_existing_images endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fixing images: {str(e)}")
    finally:
        # Restore original decompression bomb limit
        Image.MAX_IMAGE_PIXELS = original_max_pixels

# Add this import for the ML analysis functions
import random
from skimage import io as skimage_io, color, img_as_ubyte

# Add endpoint to fix pyramid structure for a specific image
@app.post("/fix-image-pyramid/{image_id}")
def fix_image_pyramid(image_id: int, db: Session = Depends(get_db)):
    """
    Fix pyramid structure for a specific image
    """
    try:
        # Get image file
        image_file = db.query(ImageFile).filter(ImageFile.id == image_id).first()
        if not image_file:
            raise HTTPException(status_code=404, detail="Image not found")
        
        logger.info(f"Fixing pyramid for image {image_file.filename} (ID: {image_id})")
        
        # Construct full file path
        file_path = os.path.join(os.path.dirname(__file__), image_file.file_path)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Open image and get dimensions
        with Image.open(file_path) as img:
            width, height = img.size
            
            # Convert PIL image to numpy array for pyramid creation
            img_array = np.array(img)
            
            # Extract base filename (without extension)
            base_filename = os.path.splitext(os.path.basename(image_file.file_path))[0]
            
            # Create image pyramid
            pyramid_files = create_image_pyramid(img_array, base_filename)
            
            # Delete existing pyramid levels
            db.query(PyramidLevel).filter(PyramidLevel.image_file_id == image_id).delete()
            
            # Save new pyramid level information
            for p in pyramid_files:
                pyramid_level = PyramidLevel(
                    image_file_id=image_id,
                    level=p["level"],
                    width=p["info"]["width"],
                    height=p["info"]["height"],
                    tile_width=p["info"]["tile_width"],
                    tile_height=p["info"]["tile_height"],
                    file_path=p["path"]
                )
                db.add(pyramid_level)
            
            db.commit()
            
            logger.info(f"Successfully fixed pyramid for image {image_file.filename} (ID: {image_id})")
            
            return {
                "message": f"Successfully fixed pyramid for image {image_file.filename}",
                "image_id": image_id,
                "levels_created": len(pyramid_files)
            }
    except Exception as e:
        db.rollback()
        logger.error(f"Error fixing pyramid for image ID {image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fixing pyramid: {str(e)}")

@app.post("/datasets/{dataset_id}/analyze-ml")
async def analyze_dataset_with_ml(
    dataset_id: int,
    model_type: Optional[str] = Query(None),  # Optional model type override
    db: SessionLocal = Depends(get_db)
):
    """Run ML analysis on a dataset and generate candidate annotations"""
    # Get the dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Determine which ML model to use based on dataset type or user selection
    if model_type:
        selected_model = model_type.lower()
    else:
        # Auto-select model based on dataset type
        if dataset.type and "earth" in dataset.type.lower():
            selected_model = "faster_rcnn"
        elif dataset.type and "mars" in dataset.type.lower():
            selected_model = "deep_source_finder"
        elif dataset.type and "galaxy" in dataset.type.lower():
            selected_model = "astronet"
        else:
            # Default to faster_rcnn for other types
            selected_model = "faster_rcnn"
    
    # Get images in the dataset
    images = db.query(ImageFile).filter(ImageFile.dataset_id == dataset_id).all()
    if not images:
        return {"success": False, "message": "No images found in dataset"}
    
    # Generate annotations based on the selected model
    annotations = []
    try:
        if selected_model == "faster_rcnn":
            annotations = simulate_faster_rcnn_detection(dataset_id, images, db)
        elif selected_model == "deep_source_finder":
            annotations = simulate_deep_source_finder(dataset_id, images, db)
        elif selected_model == "astronet":
            annotations = simulate_astronet_detection(dataset_id, images, db)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        # Save annotations to database with special user_id
        saved_annotations = []
        label_objects = []
        for annotation in annotations:
            new_label = Label(
                dataset_id=dataset_id,
                user_id="ml_analysis",  # Special user ID to indicate ML-generated
                x=annotation["x"],
                y=annotation["y"],
                width=annotation.get("width"),
                height=annotation.get("height"),
                label=annotation["label"],
                description=annotation.get("description")
            )
            db.add(new_label)
            label_objects.append(new_label)
        
        db.commit()
        
        # Create response annotations with database-generated IDs
        for i, label in enumerate(label_objects):
            annotation_with_id = {
                **annotations[i],
                "id": label.id  # Include the database-generated ID
            }
            saved_annotations.append(annotation_with_id)
        
        return {
            "success": True,
            "message": f"ML analysis completed with {selected_model}",
            "model_used": selected_model,
            "dataset_type": dataset.type,
            "annotations": saved_annotations
        }
    except Exception as e:
        logger.error(f"ML analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ML analysis failed: {str(e)}")

# Helper functions for simulating ML model detections
def simulate_faster_rcnn_detection(dataset_id, images, db):
    """Simulate object detection using Faster R-CNN"""
    # In a real implementation, this would use a pre-trained Faster R-CNN model
    detected_labels = []
    
    # Generate random detections based on dataset type
    for image_file in images:
        # Get dataset information
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        dataset_type = dataset.type.lower() if dataset and dataset.type else "unknown"
        
        # Adjust number of detections based on dataset type
        if "galaxy" in dataset_type:
            labels = ['Galaxy', 'Star', 'Nebula']
            descriptions = [
                'Spiral galaxy with distinct arms',
                'Bright star with possible planetary system',
                'Diffuse nebula with star formation'
            ]
        elif "mars" in dataset_type:
            labels = ['Crater', 'Mountain', 'Valley']
            descriptions = [
                'Impact crater with ejecta blanket',
                'Volcanic mountain formation',
                'Ancient river valley system'
            ]
        elif "earth" in dataset_type:
            labels = ['Cloud formation', 'Coastal feature', 'Mountain range']
            descriptions = [
                'Cumulus cloud formation',
                'Complex coastal geomorphology',
                'Mountain range with snow cover'
            ]
        else:
            labels = ['Feature', 'Region', 'Anomaly']
            descriptions = [
                'Interesting astronomical feature',
                'Notable region of interest',
                'Unusual or unexpected anomaly'
            ]
        
        # Generate random detections for this image
        np.random.seed(image_file.id)  # Use image ID as seed for consistent results
        num_detections = 3  # Default number of detections per image
        
        for i in range(num_detections):
            # Get width and height from image_file or use defaults if not available
            width = getattr(image_file, 'width', 1000)
            height = getattr(image_file, 'height', 1000)
            
            # Ensure width and height are reasonable values
            if width <= 0 or height <= 0:
                width, height = 1000, 1000
            
            # Generate random position and size
            box_width = max(50, int(width * np.random.uniform(0.05, 0.2)))
            box_height = max(50, int(height * np.random.uniform(0.05, 0.2)))
            
            x = np.random.uniform(box_width/2, width - box_width/2)
            y = np.random.uniform(box_height/2, height - box_height/2)
            
            # Select a label and description
            label_idx = i % len(labels)
            label = labels[label_idx]
            description = descriptions[label_idx]
            
            # Generate a reasonable confidence value between 0.7 and 0.95
            confidence = np.random.uniform(0.7, 0.95)
            detected_labels.append({
                "x": x,
                "y": y,
                "width": box_width,
                "height": box_height,
                "label": label,
                "description": description,
                "confidence": confidence
            })
    
    return detected_labels

def simulate_deep_source_finder(dataset_id, images, db):
    """Simulate source finding using Deep Source Finder"""
    # In a real implementation, this would use a specialized deep learning model for astronomical source detection
    detected_labels = []
    
    # Generate more point sources rather than bounding boxes
    for image_file in images:
        # Get dataset information
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        dataset_type = dataset.type.lower() if dataset and dataset.type else "unknown"
        
        # Get width and height from image_file or use defaults if not available
        width = getattr(image_file, 'width', 1000)
        height = getattr(image_file, 'height', 1000)
        
        # Ensure width and height are reasonable values
        if width <= 0 or height <= 0:
            width, height = 1000, 1000
        
        # Generate random point sources
        np.random.seed(image_file.id + 1)  # Use image ID + 1 as seed for consistent but different results
        num_sources = 5  # Default number of sources per image
        
        for i in range(num_sources):
            # Generate random position
            x = np.random.uniform(0, width)
            y = np.random.uniform(0, height)
            
            # Based on dataset type, assign different probabilities to different source types
            if "galaxy" in dataset_type:
                # More galaxies and stars
                p = np.random.random()
                if p < 0.6:
                    label = 'Galaxy'
                    description = 'Potential galaxy source detected'
                elif p < 0.9:
                    label = 'Star'
                    description = 'Point source likely a star'
                else:
                    label = 'Quasar'
                    description = 'Compact source with high redshift probability'
            elif "mars" in dataset_type:
                # More geological features
                p = np.random.random()
                if p < 0.5:
                    label = 'Impact site'
                    description = 'Possible impact crater candidate'
                elif p < 0.8:
                    label = 'Geological feature'
                    description = 'Notable geological formation'
                else:
                    label = 'Anomaly'
                    description = 'Unexplained surface feature'
            elif "earth" in dataset_type:
                # More Earth observation features
                p = np.random.random()
                if p < 0.4:
                    label = 'Urban area'
                    description = 'Potential urban development'
                elif p < 0.7:
                    label = 'Water body'
                    description = 'Possible water feature'
                else:
                    label = 'Vegetation'
                    description = 'Area with vegetation signature'
            else:
                label = 'Source'
                description = 'Astronomical source detected'
            
            # Generate a reasonable confidence value between 0.7 and 0.95
            confidence = np.random.uniform(0.7, 0.95)
            detected_labels.append({
                "x": x,
                "y": y,
                "label": label,
                "description": description,
                "confidence": confidence
            })
    
    return detected_labels

def simulate_astronet_detection(dataset_id, images, db):
    """Simulate astronomical object classification using AstroNet"""
    # In a real implementation, this would use a model like AstroNet for astronomical object classification
    detected_labels = []
    
    # Generate a mix of point sources and regions
    for image_file in images:
        # Get width and height from image_file or use defaults if not available
        width = getattr(image_file, 'width', 1000)
        height = getattr(image_file, 'height', 1000)
        
        # Ensure width and height are reasonable values
        if width <= 0 or height <= 0:
            width, height = 1000, 1000
        
        # Generate random astronomical objects
        np.random.seed(image_file.id + 2)  # Use image ID + 2 as seed for consistent but different results
        num_objects = 4  # Default number of objects per image
        
        # Define specific astronomical object types and descriptions
        astronomical_objects = [
            {
                "types": ['Spiral Galaxy', 'Elliptical Galaxy', 'Irregular Galaxy'],
                "descriptions": [
                    'Galaxy with spiral structure and arms',
                    'Smooth, featureless elliptical galaxy',
                    'Galaxy with irregular shape and structure'
                ]
            },
            {
                "types": ['Star', 'Binary Star', 'Variable Star'],
                "descriptions": [
                    'Main sequence star similar to our Sun',
                    'Two stars orbiting around a common center of mass',
                    'Star with varying brightness over time'
                ]
            },
            {
                "types": ['Nebula', 'Supernova Remnant', 'Planetary Nebula'],
                "descriptions": [
                    'Interstellar cloud of dust, hydrogen, and other ionized gases',
                    'Remnant of a massive star that exploded as a supernova',
                    'Expanding shell of gas ejected from a star at the end of its life'
                ]
            },
            {
                "types": ['Quasar', 'Active Galactic Nucleus', 'Gamma-Ray Burst'],
                "descriptions": [
                    'Extremely luminous active galactic nucleus powered by a supermassive black hole',
                    'Compact region at the center of a galaxy with intense energy output',
                    'Extremely energetic explosion believed to occur when a massive star collapses'
                ]
            }
        ]
        
        for i in range(num_objects):
            # Generate random position
            x = np.random.uniform(0, width)
            y = np.random.uniform(0, height)
            
            # For some objects, add a small region (width and height)
            add_region = np.random.random() > 0.5
            width_obj = int(width * 0.1) if add_region else None
            height_obj = int(height * 0.1) if add_region else None
            
            # Select an object category
            category_idx = i % len(astronomical_objects)
            category = astronomical_objects[category_idx]
            
            # Select a specific type from the category
            type_idx = np.random.randint(0, len(category["types"]))
            label = category["types"][type_idx]
            description = category["descriptions"][type_idx]
            
            # Generate a reasonable confidence value between 0.7 and 0.95
            confidence = np.random.uniform(0.7, 0.95)
            detected_labels.append({
                "x": x,
                "y": y,
                "width": width_obj,
                "height": height_obj,
                "label": label,
                "description": description,
                "confidence": confidence
            })
    
    return detected_labels

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
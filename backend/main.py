from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from fastapi.responses import JSONResponse, FileResponse
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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建数据库引擎和会话
DATABASE_URL = "sqlite:///nasa_images.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 数据模型
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
    # 对于金字塔结构的图像存储
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
    file_path = Column(String)  # 存储金字塔层级的路径
    image_file = relationship("ImageFile", back_populates="pyramid_levels")

class Label(Base):
    __tablename__ = "labels"
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    user_id = Column(String)  # 可以是用户名或匿名ID
    x = Column(Float)
    y = Column(Float)
    width = Column(Float, nullable=True)
    height = Column(Float, nullable=True)
    label = Column(String)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # 支持多边形标签
    polygon_points = Column(Text, nullable=True)  # JSON格式的点列表
    dataset = relationship("Dataset", back_populates="labels")

# 创建数据库表
Base.metadata.create_all(bind=engine)

# 依赖项：获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 创建FastAPI应用
app = FastAPI(title="NASA Space App Challenge - Embiggen Your Eyes", 
              description="A platform for exploring massive NASA image datasets")

# 创建必要的目录
UPLOAD_DIR = "uploads"
PYRAMID_DIR = "pyramids"
for dir_path in [UPLOAD_DIR, PYRAMID_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 辅助函数：创建图像金字塔
def create_image_pyramid(image_data, base_filename, tile_size=256):
    """为大型图像创建金字塔结构以支持高效缩放"""
    pyramid_files = []
    current_image = image_data
    level = 0
    
    # 确保图像是2D或3D（如果是彩色）
    if len(current_image.shape) > 3:
        current_image = current_image[0]  # 取第一个通道
    
    while True:
        # 为当前层级创建目录
        level_dir = os.path.join(PYRAMID_DIR, f"{base_filename}_level_{level}")
        if not os.path.exists(level_dir):
            os.makedirs(level_dir)
        
        height, width = current_image.shape[:2]
        tiles_x = (width + tile_size - 1) // tile_size
        tiles_y = (height + tile_size - 1) // tile_size
        
        # 保存当前层级信息
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
        
        # 创建并保存所有瓦片
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
        
        # 保存层级信息
        info_path = os.path.join(level_dir, "info.json")
        with open(info_path, "w") as f:
            json.dump(pyramid_info, f)
        
        pyramid_files.append({
            "level": level,
            "path": level_dir,
            "info": pyramid_info
        })
        
        # 检查是否需要继续创建更小的层级
        if width <= tile_size and height <= tile_size:
            break
        
        # 缩小图像（下采样）
        current_image = transform.rescale(current_image, 0.5, anti_aliasing=True)
        level += 1
    
    return pyramid_files

# API端点
@app.post("/datasets/")
async def create_dataset(
    name: str = Query(...),
    description: str = Query(...),
    type: str = Query(...),
    source: str = Query(...),
    db: SessionLocal = Depends(get_db)
):
    """创建新的数据集"""
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

@app.get("/datasets/")
async def list_datasets(
    type: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: SessionLocal = Depends(get_db)
):
    """列出所有数据集"""
    query = db.query(Dataset)
    if type:
        query = query.filter(Dataset.type == type)
    if source:
        query = query.filter(Dataset.source == source)
    
    datasets = query.offset(skip).limit(limit).all()
    return datasets

@app.post("/datasets/{dataset_id}/upload-image")
async def upload_image_to_dataset(
    dataset_id: int,
    file: UploadFile = File(...),
    is_base_layer: bool = Query(False),
    db: SessionLocal = Depends(get_db)
):
    """上传图像文件到指定数据集"""
    # 检查数据集是否存在
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # 读取文件内容
        content = await file.read()
        
        # 生成唯一的文件名
        file_extension = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # 保存文件
        with open(file_path, "wb") as f:
            f.write(content)
        
        # 处理FITS文件
        if file_extension == ".fits":
            with fits.open(io.BytesIO(content)) as hdul:
                data = hdul[0].data
                height, width = data.shape[:2]
                
                # 创建图像金字塔以支持缩放
                pyramid_base_filename = os.path.splitext(unique_filename)[0]
                pyramid_files = create_image_pyramid(data, pyramid_base_filename)
                
                # 保存图像信息到数据库
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
                
                # 保存金字塔层级信息
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
            # 处理其他图像格式（简化版）
            # 在实际应用中，应该使用适当的库来读取其他格式的图像
            image_file = ImageFile(
                dataset_id=dataset_id,
                filename=file.filename,
                file_path=file_path,
                width=0,  # 占位符
                height=0,  # 占位符
                file_size=len(content),
                image_type=file_extension[1:],  # 去掉点号
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
    """列出指定数据集的所有图像"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return dataset.files

@app.get("/images/{image_id}/pyramid-info")
async def get_image_pyramid_info(
    image_id: int,
    db: SessionLocal = Depends(get_db)
):
    """获取图像金字塔信息，用于前端缩放"""
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
    """获取图像的特定瓦片，用于前端显示"""
    image_file = db.query(ImageFile).filter(ImageFile.id == image_id).first()
    if not image_file:
        raise HTTPException(status_code=404, detail="Image not found")
    
    pyramid_level = db.query(PyramidLevel).filter(
        PyramidLevel.image_file_id == image_id, 
        PyramidLevel.level == level
    ).first()
    
    if not pyramid_level:
        raise HTTPException(status_code=404, detail="Pyramid level not found")
    
    # 读取瓦片信息
    base_filename = os.path.splitext(os.path.basename(image_file.file_path))[0]
    level_dir = os.path.join(PYRAMID_DIR, f"{base_filename}_level_{level}")
    
    try:
        # 读取层级信息
        with open(os.path.join(level_dir, "info.json"), "r") as f:
            level_info = json.load(f)
        
        # 检查请求的瓦片是否存在
        tile_found = False
        for tile in level_info["tiles"]:
            if tile["x"] == x and tile["y"] == y:
                tile_path = os.path.join(level_dir, tile["filename"])
                if os.path.exists(tile_path):
                    # 加载瓦片数据
                    tile_data = np.load(tile_path)
                    # 将NumPy数组转换为列表返回
                    return {
                        "data": tile_data.tolist(),
                        "width": tile_data.shape[1],
                        "height": tile_data.shape[0]
                    }
                tile_found = True
                break
        
        if not tile_found:
            raise HTTPException(status_code=404, detail="Tile not found")
    except Exception as e:
        logger.error(f"Error retrieving tile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving tile: {str(e)}")

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
    """在数据集中创建新标签"""
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
    """获取数据集中的标签，支持按区域和类型过滤"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    query = db.query(Label).filter(Label.dataset_id == dataset_id)
    
    # 按区域过滤
    if x_min is not None and y_min is not None and x_max is not None and y_max is not None:
        # 简化的区域过滤，实际应用中可能需要更复杂的几何计算
        query = query.filter(
            Label.x.between(x_min, x_max),
            Label.y.between(y_min, y_max)
        )
    
    # 按标签类型过滤
    if label_type:
        query = query.filter(Label.label == label_type)
    
    labels = query.all()
    return labels

@app.post("/process-image/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    method: str = Query("denoise", enum=["denoise", "contrast", "sharpen"]),
    strength: float = Query(0.1, ge=0.01, le=1.0)
):
    """增强图像质量"""
    try:
        content = await file.read()
        
        with fits.open(io.BytesIO(content)) as hdul:
            data = hdul[0].data
            
            # 根据选择的方法增强图像
            if method == "denoise":
                enhanced = restoration.denoise_tv_chambolle(data, weight=strength)
            elif method == "contrast":
                # 简单的对比度增强
                p2, p98 = np.percentile(data, (2, 98))
                enhanced = np.clip((data - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
            elif method == "sharpen":
                # 简单的锐化
                from skimage.filters import unsharp_mask
                if len(data.shape) > 2:
                    # 对多通道图像应用锐化
                    enhanced = np.zeros_like(data)
                    for i in range(data.shape[0]):
                        enhanced[i] = unsharp_mask(data[i], radius=1, amount=strength)
                else:
                    enhanced = unsharp_mask(data, radius=1, amount=strength)
            
        # 返回增强后的图像数据（限制大小用于演示）
        return {
            "data": enhanced.tolist()[:1000],
            "method": method,
            "strength": strength
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
    """搜索图像中的特征（简化版，实际应用中可能需要AI支持）"""
    # 这里只是一个简单的实现，匹配标签中的文本
    # 在实际应用中，可以集成AI模型进行更复杂的图像特征搜索
    
    search_query = f"%{query.lower()}%"
    
    if dataset_id:
        # 检查数据集是否存在
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
async def root():
    """API根端点"""
    return {
        "message": "Welcome to NASA Space App Challenge - Embiggen Your Eyes",
        "version": "1.0",
        "endpoints": [
            {"method": "GET", "path": "/", "description": "API root endpoint"},
            {"method": "POST", "path": "/datasets/", "description": "Create a new dataset"},
            {"method": "GET", "path": "/datasets/", "description": "List all datasets"},
            {"method": "POST", "path": "/datasets/{dataset_id}/upload-image", "description": "Upload an image to a dataset"},
            {"method": "GET", "path": "/datasets/{dataset_id}/images/", "description": "List images in a dataset"},
            {"method": "GET", "path": "/images/{image_id}/pyramid-info", "description": "Get image pyramid information"},
            {"method": "GET", "path": "/images/{image_id}/tile/{level}/{x}/{y}", "description": "Get a specific image tile"},
            {"method": "POST", "path": "/datasets/{dataset_id}/labels/", "description": "Create a new label"},
            {"method": "GET", "path": "/datasets/{dataset_id}/labels/", "description": "Get labels in a dataset"},
            {"method": "POST", "path": "/process-image/enhance", "description": "Enhance image quality"},
            {"method": "GET", "path": "/search-features", "description": "Search for features in images"}
        ]
    }




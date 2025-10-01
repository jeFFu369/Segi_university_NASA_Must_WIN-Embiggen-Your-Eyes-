# Segi_university_NASA_Must_WIN-Embiggen-Your-Eyes-

# Backend Development
## The challenge of creating a platform to browse NASA's massive imagery datasets. Requirements included:

1. Allowing users to zoom in and out of these massive imagery datasets
2. Annotating known features
3. Discovering new patterns
4. Processing different types of data products (different time periods, different colors, different types of data)

The main.py file has been enhanced to add a comprehensive set of features to support processing, scaling, labeling, and pattern discovery for large-scale image datasets:
- Designed a comprehensive database model (Dataset, ImageFile, PyramidLevel, Label)
- Implemented an image pyramid structure to support efficient image scaling and browsing
- Provided a RESTful API with features for dataset management, image upload, label creation, and search
- Supported processing of FITS and other image formats
- Added image enhancement features (denoising, contrast adjustment, sharpening)
- Implemented search functionality to allow users to find features in images
- Added error handling and logging

These enhancements fully meet the requirements of the challenge and include:
- Support for zooming and navigating large NASA image datasets
- Allowing users to mark known features and discover new patterns
- Providing multiple ways to search image content
- Supporting image processing and enhancement
- Implementing efficient data storage and access mechanisms
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.distance import cdist
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Step 1: Load training data
data_path = "D:\\pycharm\\PythonProject\\data\\YNCGST.csv"  # 替换为您的训练数据路径
data = pd.read_csv(data_path)
target_column = 'Tetracyclines'
feature_columns = [col for col in data.columns if col not in ['Longitude', 'Latitude', target_column]]

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_features = train_data[feature_columns].values
train_target = train_data[target_column].values
train_positions = train_data[['Longitude', 'Latitude']].values

# Step 2: Load spatial raster data from F:\predict
def load_raster_data(raster_folder, feature_names):
    raster_data = {}
    for feature in feature_names:
        raster_path = f"{raster_folder}\\{feature}.tif"
        raster = gdal.Open(raster_path)
        if raster is None:
            raise FileNotFoundError(f"Raster file not found: {raster_path}")
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()
        geo_transform = raster.GetGeoTransform()
        raster_data[feature] = (array, geo_transform)
    return raster_data

raster_folder = "F:\\predict"  # 替换为您的栅格数据路径
raster_data = load_raster_data(raster_folder, feature_columns)

# Step 3: Load boundary files
boundary_paths = {
    "CX": "F:\\滇中数据\\滇中边界\\CX_boundary.shp",
    "KM": "F:\\滇中数据\\滇中边界\\KM_boundary.shp",
    "MZ": "F:\\滇中数据\\滇中边界\\MZ_boundary.shp",
    "QJ": "F:\\滇中数据\\滇中边界\\QJ_boundary.shp",
    "YX": "F:\\滇中数据\\滇中边界\\YX_boundary.shp"
}

boundaries = [gpd.read_file(path) for path in boundary_paths]

# Step 4: Create grid points within the boundaries
def create_grid(boundary_gdf, resolution=0.01):
    minx, miny, maxx, maxy = boundary_gdf.total_bounds
    x_coords = np.arange(minx, maxx, resolution)
    y_coords = np.arange(miny, maxy, resolution)
    grid_points = np.array([(x, y) for x in x_coords for y in y_coords])
    grid_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in grid_points], crs=boundary_gdf.crs)
    grid_gdf = grid_gdf[grid_gdf.geometry.within(boundary_gdf.unary_union)]
    return grid_gdf

# Step 3: 为每个边界生成网格点
grid_data = {}
resolution = 0.01

for region, path in boundary_paths.items():
    print(f"Processing region: {region}")
    try:
        boundary_gdf = gpd.read_file(path)
        if boundary_gdf.empty:
            raise ValueError(f"Boundary file is empty for region: {region}")
        grid_gdf = create_grid(boundary_gdf, resolution=resolution)
        grid_data[region] = grid_gdf
        print(f"Generated {len(grid_gdf)} grid points for region: {region}")
    except Exception as e:
        print(f"Error processing region {region}: {e}")

grid_points = [create_grid(boundary) for boundary in boundaries]
grid_positions = np.array([[point.x, point.y] for point in grid_gdf.geometry])  # 转换为二维数组

# GWR Function
def gwr_predict(features, target, train_positions, grid_positions, bandwidth):
    """
    Perform Geographically Weighted Regression (GWR) predictions.
    """
    predictions = []

    for i, grid_point in enumerate(grid_positions):
        # Calculate distances and weights
        distances = np.linalg.norm(train_positions - grid_point, axis=1)
        weights = np.exp(-distances**2 / (2 * bandwidth**2))

        # Debugging: Check weight distribution
        if i < 5:
            print(f"Grid point {i}: {grid_point}")
            print(f"Weight stats - Min: {weights.min()}, Max: {weights.max()}, Mean: {weights.mean()}")

        # Weighted regression
        X_weighted = features * weights[:, np.newaxis]
        y_weighted = target * weights
        XtWX = X_weighted.T @ features + 1e-3 * np.eye(features.shape[1])  # Enhanced regularization
        XtWy = X_weighted.T @ y_weighted

        # Solve for coefficients
        try:
            coeffs = np.linalg.solve(XtWX, XtWy)
            prediction = coeffs @ features.mean(axis=0)
        except np.linalg.LinAlgError:
            prediction = np.nan  # Handle singular matrix cases
        predictions.append(prediction)

    return np.array(predictions)



# Bandwidth optimization
def adaptive_bandwidth(train_positions, grid_point, k=30):
    """
    Calculate adaptive bandwidth based on k-nearest neighbors.
    """
    distances = np.linalg.norm(train_positions - grid_point, axis=1)
    sorted_distances = np.sort(distances)
    return sorted_distances[k-1]  # Use the distance to the k-th nearest neighbor
for grid_point in grid_positions:
    bandwidth = adaptive_bandwidth(train_positions, grid_point, k=30)
    prediction = gwr_predict(train_features, train_target, train_positions, [grid_point], bandwidth)



# GOS Function
def gos_predict(features, target, grid_features, kappa=0.01):
    predictions = []
    for grid_point in grid_features:
        similarities = np.exp(-np.linalg.norm(features - grid_point, axis=1)**2 / (2 * np.var(features)))
        top_indices = np.argsort(similarities)[-int(len(similarities) * kappa):]
        weights = similarities[top_indices]
        prediction = np.sum(weights * target[top_indices]) / np.sum(weights)
        predictions.append(prediction)
    return np.array(predictions)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(train_features, train_target)

# Generate Predictions for Each Boundary
results = []
for i, (boundary, grid_gdf) in enumerate(zip(boundaries, grid_points)):
    grid_coords = np.array([[point.x, point.y] for point in grid_gdf.geometry])


    gwr_predictions = gwr_predict(train_features, train_target, train_positions, grid_coords, adaptive_bandwidth)
    gos_predictions = gos_predict(train_features, train_target, grid_coords)
    rf_predictions = rf_model.predict(grid_coords)

    result_df = pd.DataFrame({
        'Longitude': grid_coords[:, 0],
        'Latitude': grid_coords[:, 1],
        'GWR': gwr_predictions,
        'GOS': gos_predictions,
        'RF': rf_predictions
    })
    result_df.to_csv(f"D:\\pycharm\\PythonProject\\Predictions_Boundary_{i}.csv", index=False)
    print(f"Results for boundary {i} saved.")


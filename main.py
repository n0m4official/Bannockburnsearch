import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import json

# -----------------------------
# CONFIG / JSON DATA
# -----------------------------
np.random.seed(42)

N_SIM = 250_000
HOURS = 24
DT = 600  # 10-minute steps
N_STEPS = HOURS * 3600 // DT

# Vessel data (from JSON)
ship_data = {
    "length_ft": 245.0,
    "beam_ft": 40.1,
    "depth_of_hold_ft": 18.4,
    "gross_tonnage": 1620,
    "cargo_tons": 1500
}
ship = {
    "draft_m": ship_data["depth_of_hold_ft"] * 0.3048,
    "gross_tonnage": ship_data["gross_tonnage"],
    "cargo_tons": ship_data["cargo_tons"]
}

# Storm reconstruction
wave_height_ft = 15
start_lat = 47.75
start_lon = -87.16

# Recovered debris points
wreckage_points = [
    (46.5, -84.35),  # Grand Marais
    (46.5, -83.7)    # Near Sault Ste. Marie
]

# -----------------------------
# LOAD BATHYMETRY
# -----------------------------
ds = xr.open_dataset("superior_lld.grd")
lat_grid = ds["y"].values
lon_grid = ds["x"].values
depth_grid = ds["z"].values

if lat_grid[0] > lat_grid[-1]:
    lat_grid = lat_grid[::-1]
    depth_grid = depth_grid[::-1, :]

depth_grid[depth_grid < -10000] = np.nan

bathy_interp = RegularGridInterpolator(
    (lat_grid, lon_grid),
    depth_grid,
    bounds_error=False,
    fill_value=np.nan
)

# -----------------------------
# Bathymetric Slope
# -----------------------------
dy, dx = np.gradient(depth_grid)
slope_grid = np.sqrt(dx**2 + dy**2)
slope_interp = RegularGridInterpolator(
    (lat_grid, lon_grid),
    slope_grid,
    bounds_error=False,
    fill_value=0
)

# -----------------------------
# Coordinate Conversion
# -----------------------------
EARTH_RADIUS = 6371000
LAT_REF = start_lat

def latlon_to_xy(lat, lon):
    x = np.deg2rad(lon) * EARTH_RADIUS * np.cos(np.deg2rad(LAT_REF))
    y = np.deg2rad(lat) * EARTH_RADIUS
    return np.array([x, y])

def xy_to_latlon(x, y):
    lat = np.rad2deg(y / EARTH_RADIUS)
    lon = np.rad2deg(x / (EARTH_RADIUS * np.cos(np.deg2rad(LAT_REF))))
    return lat, lon

# -----------------------------
# Synthetic Basin Current
# -----------------------------
def realistic_current_v2(lat, lon, wind_mps, wind_dir_deg, depth_grid, lat_grid, lon_grid):
    rho_air = 1.225
    C_d = 1.3e-3
    
    # 1. Calculate the 'Base' Direction (Wind-Driven)
    # Wind direction in your code is 'blowing toward', so we add deflection to the right
    deflection_deg = 25.0 
    deflected_dir_rad = np.deg2rad(wind_dir_deg + deflection_deg)
    
    tau_x = rho_air * C_d * wind_mps**2 * np.sin(deflected_dir_rad)
    tau_y = rho_air * C_d * wind_mps**2 * np.cos(deflected_dir_rad)
    
    # 2. Get depth from grid
    interp = RegularGridInterpolator((lat_grid, lon_grid), depth_grid, 
                                     bounds_error=False, fill_value=np.nan)
    points = np.vstack((lat, lon)).T
    depth_m = interp(points)
    depth_m = np.nan_to_num(depth_m, nan=100.0) # Default to 100m for deep basin
    depth_m = np.clip(depth_m, 5.0, None) # Avoid divide-by-zero
    
    # 3. Final Current Velocity (m/s)
    rho_water = 1000
    u = tau_x / (rho_water * depth_m)
    v = tau_y / (rho_water * depth_m)
    
    return u, v

# -----------------------------
# Time-Varying Storm Wind
# -----------------------------
def wind_field(hour):
    # NW storm (historical "Big Push")
    if hour < 8:
        return 45 * 0.44704, 135  # blowing TO SE
    elif hour < 16:
        return 50 * 0.44704, 135
    else:
        return 47 * 0.44704, 135

# -----------------------------
# Structural Failure with Rogue Waves
# -----------------------------
def failure_probability(wind_mps, wave_ft, ship, depth):
    stress = wind_mps * wave_ft / np.log1p(ship["gross_tonnage"])
    # Rogue wave spike for deep water (>200m)
    rogue = np.where(depth < -200, np.random.uniform(0,0.1), 0)
    return 1 - np.exp(-stress / 1800) + rogue

# -----------------------------
# Initialize Particles
# -----------------------------
start_xy = latlon_to_xy(start_lat, start_lon)
positions = np.tile(start_xy, (N_SIM, 1))
alive = np.ones(N_SIM, dtype=bool)

lat_min, lat_max = lat_grid.min(), lat_grid.max()
lon_min, lon_max = lon_grid.min(), lon_grid.max()

# -----------------------------
# Simulation Loop
# -----------------------------
for step in range(N_STEPS):
    hour = step * DT / 3600
    wind_mps, wind_dir = wind_field(hour)
    wind_rad = np.deg2rad(wind_dir)

    # Convert particle positions to lat/lon
    lat_vals, lon_vals = xy_to_latlon(positions[:,0], positions[:,1])
    lat_vals = np.clip(lat_vals, lat_min, lat_max)
    lon_vals = np.clip(lon_vals, lon_min, lon_max)

    # Wind vector to correct SE drift
    wind_leeway = 0.035 if hour < 16 else 0.03
    wind_leeway /= np.sqrt(ship["cargo_tons"] / 1500)
    wind_vector = wind_leeway * wind_mps * np.tile(np.array([np.sin(wind_rad), np.cos(wind_rad)]), (N_SIM, 1))

    # Synthetic currents
    u, v = realistic_current_v2(lat_vals, lon_vals, wind_mps, wind_dir, depth_grid, lat_grid, lon_grid)

    # Turbulence + stochastic gusts
    turbulence_scale = 0.001 * (wind_mps / 20)
    turbulence = np.random.normal(scale=turbulence_scale, size=(N_SIM, 2))

    # Move particles
    positions[alive] += np.column_stack((u[alive], v[alive]))*DT + wind_vector[alive]*DT + turbulence[alive]*DT

    # Bathymetry & grounding
    depths = bathy_interp(np.vstack((lat_vals, lon_vals)).T)
    grounded = (depths > -ship["draft_m"]) | np.isnan(depths)
    alive[grounded] = False

    # Structural failure
    prob_fail = failure_probability(wind_mps, wave_height_ft, ship, depths)
    random_fail = np.random.rand(N_SIM) < prob_fail * 0.01
    alive[random_fail] = False

# -----------------------------
# Final Positions
# -----------------------------
final_lat, final_lon = xy_to_latlon(positions[:,0], positions[:,1])
valid = ~np.isnan(final_lat)
final_lat = final_lat[valid]
final_lon = final_lon[valid]

# -----------------------------
# Slope-Based Weighting
# -----------------------------
points = np.vstack((final_lat, final_lon)).T
slopes = slope_interp(points)
weights = 1 + (slopes / np.nanmax(slopes))

# -----------------------------
# Debris-Based Bayesian Filter
# -----------------------------
for lat_w, lon_w in wreckage_points:
    dist = np.sqrt((final_lat - lat_w)**2 + (final_lon - lon_w)**2)
    weights *= 1 / (dist + 1e-3)  # down-weight particles far from debris

# -----------------------------
# Top 1% Density
# -----------------------------
threshold = np.percentile(weights, 99)
top_idx = weights >= threshold
top_lat = final_lat[top_idx]
top_lon = final_lon[top_idx]

# Save CSV
df = pd.DataFrame({"lat": top_lat, "lon": top_lon})
csv_file = "bannockburn_wreck_top1percent.csv"
df.to_csv(csv_file, index=False)
print(f"Top 1% density points saved to CSV: {csv_file}")

# Save GeoJSON
features = [{"type":"Feature","geometry":{"type":"Point","coordinates":[float(lon), float(lat)]}} for lat, lon in zip(top_lat, top_lon)]
geojson_file = "bannockburn_wreck_top1percent.geojson"
with open(geojson_file, "w") as f:
    json.dump({"type":"FeatureCollection","features":features}, f)
print(f"Top 1% density points saved to GeoJSON: {geojson_file}")

# -----------------------------
# Highest Density Area
# -----------------------------
center_lat = np.mean(top_lat)
center_lon = np.mean(top_lon)
lat_bbox = (top_lat.min(), top_lat.max())
lon_bbox = (top_lon.min(), top_lon.max())

print(f"Approx. highest-density area center: lat {center_lat:.3f}, lon {center_lon:.3f}")
print(f"Rough bounding box (top 1% density): lat {lat_bbox}, lon {lon_bbox}")

# -----------------------------
# Plot Probability Density with Lake Outline
# -----------------------------
plt.figure(figsize=(10,8))
hist = plt.hist2d(final_lon, final_lat, bins=200, weights=weights, cmap='Reds')
plt.colorbar(hist[3], label="Weighted Density")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("SS Bannockburn Probable Wreck Distribution (1902 Storm)")

# Outline Lake Superior
lake_mask = np.where(np.isnan(depth_grid), 0, 1)
plt.contour(lon_grid, lat_grid, lake_mask, levels=[0.5], colors='blue', linewidths=1.0)

# Recovered wreckage points
for i, (lat_w, lon_w) in enumerate(wreckage_points):
    plt.plot(lon_w, lat_w, 'bo', markersize=6, label="Recovered Wreckage" if i==0 else "")
plt.legend()
plt.show()

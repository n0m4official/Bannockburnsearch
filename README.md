# SS Bannockburn Predictive Search Model

## Overview
This project implements a **Monte Carlo Lagrangian Particle Tracking (LPT) model** to identify the most probable sinking location of the steel-hulled freighter **SS Bannockburn** (last sighted at 47.362° N, 86.584° W on November 21, 1902). The simulation bridges the gap between historical records of the ship’s disappearance and the known debris field recovered weeks later.

The model simulates vessel drift and structural failure under historical storm conditions, producing a high-probability search grid suitable for deep-water exploration.

---

## Features

### Atmospheric & Oceanographic Physics
- **Ekman Deflection / Coriolis Effect**: Wind-driven surface currents include a 25° right-hand deflection to simulate Northern Hemisphere Coriolis forces.  
- **Wind-Leeway Dynamics**: Vessel drift uses a 3.2–3.5% leeway coefficient, reflecting a heavily-laden deep-draft freighter.  
- **Synthetic Basin Circulation**: Depth-dependent current velocities are applied using **NOAA 3-arc second bathymetry**, allowing realistic acceleration over steep lakebed topography.

### Probability Weighting & Bayesian Filtering
- **Bathymetric Slope Weighting**: Particles near high-gradient features (ridges, shoals) are weighted higher to reflect increased risk from steep “Three Sisters” waves.  
- **Debris Alignment (Bayesian Filter)**: Particle distributions are refined against the only two confirmed artifacts: an oar near the St. Marys River and a cork life preserver near Grand Marais.

### Structural Failure Stochasticity
- Hull breach and capsizing are **treated as stochastic events** at 10-minute intervals.  
- Failure probability depends on:
  - Real-time wind stress from the 1902 “Big Push” storm.  
  - Rogue wave probability in water deeper than 200 m, reflecting deep-lake swell behavior.

---

## Output
- **Top 1% density search points** saved as:
  - `bannockburn_wreck_top1percent.csv`  
  - `bannockburn_wreck_top1percent.geojson`  
- Approximate highest-density area center and bounding box are printed in the console.  
- Visualization: heatmap of predicted wreck distribution overlaid with lake bathymetry and confirmed debris points.

---

## Usage
1. Install dependencies:  
```bash
pip install numpy matplotlib xarray scipy pandas
```
2. Place bathymetry file (`superior_lld.grd`) in the project directory (not included due to file size limitations).
3. Run the simulation:
```bash
python main.py
```
4. Results will be saved in CSV and GeoJSON formats, with a plotted probability heatmap.

---
## Conclusion
The simulation identifies a 300 m × 300 m high-probability search box in a deep-water, unassessed (ZOC Category U) area. Alignment of historical sightings, physics-based drift, and debris recovery coordinates makes this area a prime candidate for high-resolution side-scan sonar and ROV exploration.

import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, cast

import folium
import gpxpy
import gpxpy.gpx
import numpy as np
from branca import colormap
from folium import LayerControl
from folium.features import ColorLine
from pyproj import Transformer
from shapely import LineString, centroid
from shapely.geometry import Point
from shapely.ops import transform
from shapely.strtree import STRtree

# WGS84 to ETRS89-TM35FIN
cartesian_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3067", always_xy=True)
# ETRS89-TM35FIN to WGS84
geodetic_transformer = Transformer.from_crs("EPSG:3067", "EPSG:4326", always_xy=True)


def parse_gpx_files(files: List[str]) -> List[List[Tuple[float, float]]]:
    tracks: List[List[Tuple[float, float]]] = []
    for file_path in files:
        with open(file_path, "r") as gpx_file:
            gpx = gpxpy.parse(gpx_file)
            for track in gpx.tracks:
                for segment in track.segments:
                    track_points = [
                        (point.latitude, point.longitude) for point in segment.points
                    ]
                    tracks.append(track_points)
    return tracks


def transform_track(
    track: List[Tuple[float, float]], transformer: Transformer
) -> List[Tuple[float, float]]:
    """Transform track between coordinate systems using transformer"""
    lons, lats = zip(*track)
    x, y = transformer.transform(lons, lats)
    return list(zip(x, y))


def tracks_to_linestrings(tracks: List[List[Tuple[float, float]]]) -> List[LineString]:
    """Transform track coordinates into a cartesien coordinate system for geometric
    operations and make shapely objects"""
    return [
        LineString(transform_track(track, cartesian_transformer)) for track in tracks
    ]


def calculate_midpoints(
    tracks: List[LineString],
) -> Dict[int, List[Tuple[float, float]]]:
    """Get midpoint for each line between coordinate points. Return all midpoints with
    track id reference"""
    segment_midpoints = defaultdict(list)
    for track_id, line in enumerate(tracks):
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            midpoint = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            segment_midpoints[track_id].append(midpoint)
    return segment_midpoints


def calculate_densities(
    track_midpoints: Dict[int, List[Tuple[float, float]]],
    rtree: STRtree,
) -> Dict[int, List[int]]:
    """Calculate density value for each line segment in each track.
    Very slow process so farm the work out to all cpu cores."""
    densities: Dict[int, List[int]] = defaultdict(list)
    futures = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for track_id, midpoints in track_midpoints.items():
            futures.append(
                executor.submit(calculate_track_densities, track_id, midpoints, rtree)
            )
    for future in as_completed(futures):
        track_id, track_densities = future.result()
        densities[track_id] = track_densities
    return densities


def calculate_track_densities(
    track_id: int,
    midpoints: List[Tuple[float, float]],
    rtree: STRtree,
) -> Tuple[int, List[int]]:
    """Calculate density, which is a count of unique nearby tracks,
    for the midpoint of each line segment. RTree is queried for
    candidates which is a fast check based on bounding boxes.
    Each candidate is checked for actual distance to the midpoint."""
    R = 50.0  # distance in meters required for nearness
    densities = []
    for midpoint in midpoints:
        pt = Point(midpoint)
        buffer_area = pt.buffer(R)

        # Get candidate track ids from spatial index
        candidate_indexes = rtree.query(buffer_area)
        candidates = rtree.geometries.take(candidate_indexes)

        # Count how many *distinct tracks* are near
        near_tracks = set()
        for candidate_line in candidates:
            dist = candidate_line.distance(pt)
            if dist <= R:
                near_tracks.add(candidate_line)
        densities.append(len(near_tracks))
    return track_id, densities


def plot(gpx_files: List[str], output_file="map.html"):
    tracks = parse_gpx_files(gpx_files)
    if tracks is None:
        print("No tracks to plot.")
        return
    original_lines = tracks_to_linestrings(tracks)
    # GPX tracks are simplified to reduce the number of coordinate points
    # to significantly reduce the density calculation effort
    lines = [line.simplify(2, preserve_topology=False) for line in original_lines]
    lines = cast(List[LineString], lines)
    midpoints = calculate_midpoints(lines)
    # construct a spatial index of the track geometries for nearness querying
    rtree = STRtree(lines)
    densities = calculate_densities(midpoints, rtree)
    all_densities = list(
        density for line_densities in densities.values() for density in line_densities
    )
    min_density = min(all_densities)
    # max_density = max(all_densities)
    # 80th percentile is used to cut top end of the densities
    # this is highly case specific - if all tracks start and end at the same spot
    # the density peaks there
    max_density = float(np.percentile(all_densities, 80))

    center_point = centroid(lines[0])
    center_x, center_y = geodetic_transformer.transform(center_point.x, center_point.y)
    base_map = folium.Map(
        location=[center_x, center_y],
        zoom_start=13,
        tiles="https://tiles.kartat.kapsi.fi/taustakartta/{z}/{x}/{y}.jpg",
        attr='Map data &copy; <a href="https://www.maanmittauslaitos.fi/en">Maanmittauslaitos</a>',
    )
    cm = colormap.LinearColormap(
        ["green", "yellow", "orange", "red"], vmin=min_density, vmax=max_density
    )

    geodetic_projection = geodetic_transformer.transform
    for i, line in enumerate(lines):
        density = np.array(densities[i])
        geodetic_line = transform(geodetic_projection, line)
        ColorLine(positions=geodetic_line.coords, colors=density, colormap=cm, weight=3).add_to(
            base_map
        )

    LayerControl().add_to(base_map)
    base_map.save(output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gpx_files", metavar="GPX_FILE", type=str, nargs="+", help="GPX files to plot"
    )
    parser.add_argument("--output", "-o", default="map.html")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(args.gpx_files, args.output)

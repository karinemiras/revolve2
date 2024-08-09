import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Coordinates
coordinates = [(0, 0), (-1, 0), (0, 1), (-2, 0), (1, 0), (0, -1)]


# Define the squares
def create_square(x, y):
    return Polygon([(x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x + 0.5, y + 0.5), (x - 0.5, y + 0.5)])


# Create square polygons
polygons = [create_square(x, y) for x, y in coordinates]

# Combine all polygons into one shape
shape = unary_union(polygons)

# Extract the exterior coordinates of the shape
if shape.geom_type == 'Polygon':
    exterior_coords = np.array(shape.exterior.coords)
elif shape.geom_type == 'MultiPolygon':
    exterior_coords = np.concatenate([np.array(p.exterior.coords) for p in shape.geoms])
else:
    raise ValueError("Shape must be a Polygon or MultiPolygon")


# Function to remove collinear points and find unique boundary points
def remove_collinear_points(coords):
    unique_points = []
    for i in range(len(coords)):
        p1 = coords[i]
        p2 = coords[(i + 1) % len(coords)]
        p3 = coords[(i + 2) % len(coords)]

        # Vector calculations
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)

        # Check for collinearity
        if not np.allclose(np.cross(v1, v2), 0):
            unique_points.append(p2)

    return np.array(unique_points)


# Function to count and draw 90-degree changes
def count_and_draw_90_degree_changes(coords):
    changes = 0
    marked_segments = []
    seen_segments = set()  # To avoid counting duplicate segments

    for i in range(len(coords)):
        p1 = coords[i]
        p2 = coords[(i + 1) % len(coords)]
        p3 = coords[(i + 2) % len(coords)]

        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)

        # Compute dot product
        dot_product = np.dot(v1, v2)

        # Compute the magnitude of vectors
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)

        # Compute the angle between vectors
        angle = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
        angle_degrees = np.degrees(angle)

        # Check for 90-degree angle
        if np.isclose(angle_degrees, 90):
            changes += 1
            # Mark the segments where the changes occur
            seg1 = tuple(sorted([tuple(p1), tuple(p2)]))
            seg2 = tuple(sorted([tuple(p2), tuple(p3)]))

            if seg1 not in seen_segments:
                marked_segments.append((p1, p2))
                seen_segments.add(seg1)

            if seg2 not in seen_segments:
                marked_segments.append((p2, p3))
                seen_segments.add(seg2)

    return changes, marked_segments


# Remove collinear points
unique_coords = remove_collinear_points(exterior_coords)

# Calculate the number of 90-degree changes and marked segments
num_changes, marked_segments = count_and_draw_90_degree_changes(unique_coords)

# Plot all lines in black
plt.figure()

# Draw the full boundary of the shape in black
plt.plot(exterior_coords[:, 0], exterior_coords[:, 1], 'k-', alpha=0.7)

# Draw the segments with 90-degree changes in black, ensuring no redundancy
for (start, end) in marked_segments:
    plt.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Envelope of the Shape with Direction Changes')
plt.grid(True)
plt.gca().set_aspect('equal')  # Ensure the aspect ratio is equal
plt.show()

print(f"Number of 90-degree changes: {num_changes}")

# Output the exact segments where changes occur
print("Segments with 90-degree changes:")
for (start, end) in marked_segments:
    print(f"From {start} to {end}")

plt.savefig(f'cu.png')

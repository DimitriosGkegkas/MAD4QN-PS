import math

class Vehicle:
    def __init__(self, vehicle_id, position, velocity, goal):
        self.vehicle_id = vehicle_id
        self.position = position  # Current position (x, y)
        self.velocity = velocity  # Current velocity (m/s)
        self.goal = goal  # Goal position (x, y)

class IntersectionManager:
    def __init__(self, intersection_width, intersection_height, center=(0, 0)):
        self.vehicles = []
        self.intersection_width = intersection_width
        self.intersection_height = intersection_height
        self.center = center  # Center of the intersection (x, y)

    def add_vehicles(self, vehicles):
        self.vehicles = vehicles

    def calculate_conflict_points(self):
        conflict_points = []
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                conflict_point = self.find_conflict_point(self.vehicles[i], self.vehicles[j])
                if conflict_point:
                    conflict_points.append((self.vehicles[i], self.vehicles[j], conflict_point))
        return conflict_points

    def find_conflict_point(self, vehicle1, vehicle2):
        entry1 = self.calculate_intersection_point(vehicle1.position)
        entry2 = self.calculate_intersection_point(vehicle2.position)
        exit1 = self.calculate_intersection_point(vehicle1.goal)
        exit2 = self.calculate_intersection_point(vehicle2.goal)

        # Check if their paths intersect within the intersection
        denom = (exit1[0] - entry1[0]) * (exit2[1] - entry2[1]) - (exit1[1] - entry1[1]) * (exit2[0] - entry2[0])
        if abs(denom) < 1e-6:  # Parallel paths
            return None

        t1 = ((entry2[0] - entry1[0]) * (exit2[1] - entry2[1]) - (entry2[1] - entry1[1]) * (exit2[0] - entry2[0])) / denom
        t2 = ((entry2[0] - entry1[0]) * (exit1[1] - entry1[1]) - (entry2[1] - entry1[1]) * (exit1[0] - entry1[0])) / denom

        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            # Conflict point within the intersection
            cx = entry1[0] + t1 * (exit1[0] - entry1[0])
            cy = entry1[1] + t1 * (exit1[1] - entry1[1])
            return (cx, cy)
        return None

    def calculate_intersection_point(self, position):
        # Map position to nearest entry/exit point of the intersection relative to the center
        cx, cy = self.center
        x, y = position
        return (
            max(cx - self.intersection_width / 2, min(cx + self.intersection_width / 2, x)),
            max(cy - self.intersection_height / 2, min(cy + self.intersection_height / 2, y))
        )

    def calculate_target_velocities(self):
        target_velocities = {}
        conflict_points = self.calculate_conflict_points()

        for vehicle in self.vehicles:
            target_velocities[vehicle.vehicle_id] = 13  # Default to current velocity

        # Build a conflict graph
        conflict_graph = {vehicle.vehicle_id: [] for vehicle in self.vehicles}
        conflict_details = {}
        for v1, v2, conflict_point in conflict_points:
            conflict_graph[v1.vehicle_id].append(v2.vehicle_id)
            conflict_details[(v1.vehicle_id, v2.vehicle_id)] = (v1, v2, conflict_point)

        # Perform topological sort to resolve cascading conflicts
        resolved = set()
        stack = []
        result = []

        def topological_sort(vehicle_id):
            if vehicle_id in resolved:
                return
            stack.append(vehicle_id)
            for neighbor in conflict_graph[vehicle_id]:
                if neighbor in stack:  # Cycle detection
                    raise ValueError("Cyclic conflict detected!")
                topological_sort(neighbor)
            stack.pop()
            resolved.add(vehicle_id)
            result.append(vehicle_id)

        for vehicle in self.vehicles:
            if vehicle.vehicle_id not in resolved:
                topological_sort(vehicle.vehicle_id)

        # Resolve conflicts in precedence order
        for vehicle_id in reversed(result):
            for neighbor_id in conflict_graph[vehicle_id]:
                if (vehicle_id, neighbor_id) in conflict_details:
                    v1, v2, conflict_point = conflict_details[(vehicle_id, neighbor_id)]
                    t1 = self.time_to_reach(v1, conflict_point)
                    t2 = self.time_to_reach(v2, conflict_point)

                    if t1 < t2:
                        # Vehicle 1 reaches first, adjust Vehicle 2's velocity
                        adjusted_velocity = self.adjust_velocity(v2, conflict_point, t1 + 4)  # Add buffer time
                        target_velocities[v2.vehicle_id] = adjusted_velocity
                    else:
                        # Vehicle 2 reaches first, adjust Vehicle 1's velocity
                        adjusted_velocity = self.adjust_velocity(v1, conflict_point, t2 + 4)  # Add buffer time
                        target_velocities[v1.vehicle_id] = adjusted_velocity

        return target_velocities

    def time_to_reach(self, vehicle, point):
        pxe, pye = self.calculate_intersection_point(vehicle.position)
        px, py = vehicle.position
        cx, cy = point
        if self.is_turn(px, py, cx, cy):
            distance = self.calculate_curve_distance(pxe, pye, cx, cy) + math.sqrt((cx - pxe) ** 2 + (cy - pye) ** 2)
        else:
            distance = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        return distance / max(vehicle.velocity, 1e-3)  # Prevent division by zero

    def is_turn(self, px, py, cx, cy):
        # Determines if the path involves a turn (not on the same road)
        return px != cx and py != cy

    def calculate_curve_distance(self, px, py, cx, cy):
        # Approximate the curve distance as a quarter-circle for a 90-degree turn
        radius = math.sqrt((cx - px) ** 2 + (cy - py) ** 2) / 2
        return math.pi * radius / 2  # Arc length of a quarter-circle


    def adjust_velocity(self, vehicle, point, time_to_reach):
        px, py = vehicle.position
        cx, cy = point
        distance = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        return distance / time_to_reach

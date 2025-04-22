
def get_bbox_center(bbox: list[float]) -> tuple:
    """Get the center of a bounding box"""
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_bbox_width(bbox: list[float]) -> float:
    """Get the width of a bounding box"""
    return bbox[2]-bbox[0]

def measure_distance(p1, p2):
    """Calculate the distance between two points"""
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_foot_position(bbox):
    """Get the foot position of a player based on their bounding box"""
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2),int(y2)

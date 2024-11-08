def check_overlaps(object_positions):
    overlaps = []
    for i, obj1 in enumerate(object_positions):
        x1_min = obj1['xcoord'] - obj1['w'] / 2
        x1_max = obj1['xcoord'] + obj1['w'] / 2
        y1_min = obj1['ycoord'] - obj1['w'] / 2
        y1_max = obj1['ycoord'] + obj1['w'] / 2
        for j, obj2 in enumerate(object_positions):
            if i >= j:
                continue  # Skip self-comparison and duplicates
            x2_min = obj2['xcoord'] - obj2['w'] / 2
            x2_max = obj2['xcoord'] + obj2['w'] / 2
            y2_min = obj2['ycoord'] - obj2['w'] / 2
            y2_max = obj2['ycoord'] + obj2['w'] / 2
            # Check for overlap
            if not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max):
                overlaps.append((obj1['obj_file'], obj2['obj_file']))
    return overlaps

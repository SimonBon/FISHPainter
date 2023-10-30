def should_merge(box1, box2):
    """
    Determine if two bounding boxes should be merged.

    Parameters:
    - box1: A bounding box in format (color, y1, x1, y2, x2)
    - box2: A bounding box in format (color, y1, x1, y2, x2)

    Returns:
    - bool: True if boxes should be merged, False otherwise.
    """
    y1_inter = max(box1[1], box2[1])
    x1_inter = max(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    x2_inter = min(box1[4], box2[4])

    inter_area = max(0, y2_inter - y1_inter) * max(0, x2_inter - x1_inter)

    area1 = (box1[3] - box1[1]) * (box1[4] - box1[2])
    area2 = (box2[3] - box2[1]) * (box2[4] - box2[2])

    overlap_box1 = inter_area / area1
    overlap_box2 = inter_area / area2

    return overlap_box1 > 0.3 or overlap_box2 > 0.3

def merge_boxes_for_color(boxes):
    """
    Merge bounding boxes of the same color based on overlap criteria.

    Parameters:
    - boxes: List of bounding boxes of the same color in format [(color, y1, x1, y2, x2), ...]

    Returns:
    - List of merged bounding boxes
    """
    merged_boxes = []
    while boxes:
        main_box = boxes.pop(0)

        other_boxes = []
        merged_area = [main_box]

        for box in boxes:
            if should_merge(main_box, box):
                merged_area.append(box)
            else:
                other_boxes.append(box)

        if len(merged_area) >= 3:
            merged_box = (
                (0, 0, 1),  # change color to blue
                min(box[1] for box in merged_area),
                min(box[2] for box in merged_area),
                max(box[3] for box in merged_area),
                max(box[4] for box in merged_area)
            )
        elif len(merged_area) > 1:
            merged_box = (
                main_box[0],  # retain original color
                min(box[1] for box in merged_area),
                min(box[2] for box in merged_area),
                max(box[3] for box in merged_area),
                max(box[4] for box in merged_area)
            )
        else:
            merged_box = main_box

        merged_boxes.append(merged_box)
        boxes = other_boxes

    return merged_boxes

def merge_boxes_by_color(all_boxes):
    """
    Merge bounding boxes based on their colors.

    Parameters:
    - all_boxes: List of all bounding boxes in format [(color, y1, x1, y2, x2), ...]

    Returns:
    - List of merged bounding boxes.
    """
    colors = set(box[0] for box in all_boxes)
    merged_boxes = []

    for color in colors:
        boxes_of_color = [box for box in all_boxes if box[0] == color]
        merged_boxes.extend(merge_boxes_for_color(boxes_of_color))

    return merged_boxes

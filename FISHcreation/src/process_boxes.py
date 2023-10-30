def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1: A bounding box in format (y1, x1, y2, x2)
    - box2: A bounding box in format (y1, x1, y2, x2)

    Returns:
    - float: IoU ratio
    """
    y1_inter = max(box1[0], box2[0])
    x1_inter = max(box1[1], box2[1])
    y2_inter = min(box1[2], box2[2])
    x2_inter = min(box1[3], box2[3])

    inter_area = max(0, y2_inter - y1_inter + 1) * max(0, x2_inter - x1_inter + 1)

    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area

def merge_boxes(boxes):
    """
    Merge bounding boxes that have more than 75% overlap.

    Parameters:
    - boxes: List of bounding boxes in format [(y1, x1, y2, x2), ...]

    Returns:
    - List of merged bounding boxes
    """
    merged_boxes = []
    while boxes:
        main_box = boxes.pop(0)

        other_boxes = []
        merged_area = [main_box]

        for box in boxes:
            if compute_iou(main_box, box) > 0.75:
                merged_area.append(box)
            else:
                other_boxes.append(box)

        if len(merged_area) > 1:
            merged_box = (
                min(box[0] for box in merged_area),
                min(box[1] for box in merged_area),
                max(box[2] for box in merged_area),
                max(box[3] for box in merged_area)
            )
            merged_boxes.append(merged_box)
        else:
            merged_boxes.append(main_box)

        boxes = other_boxes

    return merged_boxes

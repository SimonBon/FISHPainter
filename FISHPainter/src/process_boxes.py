combine = {1: [1, 4], 2: [2, 3], 3: [3, 2], 4: [4, 1]}

def should_merge(box1, box2):
    """
    Determine if two bounding boxes should be merged.

    Parameters:
    - box1: A bounding box in format (labels, y1, x1, y2, x2)
    - box2: A bounding box in format (labels, y1, x1, y2, x2)

    Returns:
    - bool: True if boxes should be merged, False otherwise.
    """
    y1_inter = max(box1[0], box2[0])
    x1_inter = max(box1[1], box2[1])
    y2_inter = min(box1[2], box2[2])
    x2_inter = min(box1[3], box2[3])

    inter_area = max(0, y2_inter - y1_inter) * max(0, x2_inter - x1_inter)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    overlap_box1 = inter_area / area1
    overlap_box2 = inter_area / area2

    return overlap_box1 > 0.3 or overlap_box2 > 0.3


def merge_boxes_for_label(boxes_of_labels, labels_of_labels, combined_label):
    """
    Merge bounding boxes of the same label based on overlap criteria.

    Parameters:
    - boxes_of_labels: List of bounding boxes [(y1, x1, y2, x2)]
    - labels_of_labels: List of labels corresponding to each box in boxes_of_labels.

    Returns:
    - List of merged bounding boxes with labels.
    """
    new_merge = False
    merged_boxes_with_labels = []
    boxes_with_labels = list(zip(boxes_of_labels, labels_of_labels))
    
    while boxes_with_labels:
        main_box, main_label = boxes_with_labels.pop(0)

        other_boxes = []
        merged_area = [(main_box, main_label)]

        for box, label in boxes_with_labels:
            # Skip comparison if it's the same box
            if box == main_box:
                continue

            if should_merge(main_box, box):
                new_merge = True
                merged_area.append((box, label))
            else:
                other_boxes.append((box, label))

        if len(merged_area) >= 3:
            merged_box = (
                min(box[0] for box, _ in merged_area),
                min(box[1] for box, _ in merged_area),
                max(box[2] for box, _ in merged_area),
                max(box[3] for box, _ in merged_area)
            )
            merged_label = combined_label  # change label to 3, you can modify as needed
        elif len(merged_area) > 1:
            merged_box = (
                min(box[0] for box, _ in merged_area),
                min(box[1] for box, _ in merged_area),
                max(box[2] for box, _ in merged_area),
                max(box[3] for box, _ in merged_area)
            )
            merged_label = main_label  # retain original label
        else:
            merged_box = main_box
            merged_label = main_label

        merged_boxes_with_labels.append((merged_box, merged_label))
        boxes_with_labels = other_boxes

    merged_boxes, merged_labels = zip(*merged_boxes_with_labels)
    return list(merged_boxes), list(merged_labels), new_merge


def merge_boxes_by_labels(boxes, labels):
    """
    Merge bounding boxes based on their unique_labels.

    Parameters:
    - boxes: List of all bounding boxes in format [(labels, y1, x1, y2, x2), ...]

    Returns:
    - List of merged bounding boxes.
    """
    unique_labels = set(label for label in labels)


    new_merge = True 
    while new_merge:
        
        merged_boxes = []
        merged_labels = []
        
        for label in unique_labels:
            
            label = combine[label]
            
            boxes_of_labels, labels_of_labels = zip(*[(box, lbl) for box, lbl in zip(boxes, labels) if lbl in label])
            merged_boxes_for_label, merged_labels_for_label, new_merge = merge_boxes_for_label(boxes_of_labels, labels_of_labels, combined_label=max(label))
            
            merged_boxes.extend(merged_boxes_for_label)
            merged_labels.extend(merged_labels_for_label)
        
        boxes = merged_boxes
        labels = merged_labels

    return merged_boxes, merged_labels

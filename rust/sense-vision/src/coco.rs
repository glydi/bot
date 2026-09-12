//! The 80 COCO class names, in the index order every ultralytics YOLO export
//! (v5 and v8 alike) uses for its class scores. Embedded so the crate never
//! has to parse a sidecar `.yaml`/`.json` next to the model.

/// Index 0 is `person`, which the object path deliberately never reports:
/// people are covered by the face pipeline, which knows *who* they are.
pub const PERSON: usize = 0;

/// COCO-80 labels, ultralytics order.
pub const CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// The label for a class index, or `None` past the end.
pub fn name(class: usize) -> Option<&'static str> {
    CLASSES.get(class).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eighty_unique_names_with_the_anchors_where_ultralytics_puts_them() {
        let mut sorted = CLASSES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 80);
        assert_eq!(name(PERSON), Some("person"));
        assert_eq!(name(39), Some("bottle"));
        assert_eq!(name(56), Some("chair"));
        assert_eq!(name(67), Some("cell phone"));
        assert_eq!(name(79), Some("toothbrush"));
        assert_eq!(name(80), None);
    }
}

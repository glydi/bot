package vision

import (
	"fmt"
	"image"
	"sort"

	ort "github.com/yalue/onnxruntime_go"
)

// Face is one detection: a box in the *original* image's pixel coordinates,
// the detector confidence, and the five ArcFace landmarks (left eye, right
// eye, nose, left mouth corner, right mouth corner), also in original
// coordinates.
type Face struct {
	Box       image.Rectangle
	Score     float32
	Landmarks [5][2]float32

	// BoxF keeps the sub-pixel box, which is what insightface reports and
	// what should be compared against the Python reference.
	BoxF [4]float32
}

// SCRFD decoding constants for det_500m / det_10g (SCRFD-500MF, "2.5g"
// family): three FPN levels, two anchors per location, keypoints enabled.
const (
	scrfdInputSize   = 640
	scrfdInputMean   = 127.5
	scrfdInputStd    = 128.0
	scrfdNumAnchors  = 2
	scrfdNumKeypoint = 5
)

var scrfdStrides = [3]int{8, 16, 32}

// DetectorOptions tunes detection thresholds.
type DetectorOptions struct {
	ScoreThreshold float32 // default 0.5
	NMSThreshold   float32 // default 0.4
	InputSize      int     // default 640 (square)
}

// Detector runs SCRFD face detection.
type Detector struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	outputs []*ort.Tensor[float32]

	size    int
	scoreTh float32
	nmsTh   float32

	// anchors[i] holds the (x,y) centres for stride level i, already
	// expanded by the anchor count, in network pixel coordinates.
	anchors [3][][2]float32
	// preallocated letterbox canvas reused across frames
	canvas *RGB
}

// NewDetector loads a SCRFD ONNX model (e.g. buffalo_s/det_500m.onnx).
func NewDetector(modelPath string, opts DetectorOptions) (*Detector, error) {
	if err := InitORT(); err != nil {
		return nil, fmt.Errorf("init onnxruntime: %w", err)
	}
	size := opts.InputSize
	if size == 0 {
		size = scrfdInputSize
	}
	if size%32 != 0 {
		return nil, fmt.Errorf("input size %d must be a multiple of 32", size)
	}
	d := &Detector{
		size:    size,
		scoreTh: opts.ScoreThreshold,
		nmsTh:   opts.NMSThreshold,
		canvas:  NewRGB(size, size),
	}
	if d.scoreTh == 0 {
		d.scoreTh = 0.5
	}
	if d.nmsTh == 0 {
		d.nmsTh = 0.4
	}

	inputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 3, int64(size), int64(size)))
	if err != nil {
		return nil, err
	}
	d.input = inputTensor

	// Output order in the graph: 3 score maps, 3 bbox maps, 3 kps maps.
	outNames := []string{"443", "468", "493", "446", "471", "496", "449", "474", "499"}
	widths := []int64{1, 1, 1, 4, 4, 4, 10, 10, 10}
	outValues := make([]ort.Value, len(outNames))
	for i, stride := range []int{8, 16, 32, 8, 16, 32, 8, 16, 32} {
		n := int64((size / stride) * (size / stride) * scrfdNumAnchors)
		t, err := ort.NewEmptyTensor[float32](ort.NewShape(n, widths[i]))
		if err != nil {
			return nil, err
		}
		d.outputs = append(d.outputs, t)
		outValues[i] = t
	}

	sess, err := ort.NewAdvancedSession(modelPath,
		[]string{"input.1"}, outNames,
		[]ort.Value{inputTensor}, outValues, nil)
	if err != nil {
		return nil, fmt.Errorf("create scrfd session: %w", err)
	}
	d.session = sess

	for i, stride := range scrfdStrides {
		h, w := size/stride, size/stride
		centers := make([][2]float32, 0, h*w*scrfdNumAnchors)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				// Two anchors share a location; insightface stacks them
				// so consecutive rows repeat the same centre.
				for a := 0; a < scrfdNumAnchors; a++ {
					centers = append(centers, [2]float32{
						float32(x * stride), float32(y * stride),
					})
				}
			}
		}
		d.anchors[i] = centers
	}
	return d, nil
}

// Close releases the ONNX session and tensors.
func (d *Detector) Close() error {
	if d.session != nil {
		d.session.Destroy()
		d.session = nil
	}
	if d.input != nil {
		d.input.Destroy()
		d.input = nil
	}
	for _, t := range d.outputs {
		t.Destroy()
	}
	d.outputs = nil
	return nil
}

// letterbox resizes src so that it fits into size x size while preserving
// aspect ratio, pastes it at the top-left of a zero canvas (exactly what
// insightface does — the padding is *not* centred) and returns the scale
// factor applied.
func (d *Detector) letterbox(src *RGB) float32 {
	imRatio := float64(src.H) / float64(src.W)
	var nw, nh int
	if imRatio > 1.0 { // model ratio is 1.0 for a square input
		nh = d.size
		nw = int(float64(nh) / imRatio)
	} else {
		nw = d.size
		nh = int(float64(nw) * imRatio)
	}
	scale := float32(nh) / float32(src.H)
	resized := resizeBilinear(src, nw, nh)
	for i := range d.canvas.Pix {
		d.canvas.Pix[i] = 0
	}
	for y := 0; y < nh; y++ {
		copy(d.canvas.Pix[y*d.size*3:y*d.size*3+nw*3], resized.Pix[y*nw*3:(y+1)*nw*3])
	}
	return scale
}

// fillBlob writes the canvas into the NCHW float input tensor using
// (pixel - 127.5) / 128.0 on RGB channels, matching cv2.dnn.blobFromImage
// with swapRB=True on a BGR source.
func (d *Detector) fillBlob() {
	data := d.input.GetData()
	plane := d.size * d.size
	for i := 0; i < plane; i++ {
		p := i * 3
		data[i] = (float32(d.canvas.Pix[p]) - scrfdInputMean) / scrfdInputStd
		data[plane+i] = (float32(d.canvas.Pix[p+1]) - scrfdInputMean) / scrfdInputStd
		data[2*plane+i] = (float32(d.canvas.Pix[p+2]) - scrfdInputMean) / scrfdInputStd
	}
}

// Detect runs the detector on img and returns faces in original-image
// coordinates, sorted by descending score (post-NMS order).
func (d *Detector) Detect(img image.Image) ([]Face, error) {
	return d.DetectRGB(FromImage(img))
}

// DetectRGB is Detect without the image.Image conversion.
func (d *Detector) DetectRGB(src *RGB) ([]Face, error) {
	if src.W == 0 || src.H == 0 {
		return nil, fmt.Errorf("empty image")
	}
	scale := d.letterbox(src)
	d.fillBlob()
	if err := d.session.Run(); err != nil {
		return nil, fmt.Errorf("scrfd run: %w", err)
	}

	var cands []Face
	for i, stride := range scrfdStrides {
		scores := d.outputs[i].GetData()
		bboxes := d.outputs[i+3].GetData()
		kps := d.outputs[i+6].GetData()
		centers := d.anchors[i]
		fs := float32(stride)
		for n := 0; n < len(centers); n++ {
			s := scores[n]
			if s < d.scoreTh {
				continue
			}
			cx, cy := centers[n][0], centers[n][1]
			b := bboxes[n*4 : n*4+4]
			// distance2bbox: l, t, r, b distances from the anchor centre,
			// expressed in stride units.
			f := Face{
				Score: s,
				BoxF: [4]float32{
					(cx - b[0]*fs) / scale,
					(cy - b[1]*fs) / scale,
					(cx + b[2]*fs) / scale,
					(cy + b[3]*fs) / scale,
				},
			}
			// distance2kps: per-landmark dx, dy offsets, also in stride units.
			k := kps[n*10 : n*10+10]
			for j := 0; j < scrfdNumKeypoint; j++ {
				f.Landmarks[j][0] = (cx + k[j*2]*fs) / scale
				f.Landmarks[j][1] = (cy + k[j*2+1]*fs) / scale
			}
			cands = append(cands, f)
		}
	}
	if len(cands) == 0 {
		return nil, nil
	}
	// Stable sort by descending score, matching numpy's stable argsort.
	sort.SliceStable(cands, func(a, b int) bool { return cands[a].Score > cands[b].Score })
	kept := nms(cands, d.nmsTh)
	for i := range kept {
		kept[i].Box = image.Rect(
			int(kept[i].BoxF[0]+0.5), int(kept[i].BoxF[1]+0.5),
			int(kept[i].BoxF[2]+0.5), int(kept[i].BoxF[3]+0.5))
	}
	return kept, nil
}

// nms implements insightface's greedy IoU suppression. Note the "+1" in the
// area and overlap terms — it is part of the reference implementation and
// changing it shifts results slightly.
func nms(dets []Face, thresh float32) []Face {
	areas := make([]float32, len(dets))
	for i, d := range dets {
		areas[i] = (d.BoxF[2] - d.BoxF[0] + 1) * (d.BoxF[3] - d.BoxF[1] + 1)
	}
	suppressed := make([]bool, len(dets))
	out := make([]Face, 0, len(dets))
	for i := range dets {
		if suppressed[i] {
			continue
		}
		out = append(out, dets[i])
		for j := i + 1; j < len(dets); j++ {
			if suppressed[j] {
				continue
			}
			xx1 := max32(dets[i].BoxF[0], dets[j].BoxF[0])
			yy1 := max32(dets[i].BoxF[1], dets[j].BoxF[1])
			xx2 := min32(dets[i].BoxF[2], dets[j].BoxF[2])
			yy2 := min32(dets[i].BoxF[3], dets[j].BoxF[3])
			w := max32(0, xx2-xx1+1)
			h := max32(0, yy2-yy1+1)
			inter := w * h
			if inter/(areas[i]+areas[j]-inter) > thresh {
				suppressed[j] = true
			}
		}
	}
	return out
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

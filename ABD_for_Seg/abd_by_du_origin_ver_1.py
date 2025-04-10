import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cosine
import cv2

def extract_frame_features(video_path, grid_size=(4, 4), bins=8, frame_skip=2, resize_wh=(64, 64)):
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    hof_features = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        resized = cv2.resize(frame, resize_wh)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
            h, w = gray.shape
            cell_h, cell_w = h // grid_size[0], w // grid_size[1]
            hof_vec = []

            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    y1, y2 = i * cell_h, (i + 1) * cell_h
                    x1, x2 = j * cell_w, (j + 1) * cell_w
                    angle = ang[y1:y2, x1:x2].flatten()
                    magnitude = mag[y1:y2, x1:x2].flatten()
                    hist, _ = np.histogram(angle, bins=bins, range=(0, 360), weights=magnitude)
                    hof_vec.extend(hist)

            hof_features.append(hof_vec)

        prev_gray = gray
        frame_idx += 1

    cap.release()
    return np.array(hof_features), frame_skip

def smooth_features(features, sigma=2):
    return gaussian_filter1d(features, sigma=sigma, axis=0)

def cosine_sim(a, b):
    return 1 - cosine(a, b)

def compute_similarities(smoothed):
    return [cosine_sim(smoothed[i], smoothed[i+1]) for i in range(len(smoothed)-1)]

def detect_boundaries(similarities, window_size=10, min_boundaries=30):
    boundaries = []
    for i in range(window_size, len(similarities) - window_size):
        local = similarities[i - window_size:i + window_size + 1]
        if similarities[i] == min(local):
            boundaries.append(i)

    if len(boundaries) < min_boundaries:
        total = len(similarities)
        step = total // (min_boundaries + 1)
        boundaries += list(range(step, total, step))

    return sorted(set(boundaries))

def segment_from_boundaries(boundaries, total_length):
    boundaries = [0] + boundaries + [total_length]
    return [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

def visualize_segments(similarities, segments):
    plt.figure(figsize=(14, 5))
    plt.plot(similarities, label="Cosine Similarity", color="blue")
    cmap = cm.get_cmap("Set3", len(segments))
    for i, (start, end) in enumerate(segments):
        plt.axvspan(start, end, alpha=0.3, color=cmap(i), label=f"Seg {i+1}")
    plt.title("DU Segments with Cosine Similarity")
    plt.xlabel("Feature Index")
    plt.ylabel("Similarity")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def run_abd_safe(video_path, sigma=2, window_size=10, min_boundaries=30, frame_skip=2):
    features, frame_skip_used = extract_frame_features(video_path, frame_skip=frame_skip)
    smoothed = smooth_features(features, sigma=sigma)
    similarities = compute_similarities(smoothed)
    boundaries = detect_boundaries(similarities, window_size=window_size, min_boundaries=min_boundaries)
    segments = segment_from_boundaries(boundaries, len(features))

    print(f"\\n[✔] Final {len(segments)} segments (based on frame_skip={frame_skip_used})")
    fps = 120  # 💡 너희 영상에 맞게 설정!

    for i, (start, end) in enumerate(segments):
        real_start = start * frame_skip
        real_end = end * frame_skip
        duration = (real_end - real_start) / fps
        print(f"Segment {i+1}: Frame {real_start} → {real_end} ({duration:.2f} sec)")

    visualize_segments(similarities, segments)
    return segments

if __name__ == "__main__":
    video_path = "/Users/heejeong/vscode/TimeMachine/CPR_data/CPR_frontview.mp4"
    run_abd_safe(video_path,sigma=2,window_size=10, avg_action_length=30)
    """ run_abd(
    video_path=video_path,
    fast_mode=True,           # 🔥 빠른 실행용
    temporal_weight=0.4,
    tau=10,
    sigma=2,
    window_size=10,
    auto_K=True
) """
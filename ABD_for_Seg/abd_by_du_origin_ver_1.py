#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cosine
import cv2
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import json

def export_du_segments_to_json(segments, output_path="du_segments.json"):
    export_list = [[start, end] for start, end in segments]
    with open(output_path, "w") as f:
        json.dump(export_list, f, indent=2)
    print(f"✔️ Exported {len(segments)} DU segments to {output_path}")

#############################
# Feature Extraction
#############################
def extract_frame_features(video_path, grid_size=(4, 4), bins=8, frame_skip=2, resize_wh=(64, 64)):
    """
    영상에서 HOF(Optical Flow Histogram) feature를 추출한다.
    
    Args:
        video_path: 입력 영상 파일 경로
        grid_size: 영상 그리드 분할 (행, 열) – 각 그리드의 히스토그램을 추출
        bins: 각 그리드에서 사용할 히스토그램 bin 개수 (각도 0~360)
        frame_skip: 몇 프레임마다 처리할지 (속도 조절용)
        resize_wh: 영상의 리사이즈 크기 (너비, 높이)
    
    Returns:
        hof_features: (N x D) numpy array, N은 추출된 frame 수, D는 feature 차원
        frame_skip: 실제로 사용된 frame_skip 값
    """
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

#############################
# Smoothing and Similarity
#############################
def smooth_features(features, sigma=2):
    """
    Gaussian 필터를 이용하여 프레임별 feature에 대해 smoothing 수행.
    
    Args:
        features: (N x D) numpy array, frame별 feature
        sigma: Gaussian smoothing에 사용될 sigma 값
    
    Returns:
        smoothed_features: (N x D) numpy array
    """
    return gaussian_filter1d(features, sigma=sigma, axis=0)

def cosine_sim(a, b):
    """
    두 벡터 간 cosine similarity를 계산 (scipy.spatial.distance.cosine은 cosine distance 리턴하므로 변환)
    """
    return 1 - cosine(a, b)

def compute_similarities(smoothed):
    """
    연속된 프레임 간 cosine similarity를 계산.
    
    Args:
        smoothed: (N x D) numpy array, smoothing된 feature
    
    Returns:
        similarities: 리스트 (길이: N-1) cosine similarity 값
    """
    return [cosine_sim(smoothed[i], smoothed[i+1]) for i in range(len(smoothed)-1)]

#############################
# Boundary Detection & Segmentation
#############################
def detect_boundaries(similarities, window_size=10, min_boundaries=30):
    """
    cosine similarity 시퀀스에 대해 NMS 기반 경계 검출.
    또한, 경계 개수가 min_boundaries 미만이면 일정 간격으로 추가.
    
    Args:
        similarities: 1차원 리스트 혹은 array, 연속 프레임 간 cosine similarity
        window_size: 로컬 윈도우 크기
        min_boundaries: 최소 요구 경계 수 (너무 적은 경우 보완)
    
    Returns:
        boundaries: 경계 인덱스 리스트 (프레임 index 기준)
    """
    boundaries = []
    for i in range(window_size, len(similarities) - window_size):
        local = similarities[i - window_size : i + window_size + 1]
        if similarities[i] == min(local):
            boundaries.append(i)
            
    if len(boundaries) < min_boundaries:
        total = len(similarities)
        step = total // (min_boundaries + 1)
        extra = list(range(step, total, step))
        boundaries = list(set(boundaries + extra))
    
    return sorted(boundaries)

def segment_from_boundaries(boundaries, total_length):
    """
    경계 리스트를 바탕으로 segment를 생성.
    
    Args:
        boundaries: 경계 인덱스 리스트 (예: [b1, b2, ..., bn])
        total_length: 전체 frame 수
    
    Returns:
        segments: 각 segment를 (start, end) 튜플로 나타낸 리스트.
                  시작과 끝 경계는 자동으로 포함시킴.
    """
    boundaries = [0] + boundaries + [total_length]
    segments = []
    for i in range(len(boundaries)-1):
        segments.append( (boundaries[i], boundaries[i+1]) )
    return segments

def visualize_segments(similarities, segments):
    """
    similarity curve와 segmentation 결과를 시각화.
    
    Args:
        similarities: cosine similarity 시퀀스 (1D array)
        segments: (start, end) 튜플 리스트
    """
    plt.figure(figsize=(14, 5))
    plt.plot(similarities, label="Cosine Similarity", color="blue")
    cmap = cm.get_cmap("Set3", len(segments))
    for i, (start, end) in enumerate(segments):
        plt.axvspan(start, end, alpha=0.3, color=cmap(i), label=f"Seg {i+1}")
    plt.title("Action Boundary Detection Visualization")
    plt.xlabel("Frame Index")
    plt.ylabel("Cosine Similarity")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

#############################
# Refinement (Merge Over-segmentation)
#############################
def segment_feature_average(features, boundaries):
    """
    후보 boundary를 이용해 각 segment의 평균 feature를 계산.
    
    Args:
        features: (N x D) numpy array, smoothed features
        boundaries: 경계 인덱스 리스트 (예: [0, b1, b2, ..., N])
    
    Returns:
        segments: (num_segments x D) numpy array, 각 segment의 평균 feature
    """
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        if end > start:
            seg_feat = np.mean(features[start:end], axis=0)
        else:
            seg_feat = features[start]
        segments.append(seg_feat)
    return np.array(segments)

def refine_segments_adjacent(features, boundaries, K):
    """
    인접한 segment끼리만 병합하여 over-segmentation을 줄이고 최종적으로 K개의 segment로 만든다.
    (비인접 segment 병합 시 시간 순서가 꼬이는 문제를 방지)
    
    Args:
        features: (N x D) numpy array, smoothed features
        boundaries: 경계 인덱스 리스트 (예: [0, b1, b2, ..., N])
        K: 최종 목표 segment 수
        
    Returns:
        seg_boundaries: 각 segment의 (start, end) index 튜플 리스트, 총 K개
        segments: 최종 segment들의 feature 리스트 (K x D)
    """
    # 각 segment의 평균 feature 계산
    segment_feats = segment_feature_average(features, boundaries)
    segments = list(segment_feats)
    seg_boundaries = list(zip(boundaries[:-1], boundaries[1:]))
    
    # 오직 인접한 segment끼리만 병합
    while len(segments) > K:
        max_sim = -np.inf
        merge_index = -1
        for i in range(len(segments)-1):
            sim = cosine_sim(segments[i], segments[i+1])
            if sim > max_sim:
                max_sim = sim
                merge_index = i
        # 인접한 두 segment 병합
        merged_feat = (segments[merge_index] + segments[merge_index+1]) / 2
        merged_boundary = (seg_boundaries[merge_index][0], seg_boundaries[merge_index+1][1])
        segments[merge_index] = merged_feat
        seg_boundaries[merge_index] = merged_boundary
        segments.pop(merge_index+1)
        seg_boundaries.pop(merge_index+1)
    

    return seg_boundaries, segments

#############################
# Main Pipeline
#############################
def run_abd_safe(video_path, sigma=2, window_size=10, min_boundaries=30, frame_skip=2, K=0, fps=60):
    """
    영상 파일을 입력으로 받아 ABD 파이프라인을 수행한다.
    1. HOF 기반 feature 추출
    2. Gaussian smoothing
    3. 프레임 간 cosine similarity 계산
    4. NMS 기반 경계 검출 (유사도가 낮은 지점)
    5. 경계로부터 초기 segmentation 생성
    6. (옵션) 목표 segment 수 K가 주어지면 refinement 수행 (인접한 segment 병합)
    7. 결과 시각화 및 segment 정보 출력
    
    Args:
        video_path: 입력 영상 경로
        sigma: Gaussian smoothing의 sigma 값
        window_size: boundary 검출 시 사용할 윈도우 크기
        min_boundaries: 최소 경계 개수 (부족할 경우 보완)
        frame_skip: feature 추출 시 건너뛸 frame 간격
        K: 최종 목표 segment 수 (0이면 refinement 수행하지 않음)
        fps: 영상의 초당 프레임 수 (segment duration 계산용)
    
    Returns:
        segments: (start, end) 튜플 리스트, 각 segment의 시작/끝 frame index
    """
    # 1. Feature 추출
    features, frame_skip_used = extract_frame_features(video_path, frame_skip=frame_skip)
    print(f"[✔] {len(features)} frames의 feature 추출 완료 (frame_skip={frame_skip_used})")
    
    # 2. Gaussian smoothing
    smoothed_features = smooth_features(features, sigma=sigma)
    
    # 3. 연속 프레임 간 cosine similarity 계산
    similarities = compute_similarities(smoothed_features)
    
    # 4. 경계 검출 (similarities 기반)
    boundaries = detect_boundaries(similarities, window_size=window_size, min_boundaries=min_boundaries)
    print(f"[✔] 검출된 경계 후보 (similarity index 기준): {boundaries}")
    
    # 5. 초기 segmentation (boundary 리스트를 이용하여)
    segments_init = segment_from_boundaries(boundaries, total_length=len(features))
    
    # 6. (옵션) over-segmentation refinement: 인접 segment 병합 방식 적용
    if K > 0 and len(segments_init) > K:
        # refine_segments_adjacent를 위해 boundaries 리스트 재구성
        init_boundaries = [b for b, _ in segments_init] + [len(features)]
        refined_boundaries, _ = refine_segments_adjacent(smoothed_features, init_boundaries, K)
        segments_final = refined_boundaries
        print(f"[✔] Refinement 수행 후 최종 segment 수: {len(segments_final)}")
    else:
        segments_final = segments_init
        print(f"[✔] Refinement 미수행 (초기 segment 수: {len(segments_final)})")
    

    # 7. Segment 정보 출력 (frame_skip 반영, 시간 순 정렬 포함)
    segments_final = sorted(segments_final, key=lambda x: x[0])  # 🔥 정렬 추가

    print("\n[Segment Info]")
    for i, (start, end) in enumerate(segments_final):
        real_start = start * frame_skip_used
        real_end = end * frame_skip_used
        duration = (real_end - real_start) / fps
        print(f"Segment {i+1}: Frame {real_start} \u2192 {real_end} ({duration:.2f} sec)")
    # 8. 시각화
    visualize_segments(similarities, segments_final)

    export_du_segments_to_json(segments_final)
    
    return segments_final

#############################
# Argument Parsing & Main()
#############################
def main():
    parser = argparse.ArgumentParser(
        description="Fast and Unsupervised Action Boundary Detection (ABD)"
    )
    parser.add_argument('--video_path', type=str, default="ABD_for_CPR/CPR_data/CPR_frontview.mp4",
                        help="입력 영상 파일 경로")
    parser.add_argument('--sigma', type=float, default=2.0,
                        help="Gaussian smoothing에 사용할 sigma 값")
    parser.add_argument('--window_size', type=int, default=10,
                        help="boundary 검출을 위한 윈도우 크기")
    parser.add_argument('--min_boundaries', type=int, default=30,
                        help="최소 경계 개수 (부족할 경우 보완)")
    parser.add_argument('--frame_skip', type=int, default=2,
                        help="feature 추출 시 건너뛸 frame 간격")
    parser.add_argument('--K', type=int, default=15,
                        help="최종 목표 segment 수 (0이면 refinement 수행하지 않음)")
    parser.add_argument('--fps', type=float, default=60,
                        help="영상의 초당 프레임 수 (duration 계산용)")
    args = parser.parse_args()
    
    run_abd_safe(video_path=args.video_path, sigma=args.sigma, window_size=args.window_size,
                 min_boundaries=args.min_boundaries, frame_skip=args.frame_skip, K=args.K, fps=args.fps)

if __name__ == "__main__":
    main()

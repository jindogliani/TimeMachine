#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

#############################
# Feature Extraction
#############################
def extract_frame_features(video_path, grid_size=(4, 4), bins=8, frame_skip=2, resize_wh=(64, 64)):
    """
    영상에서 HOF (Optical Flow Histogram) feature를 추출한다.
    
    Args:
        video_path: 입력 영상 파일 경로
        grid_size: 영상 그리드 분할 (행, 열)
        bins: 각 그리드에서 사용할 히스토그램 bin 개수 (각도 0~360)
        frame_skip: 몇 프레임마다 처리할지 (속도 조절용)
        resize_wh: 영상 리사이즈 크기 (너비, 높이)
    
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
# Smoothing and Similarity Computation
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
    """두 벡터 간 cosine similarity 계산 (cosine distance를 1 - distance로 변환)"""
    return 1 - cosine(a, b)

def compute_similarities(smoothed):
    """
    연속된 프레임 간 cosine similarity를 계산.
    
    Args:
        smoothed: (N x D) numpy array, smoothing된 feature
    
    Returns:
        similarities: 길이 N-1의 리스트, 각 항목은 두 인접 프레임 간 similarity 값
    """
    return [cosine_sim(smoothed[i], smoothed[i+1]) for i in range(len(smoothed)-1)]

#############################
# Boundary Detection & Segmentation
#############################
def detect_boundaries(similarities, window_size=10):
    """
    cosine similarity 시퀀스에 대해 local minimum (NMS) 기반 경계 검출.
    
    Args:
        similarities: 1차원 리스트 혹은 array, 연속 프레임 간 cosine similarity
        window_size: 로컬 윈도우 크기 (예: 10)
    
    Returns:
        boundaries: 경계 인덱스 리스트 (프레임 index 기준)
    """
    boundaries = []
    half_win = window_size // 2
    for i in range(half_win, len(similarities) - half_win):
        local_window = similarities[i - half_win : i + half_win + 1]
        if similarities[i] == min(local_window):
            boundaries.append(i + 1)  # 경계는 보통 t+1 프레임에서 분할됨
    return sorted(boundaries)

def segment_from_boundaries(boundaries, total_length):
    """
    경계 리스트를 바탕으로 segment (각 구간의 시작, 끝 인덱스)를 생성.
    
    Args:
        boundaries: 경계 인덱스 리스트 (예: [b1, b2, ..., bn])
        total_length: 전체 frame 수
    
    Returns:
        segments: (start, end) 튜플 리스트. 시작과 끝 경계는 자동 포함.
    """
    boundaries = [0] + boundaries + [total_length]
    segments = []
    for i in range(len(boundaries)-1):
        segments.append((boundaries[i], boundaries[i+1]))
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
# Global Refinement (Bottom-up Clustering)
#############################
def segment_feature_average(features, boundaries):
    """
    후보 경계에 따라 각 세그먼트의 평균 feature를 계산.
    
    Args:
        features: (N x D) numpy array, smoothing된 feature
        boundaries: 경계 인덱스 리스트 (예: [0, b1, b2, ..., N])
    
    Returns:
        segments: (num_segments x D) numpy array, 각 세그먼트의 평균 feature
    """
    seg_feats = []
    for i in range(len(boundaries)-1):
        start, end = boundaries[i], boundaries[i+1]
        if end > start:
            seg_feat = np.mean(features[start:end], axis=0)
        else:
            seg_feat = features[start]
        seg_feats.append(seg_feat)
    return np.array(seg_feats)

def refine_segments_global(features, boundaries, K):
    """
    전역 병합 알고리즘(논문의 bottom-up 클러스터링)을 사용하여 후보 세그먼트(global segmentation)를 
    최종 목표 segment 수 K로 줄인다.
    
    Args:
        features: (N x D) numpy array, smoothing된 feature
        boundaries: 전체 경계 리스트, 즉 [0, b1, b2, ..., N]
        K: 최종 목표 segment 수
        
    Returns:
        seg_boundaries: 최종 세그먼트의 (start, end) 튜플 리스트 (길이 K)
        segments: 최종 세그먼트의 feature 리스트 (K x D)
    """
    # 초기 세그먼트 feature 계산
    seg_feats = segment_feature_average(features, boundaries)
    segments = list(seg_feats)  # 각 세그먼트 feature (list of vectors)
    seg_boundaries = list(zip(boundaries[:-1], boundaries[1:]))  # 각 세그먼트의 (start, end)
    
    # 전역적으로 가장 유사한 두 세그먼트를 찾아 병합 (논문의 Algorithm 1)
    while len(segments) > K:
        sim_matrix = cosine_similarity(segments)
        np.fill_diagonal(sim_matrix, -1)  # 자기 자신 제외
        i, j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
        # 병합: 두 세그먼트 feature의 평균을 사용
        merged_feat = (segments[i] + segments[j]) / 2
        merged_boundary = (min(seg_boundaries[i][0], seg_boundaries[j][0]),
                           max(seg_boundaries[i][1], seg_boundaries[j][1]))
        # 업데이트: 인덱스 순서를 맞춘 후 i에 병합 결과, j는 삭제
        if i > j:
            i, j = j, i
        segments[i] = merged_feat
        seg_boundaries[i] = merged_boundary
        segments.pop(j)
        seg_boundaries.pop(j)
    return seg_boundaries, segments

#############################
# Main Pipeline
#############################
def run_abd(video_path, sigma=2, window_size=10, frame_skip=2, K=0, fps=60):
    """
    영상 파일을 입력받아 ABD 파이프라인을 수행한다.
    1. HOF 기반 feature 추출
    2. Gaussian smoothing
    3. 프레임 간 cosine similarity 계산
    4. Local minimum 기반 NMS로 후보 경계 검출
    5. 후보 경계로부터 초기 세그먼트 생성
    6. (옵션) 목표 segment 수 K > 0이면 전역 병합 refinement 수행
    7. 결과 시각화 및 세그먼트 정보 출력
    
    Args:
        video_path: 입력 영상 파일 경로
        sigma: Gaussian smoothing의 sigma 값
        window_size: NMS에 사용할 로컬 윈도우 크기
        frame_skip: feature 추출 시 건너뛸 frame 간격
        K: 최종 목표 segment 수 (0이면 refinement 미수행)
        fps: 영상의 초당 프레임 수 (duration 계산용)
    
    Returns:
        segments: (start, end) 튜플 리스트 (최종 세그먼트, frame 단위)
    """
    # 1. Feature 추출
    features, frame_skip_used = extract_frame_features(video_path, frame_skip=frame_skip)
    print(f"[✔] {len(features)} frames의 feature 추출 완료 (frame_skip={frame_skip_used})")
    
    # 2. Gaussian smoothing
    smoothed_features = smooth_features(features, sigma=sigma)
    
    # 3. 프레임 간 cosine similarity 계산
    similarities = compute_similarities(smoothed_features)
    
    # 4. 후보 경계 검출
    cand_boundaries = detect_boundaries(similarities, window_size=window_size)
    print(f"[✔] 검출된 경계 후보 (similarity index 기준): {cand_boundaries}")
    
    # 5. 초기 세그먼트 구성: 전체 boundary 리스트 = [0] + 후보 + [total_frames]
    total_frames = len(features)
    init_boundaries = [0] + cand_boundaries + [total_frames]
    segments_init = segment_from_boundaries(cand_boundaries, total_frames)
    print(f"[✔] 초기 세그먼트 수: {len(segments_init)}")
    
    # 6. Refinement (전역 병합) 수행: K > 0이면 목표 세그먼트 수로 축소
    if K > 0 and len(segments_init) > K:
        refined_boundaries, _ = refine_segments_global(smoothed_features, init_boundaries, K)
        segments_final = refined_boundaries
        print(f"[✔] Refinement 수행 후 최종 세그먼트 수: {len(segments_final)}")
    else:
        segments_final = segments_init
        print(f"[✔] Refinement 미수행 (세그먼트 수: {len(segments_final)})")
    
    # 7. 결과 출력 (실제 frame index: frame_skip 반영)
    print("\n[Segment Info]")
    for idx, (start, end) in enumerate(segments_final):
        real_start = start * frame_skip_used
        real_end = end * frame_skip_used
        duration = (real_end - real_start) / fps
        print(f"Segment {idx+1}: Frame {real_start} → {real_end} ({duration:.2f} sec)")
    
    # 8. 시각화
    visualize_segments(similarities, segments_final)
    
    return segments_final

#############################
# Argument Parsing & Main()
#############################
def main():
    parser = argparse.ArgumentParser(
        description="Fast and Unsupervised Action Boundary Detection (ABD) - Global Merging Version"
    )
    parser.add_argument('--video_path', type=str, default="ABD_for_CPR/CPR_data/CPR_frontview.mp4",
                        help="입력 영상 파일 경로")
    parser.add_argument('--sigma', type=float, default=2.0,
                        help="Gaussian smoothing에 사용할 sigma 값")
    parser.add_argument('--window_size', type=int, default=10,
                        help="NMS 경계 검출을 위한 윈도우 크기")
    parser.add_argument('--frame_skip', type=int, default=2,
                        help="Feature 추출 시 건너뛸 frame 간격")
    parser.add_argument('--K', type=int, default=15,
                        help="최종 목표 segment 수 (0이면 refinement 미수행)")
    parser.add_argument('--fps', type=float, default=60,
                        help="영상의 초당 프레임 수 (duration 계산용)")
    args = parser.parse_args()
    
    run_abd(video_path=args.video_path, sigma=args.sigma, window_size=args.window_size,
            frame_skip=args.frame_skip, K=args.K, fps=args.fps)

if __name__ == "__main__":
    main()

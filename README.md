# TimeMachine
시간 기반 동작 분석 및 씬그래프 생성 프로젝트

## Setup
이 프로젝트에서 필요한 모든 패키지를 설치하려면 다음 명령어를 사용하세요:
```bash
pip install -r requirements.txt
```

## 시작하기 전에
- `ABD_for_Seg/abd_by_du_origin_ver_1.py`를 사용하려면 `CPR_data/`에 `.mp4`, `.bvh` 파일을 저장하세요.
- `Text-to-SG/0_text_to_gpt.py`를 사용하려면 `Open_AI_api_key.txt` 파일에 OpenAI API 키를 저장하세요.

## 스크립트 사용 순서
### ABD 액션 세분화 (Action Segmentation)
1. **`ABD_for_Seg/abd_by_du_origin_ver_1.py`**: 액션 경계 감지(ABD) 알고리즘 실행
   ```bash
   python ABD_for_Seg/abd_by_du_origin_ver_1.py --video_path CPR_data/CPR_frontview.mp4
   ```
   - 결과: 영상 내 액션 세그먼트 생성 (`ABD_for_CPR/du_segments.json`)

### 텍스트-그래프 생성 (Text-to-Graph)
1. **`Text-to-SG/0_text_to_gpt.py`**: ChatGPT를 사용한 텍스트 처리
   ```bash
   python Text-to-SG/0_text_to_gpt.py
   ```
   - 결과: 각 세그먼트에 대한 텍스트 설명 생성

2. **`Text-to-SG/1_text_to_graph_postprocess.py`**: 씬 그래프 후처리
   ```bash
   python Text-to-SG/1_text_to_graph_postprocess.py
   ```
   - 결과: 최종 씬 그래프 생성

## 파일 정보
- **영상 파일**:
  - CPR_frontview.mp4 (11,391 프레임, 58.68 FPS, 194초)
  - CPR_sideview.mp4 (12,600 프레임, 60.01 FPS, 209초)

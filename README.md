# TimeMachine
TimeMachine

## Setup
이 프로젝트에서 필요한 모든 패키지를 설치하려면 다음 명령어를 사용하세요:
```bash
pip install -r requirements.txt
```

## 시작하기 전에
`ABD_for_Seg/abd_by_du_origin_ver_1.py`를 사용하려면 `CPR_data/*` 디렉토리에 `.mp4`, `.bvh`파일을 저장하세요.
`Text-to-SG/0_text_to_gpt.py`를 사용하려면 `Open_AI_api_key.txt` 파일에 OpenAI API 키를 저장하세요.

## 스크립트 사용 순서
### 1. ABD Action Segmentation
1-1. ABD_for_Seg/abd_by_du_origin_ver_1.py: 액션 경계 감지(ABD) 알고리즘

### 2. Text-to-Graph Generation
2-1. Text-to-SG/text_to_gpt.py: Chat GPT를 사용한 텍스트 처리
2-2. Text-to-SG/text_to_graph_postprocess.py: 씬 그래프 후처리

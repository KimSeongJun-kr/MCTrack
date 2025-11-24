
## 1. MCTrack 퀵 스타트 가이드

### (1) 환경 세팅

1. 도커 이미지 빌드

```bash
cd MCTrack && docker build -f Dockerfile -t mctrack:v0.1 .
```

2. 컨테이너 실행

```bash
docker run -it --shm-size=16g \
    -p 8081:8081 \
    -v "{MCTrack 폴더 경로}:/3dmot_ws/MCTrack" \
    -v "{MCTrack 데이터 폴더 경로}:/3dmot_ws/MCTrack/data" \
    -v "{nuscenes 데이터 폴더 경로}:/3dmot_ws/MCTrack/data/nuscenes/datasets" \
    -v "{prsys_results 폴더 경로}:/3dmot_ws/MCTrack/prsys_results" \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    --name mctrack0 \
    mctrack:v0.1 /bin/bash
```

참고: 중괄호로 표시된 경로들은 실제 로컬 경로로 치환해 주세요.

### (2) 트래킹 실행
#### 방법 1. 직접 실행
1. 디텍션 결과 데이터 전처리

```bash
python /3dmot_ws/MCTrack/preprocess/convert2baseversion.py \
  --dets_file_path {디텍션 결과 파일 경로} \
  --save_folder_path {전처리 데이터 저장 폴더 경로} \
  --split {val/test/train/mini/mini_train/mini_val}
```

2. 트래킹 실행

```bash
python /3dmot_ws/MCTrack/main.py \
  -p {사용할 cpu 코어 개수} \
  --dets_folder_path {전처리 데이터 폴더 경로} \
  --save_folder_path {트래킹 결과 데이터 저장 폴더 경로} \
  --split {val/test/train/mini/mini_train/mini_val}
```

#### 방법 2. 자동 실행
```bash
bash /3dmot_ws/MCTrack/run_tracking.sh \
  --run_name {작업 명칭} \
  --timestamp {time(YYYYMMDD_hhmmss)} \
  --step {N} \
  --processors {N} \
  --split {val/test/train/mini/mini_train/mini_val} \
  --dets_file_path {디텍션 결과 파일 경로}
```
example
```bash
bash /3dmot_ws/MCTrack/run_tracking.sh \
  --run_name test_1 \
  --timestamp 20251104_174100 \
  --step 0 \
  --processors 20 \
  --split val \
  --dets_file_path prsys_results/20251104_174100/step_0/detection_result.json
```

### (3) 시각화 방법
#### 타입 1. scene 별 디텍션-트래킹 결과 확인
```bash
python seongjun_tools/visualize_boxes_det_trk.py \
  --detection_json {디텍션 결과 파일 경로 (Blue Color)} \
  --tracking_json {트래킹 결과 파일 경로 (Red Color)} \
  --output_dir {이미지를 저장할 폴더 경로} \
  --scene_name {시각화할 Scene Name}
```

#### 타입 2. scene 별 디텍션1-디텍션2 결과 확인

```bash
python seongjun_tools/visualize_boxes_det_det.py \
  --detection_json1 {디텍션1 결과 파일 경로 (Blue Color)} \
  --detection_json2 {디텍션2 결과 파일 경로 (Red Color)} \
  --output_dir {이미지를 저장할 폴더 경로} \
  --scene_name {시각화할 Scene Name}
```

#### 참고자료 - val split의 scene_name 목록

```bash
val = \
    ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
     'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
     'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
     'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
     'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
     'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
     'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']
```

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

test = \
    ['scene-0077', 'scene-0078', 'scene-0079', 'scene-0080', 'scene-0081', 'scene-0082', 'scene-0083', 'scene-0084',
     'scene-0085', 'scene-0086', 'scene-0087', 'scene-0088', 'scene-0089', 'scene-0090', 'scene-0091', 'scene-0111',
     'scene-0112', 'scene-0113', 'scene-0114', 'scene-0115', 'scene-0116', 'scene-0117', 'scene-0118', 'scene-0119',
     'scene-0140', 'scene-0142', 'scene-0143', 'scene-0144', 'scene-0145', 'scene-0146', 'scene-0147', 'scene-0148',
     'scene-0265', 'scene-0266', 'scene-0279', 'scene-0280', 'scene-0281', 'scene-0282', 'scene-0307', 'scene-0308',
     'scene-0309', 'scene-0310', 'scene-0311', 'scene-0312', 'scene-0313', 'scene-0314', 'scene-0333', 'scene-0334',
     'scene-0335', 'scene-0336', 'scene-0337', 'scene-0338', 'scene-0339', 'scene-0340', 'scene-0341', 'scene-0342',
     'scene-0343', 'scene-0481', 'scene-0482', 'scene-0483', 'scene-0484', 'scene-0485', 'scene-0486', 'scene-0487',
     'scene-0488', 'scene-0489', 'scene-0490', 'scene-0491', 'scene-0492', 'scene-0493', 'scene-0494', 'scene-0495',
     'scene-0496', 'scene-0497', 'scene-0498', 'scene-0547', 'scene-0548', 'scene-0549', 'scene-0550', 'scene-0551',
     'scene-0601', 'scene-0602', 'scene-0603', 'scene-0604', 'scene-0606', 'scene-0607', 'scene-0608', 'scene-0609',
     'scene-0610', 'scene-0611', 'scene-0612', 'scene-0613', 'scene-0614', 'scene-0615', 'scene-0616', 'scene-0617',
     'scene-0618', 'scene-0619', 'scene-0620', 'scene-0621', 'scene-0622', 'scene-0623', 'scene-0624', 'scene-0827',
     'scene-0828', 'scene-0829', 'scene-0830', 'scene-0831', 'scene-0833', 'scene-0834', 'scene-0835', 'scene-0836',
     'scene-0837', 'scene-0838', 'scene-0839', 'scene-0840', 'scene-0841', 'scene-0842', 'scene-0844', 'scene-0845',
     'scene-0846', 'scene-0932', 'scene-0933', 'scene-0935', 'scene-0936', 'scene-0937', 'scene-0938', 'scene-0939',
     'scene-0940', 'scene-0941', 'scene-0942', 'scene-0943', 'scene-1026', 'scene-1027', 'scene-1028', 'scene-1029',
     'scene-1030', 'scene-1031', 'scene-1032', 'scene-1033', 'scene-1034', 'scene-1035', 'scene-1036', 'scene-1037',
     'scene-1038', 'scene-1039', 'scene-1040', 'scene-1041', 'scene-1042', 'scene-1043']

mini_train = \
    ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']

mini_val = \
    ['scene-0103', 'scene-0916']

```
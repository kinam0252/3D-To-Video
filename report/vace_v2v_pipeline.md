# 3D Render to Realistic Video Pipeline (VACE V2V)

## 개요

Blender에서 렌더링한 synthetic 3D 아바타 영상을 **VACE (Wan2.1 기반 Video-to-Video diffusion model)**을 사용하여 photorealistic한 영상으로 변환하는 파이프라인.

### 핵심 아이디어
- 3D 렌더링에서 **구조적 정보(depth map 또는 grayscale)**를 추출
- 이를 conditioning으로 사용하여 AI가 **동일한 포즈/동작을 유지**하면서 텍스처만 realistic하게 재생성
- Text prompt로 원하는 외형/스타일을 제어

```
Blender 3D Render (synthetic)
        |
        v
Conditioning 추출 (Depth / Gray)
        |
        v
VACE 14B V2V + Text Prompt
        |
        v
Realistic Video 출력
```

---

## 환경 구성

### 하드웨어
- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- **CPU**: AMD Ryzen 9 9950X3D
- **RAM**: 128GB
- **OS**: Ubuntu (desktop, SSH 접근)

### 소프트웨어
- **Python**: 3.11 (venv)
- **PyTorch**: 2.11.0+cu128 (RTX 5090 Blackwell은 CUDA 12.8+ 필수)
- **VACE**: https://github.com/ali-vilab/VACE
- **Wan2.1**: https://github.com/Wan-Video/Wan2.1
- **bitsandbytes**: 0.49.2 (NF4 양자화용)

### 디렉토리 구조
```
~/Repos/
  3D-To-Video/
    vace_env/                        # Python 3.11 venv
    output/
      renders/                       # Blender 렌더 결과
      vace_v2v/                      # 1.3B depth 결과
      vace_v2v_14b/                  # 14B depth 결과 (테스트)
      vace_v2v_14b_female/           # 14B depth 여자
      vace_v2v_14b_male/             # 14B depth 남자
      vace_v2v_14b_female_gray/      # 14B gray 여자
    report/                          # 이 문서
  VACE/
    models/
      Wan2.1-VACE-1.3B/             # 1.3B 모델 (18GB)
      Wan2.1-VACE-14B/              # 14B 모델 (70GB)
      VACE-Annotators/              # Depth/Pose 추출기 (18GB)
    vace/                            # VACE 소스코드
  Wan2.1/                            # Wan2.1 소스코드 (attention 등)
```

---

## 모델 상세

### VACE-14B (Wan2.1-VACE-14B)
- **파라미터**: 17.34B (14B base + 3.3B VACE blocks)
- **원본 크기**: 69.35GB (fp32), 34.68GB (bf16)
- **양자화 후**: ~10.4GB (NF4 4-bit)
- **양자화 방식**: bitsandbytes NF4 (bnb_4bit_compute_dtype=bfloat16)

### 왜 양자화가 필요한가
| 형식 | 모델 크기 | RTX 5090 (32GB) 적합 여부 |
|------|----------|-------------------------|
| fp32 | 69.35GB  | 불가능                    |
| bf16 | 34.68GB  | VRAM 초과 (32GB)          |
| INT8 | ~17.3GB  | 활성화 메모리 부족         |
| NF4  | ~10.4GB  | 여유 ~21GB                |

### VACE-1.3B (비교용)
- **파라미터**: ~1.3B
- **크기**: ~18GB (fp32), bf16으로 GPU에 바로 로드 가능
- **속도**: 5.54초/step (14B의 ~4.4배 빠름)
- **품질**: 14B 대비 디테일 부족

---

## Conditioning 방식 비교

VACE는 여러 conditioning task를 지원:

### 1. Depth (--task depth)
- MiDaS depth map 추출하여 3D 구조만 가이드
- **장점**: 형태/동작 보존 우수
- **단점**: object 디테일 손실 (가방, 옷 등이 변형됨)
- **적합한 경우**: 전체적 구도만 유지하고 완전히 새로운 외형 생성

### 2. Gray (--task gray)
- Grayscale 변환하여 밝기/형태/텍스처 힌트 보존
- **장점**: object 형태 보존력 더 높음, 원본 구조 충실
- **단점**: 원본에 더 종속되어 창의적 변환 제한
- **적합한 경우**: 원본 object를 최대한 유지하면서 realistic하게 변환

### 3. 기타 지원 task
- `pose`: OpenPose 스켈레톤 기반 (사람 포즈만 제어)
- `scribble`: 에지/윤곽선 기반
- `flow`: 광학 흐름 기반 (모션 제어)
- `inpainting`: 마스크 영역만 재생성
- `frameref`: 참조 이미지 기반 스타일 전이

### 추천 전략
| 목표 | 추천 task |
|------|----------|
| 전체 외형 변환 (새로운 사람) | depth |
| Object 유지 + realistic 변환 | gray |
| 사람만 변환, 배경/물건 유지 | inpainting |
| 특정 스타일 참조 | frameref |

---

## 실행 방법

### 기본 명령어
```bash
cd ~/Repos/VACE

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
~/Repos/3D-To-Video/vace_env/bin/python3 vace/vace_pipeline.py \
  --base wan \
  --task depth \
  --video INPUT_VIDEO_PATH \
  --prompt "TEXT_PROMPT" \
  --ckpt_dir models/Wan2.1-VACE-14B/ \
  --model_name vace-14B \
  --frame_num 81 \
  --size 480p \
  --save_dir OUTPUT_DIR \
  --offload_model true \
  --t5_cpu
```

### 주요 파라미터
| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| --task | conditioning 방식 (depth, gray, pose, ...) | - |
| --video | 입력 비디오 경로 | - |
| --prompt | 생성 가이드 텍스트 | - |
| --model_name | 모델 선택 (vace-1.3B / vace-14B) | - |
| --frame_num | 생성 프레임 수 | 81 |
| --size | 해상도 (480p / 720p) | 480p |
| --sample_steps | diffusion step 수 (많을수록 고품질) | 50 |
| --sample_guide_scale | classifier-free guidance 강도 | 5.0 |
| --offload_model | VAE/DiT GPU 스왑 | true |
| --t5_cpu | T5 텍스트 인코더 CPU 실행 | 권장 |

### 예시: 여자 아바타 (Depth)
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
~/Repos/3D-To-Video/vace_env/bin/python3 vace/vace_pipeline.py \
  --base wan --task depth \
  --video ~/Repos/3D-To-Video/output/renders/female0_hdri_photorealistic/female0_hdri_photorealistic.mp4 \
  --prompt "A photorealistic young woman walking naturally in a sunlit park, wearing casual everyday clothes with a backpack, natural skin texture, realistic hair, cinematic lighting, shot on Sony A7III" \
  --ckpt_dir models/Wan2.1-VACE-14B/ \
  --model_name vace-14B \
  --frame_num 81 --size 480p \
  --save_dir ~/Repos/3D-To-Video/output/vace_v2v_14b_female/ \
  --offload_model true --t5_cpu
```

### 예시: 여자 아바타 (Gray - object 보존)
```bash
# --task gray로 변경하면 object 형태 더 잘 보존
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
~/Repos/3D-To-Video/vace_env/bin/python3 vace/vace_pipeline.py \
  --base wan --task gray \
  --video ~/Repos/3D-To-Video/output/renders/female0_hdri_photorealistic/female0_hdri_photorealistic.mp4 \
  --prompt "A photorealistic young woman walking naturally in a sunlit park, wearing casual everyday clothes with a backpack, natural skin texture, realistic hair, cinematic lighting, shot on Sony A7III" \
  --ckpt_dir models/Wan2.1-VACE-14B/ \
  --model_name vace-14B \
  --frame_num 81 --size 480p \
  --save_dir ~/Repos/3D-To-Video/output/vace_v2v_14b_female_gray/ \
  --offload_model true --t5_cpu
```

---

## 코드 패치 내역

RTX 5090 (Blackwell) + 32GB VRAM 제약으로 인해 아래 파일들을 수정:

### 1. ~/Repos/Wan2.1/wan/modules/attention.py
- **변경**: flash_attn 없이 SDPA fallback 추가
- **이유**: RTX 5090에 nvcc/CUDA toolkit 미설치하여 flash_attn 컴파일 불가
- **백업**: attention.py.bak

### 2. ~/Repos/VACE/vace/models/wan/wan_vace.py
- **변경 1**: from_pretrained() 후 모델을 CPU에 유지
- **변경 2**: NF4 4-bit 양자화로 GPU 로드 (bitsandbytes)
- **변경 3**: generate()에서 VAE/DiT GPU 메모리 오케스트레이션
  - 추론 전: VAE를 CPU로, DiT(NF4)는 GPU 유지
  - 추론 후: DiT GPU 유지, VAE를 GPU로 (디코딩용)
- **백업**: wan_vace.py.bak

### 3. ~/Repos/VACE/vace/annotators/gray.py
- **변경**: GrayAnnotator.__init__에 device 키워드 인자 추가
- **이유**: 전처리 코드에서 device 인자를 전달하지만 GrayAnnotator에 없었음

### 4. ~/Repos/Wan2.1/pyproject.toml
- **변경**: flash_attn 의존성 제거

---

## 성능 벤치마크

| 항목 | 1.3B | 14B NF4 |
|------|------|---------|
| GPU 메모리 | ~6GB | ~10.4GB (모델) + ~10GB (활성화) |
| 속도 (per step) | 5.54초 | 24.08초 |
| 총 시간 (50 steps) | ~4.5분 | ~20분 |
| 프레임 수 | 81 | 81 |
| 해상도 | 480p (640x624) | 480p (640x624) |
| 품질 | 보통 | 높음 |

---

## 출력 파일

각 실행 결과 디렉토리에 3개 파일 생성:
- `out_video.mp4` — 생성된 realistic 비디오
- `src_video.mp4` — conditioning 입력 (depth map 또는 grayscale)
- `src_mask.mp4` — 마스크 (inpainting 시 사용)

### 생성된 결과물
| 디렉토리 | 설명 |
|---------|------|
| output/vace_v2v/ | 1.3B depth (첫 테스트) |
| output/vace_v2v_14b/ | 14B depth (v32_front 테스트) |
| output/vace_v2v_14b_female/ | 14B depth 여자 아바타 |
| output/vace_v2v_14b_male/ | 14B depth 남자 아바타 |
| output/vace_v2v_14b_female_gray/ | 14B gray 여자 아바타 |

---

## Prompt 가이드

### 효과적인 프롬프트 구성
```
[사실감] + [인물 묘사] + [행동] + [환경] + [의상/소품] + [촬영 스타일]
```

### 예시
```
A photorealistic young woman walking naturally in a sunlit park,
wearing casual everyday clothes with a backpack,
natural skin texture, realistic hair with subtle highlights,
cinematic lighting, shallow depth of field,
shot on Sony A7III, 4K film grain
```

### 팁
- "photorealistic", "shot on [카메라]" — realistic 느낌 강화
- "natural skin texture" — 피부 질감 사실감
- "cinematic lighting" — 영화적 조명
- Object를 구체적으로 언급 (예: "with a backpack") — 보존 확률 증가
- "4K film grain" — 영상 텍스처 자연스러움

---

## 향후 개선 방향

1. **Inpainting 방식**: 사람 영역만 마스킹하여 배경/object 100% 보존
2. **2-pass 합성**: 사람만 V2V 변환 후 원본 object 합성
3. **해상도 업스케일**: 480p에서 720p/1080p로 (Real-ESRGAN 등)
4. **얼굴 보정**: CodeFormer / GFPGAN으로 얼굴 디테일 보강
5. **Temporal consistency**: 프레임 간 일관성 향상 (flow conditioning 활용)
6. **프롬프트 최적화**: 더 구체적인 의상/소품 묘사로 object 보존력 강화

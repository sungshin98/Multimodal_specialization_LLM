# 실험 로그

논문 4장의 표는 전부 이 디렉터리의 로그에서 읽은 것이다. 날짜별로 보관한다.

## 2026-08-08

| 파일 | 내용 |
|---|---|
| `bench_text.log` | 텍스트 경로 3세트 — 희석 스윕(12패킷×2회), 통합 스윕(9패킷×2회), 전체집계(12패킷×1회) |
| `bench_hotpot.log` | HotpotQA distractor 12패킷 × 5구성 |
| `packets_*.json` | 그 실행에 쓴 입력 패킷. 합성 패킷은 근거마다 `_role` 라벨이 붙어 있다 |

측정에 쓴 모델은 텍스트·통합·최종이 Upstage `solar-pro3`, 비전이 Google `gemini-3.5-flash` 다.

### 재현

```bash
# 합성 패킷 생성
python -m evidence_decoder.datagen --sweep dilution    --out packets_dilution.json
python -m evidence_decoder.datagen --sweep integration --out packets_integration.json

# 공개 벤치마크 패킷
python -m evidence_decoder.benchmarks --source hotpotqa --limit 12 --out packets_hotpot.json

# 측정
python -m evidence_decoder.bench --packets packets_dilution.json    --repeat 2 --group --score
python -m evidence_decoder.bench --packets packets_integration.json --repeat 2 --group --score
python -m evidence_decoder.bench --packets packets_hotpot.json      --repeat 1 --score
```

`UPSTAGE_API_KEY` 는 필수, `GOOGLE_API_KEY` 는 비전 경로에만 쓴다.

### 이 회차에서 유의할 점

- **요지 충족률 채점 버그를 고친 뒤 HotpotQA 를 재측정했다.** 심판이 정답 요지를
  쪼개거나 덧붙여 돌려주면 `covered / len(key_points)` 가 1 을 넘었다(1.08·1.17 관측).
  돌려받은 항목을 주어진 요지에 대응시켜 요지 단위로 한 번만 세도록 고쳤다.
  요지가 1개뿐인 HotpotQA 에서만 드러났고, 합성 패킷(요지 3개) 결과는 영향이 없다.
- `bypass` 구성은 이 회차의 문항이 전부 중간 복잡도로 분류되어 우회 조건을
  충족하지 않았다. `full` 과의 차이는 측정 잡음이며 우회 효과가 아니다.
- 비전 경로는 호출 한도 때문에 5구성 반복 비교에 포함하지 못했다.
  병렬 실행과 출력 분량 제한의 수치는 이전 회차의 별도 측정값이다.

# TTS 청음 비교 (Phase 2)

> PRD §3 Phase 2의 "TTS 후보 청음 비교" 기록. 기준: 한국어 반말 수다 톤의 자연스러움 + 두 목소리의 대비(청각적 화자 구분) + 비용.

## 2026-08-25 — 1차: provider 비교

| Provider | 모델/보이스 | 결과 |
|---|---|---|
| OpenAI TTS | tts-1, onyx/nova → ash/coral | 한국어 발음 어색(영어 최적화 모델). 배관 검증용으로만 사용 |
| ElevenLabs | multilingual v2 / v3 | 한국어 자연스러움 우위 → 채택 |

## 2026-08-25 — 2차: ElevenLabs 무료 사용 조건 규명

- 처음 고른 Voice Library(커뮤니티) 보이스 2종이 API에서 402.
- 402 응답 본문 확인: `paid_plan_required` — **크레딧 문제가 아니라 커뮤니티 보이스가 무료 플랜 API 사용 불가**.
- 실측으로 확인한 사용 가능 조합: **premade(기본) 보이스 × 모든 모델(v2/turbo/flash/v3) = 무료**.

## 2026-08-25 — 3차: premade 보이스 청음 (같은 대사, v2·v3 각 1회)

- 접근 가능 확인: 남 Adam/George/Brian/Daniel/Callum/Antoni/Liam/Will/Harry/Chris, 여 Sarah/Laura/Alice/Matilda/Jessica/Lily (Josh/Sam/Charlotte/Rachel은 402)
- 모델 비교: **v3가 v2보다 한국어 억양·표현 우위** + 오디오 태그([sighs], [excited], <break/>) 지원 → v3 채택
- 태그 연출 테스트: 페르소나 감정(도현 냉소/소은 들뜸)이 태그로 살아남 확인

## 확정 (2026-08-25)

- 모델: **eleven_v3**
- 보이스는 **쌍 단위 프리셋**으로 관리 (케미가 쌍 단위이듯 목소리 대비도 쌍 단위 — D-004):
  - `chris-jessica` (활성): Chris(도현) + Jessica(소은)
  - `liam-laura` (예비): Liam(도현) + Laura(소은)
- 전환은 duo.yaml `voice_preset` 값 교체만으로.

## 남은 과제
- 발화 LLM이 오디오 태그를 페르소나에 맞게 생성하도록 프롬프트 확장 (limit_sentences의 태그 오인 방지 포함)
- 스트리밍/문장 단위 TTS로 응답 지연(현재 발화 완성 후 합성) 단축
- 두 프리셋의 실대화 비교 청취 후 기본 조합 최종 확정

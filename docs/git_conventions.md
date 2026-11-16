# 브랜치 · 커밋 규칙

## 흐름

이슈 1개 = 브랜치 1개 = PR 1개.

1. **이슈 등록:** "작업" 양식으로 이슈를 등록하면 `type/이슈번호-설명` 브랜치가 자동으로 만들어진다(`create-issue-branch.yml`).
2. **작업:** 그 브랜치에서 작업한다. 브랜치를 따로 만들지 않는다.
3. **PR:** PR 본문에 `Closes #이슈번호`를 적는다. develop에 머지되면 이슈가 닫히고 브랜치는 자동으로 지워진다.

## 이름 규칙

`check-conventions.yml`이 PR마다 검사한다.

| 대상 | 형식 | type |
|---|---|---|
| 브랜치 | `type/설명` 또는 `type/번호-설명` (소문자, 단어는 `-`) | feature, fix, hotfix, refactor, design, docs, test, chore, release, infra |
| 커밋 | `type: 내용` | feat, fix, refactor, design, comment, style, docs, test, chore, init, rename, remove |

**헷갈리기 쉬운 점:** 브랜치는 `feature/`, 커밋은 `feat:`이다. `feat/`로 만든 브랜치는 검사에서 실패한다.

**검사에 실패하면:** 브랜치 이름을 GitHub에서 바꾸지 않는다. 바꾸면 그 브랜치에서 올린 PR이 닫힌다. 새 이름으로 브랜치를 만들고 PR을 다시 연다.

## 브랜치 보호 (develop, main 동일)

| 항목 | 설정 |
|---|---|
| 머지 전 PR 승인 | 1명 |
| 필수 검사 | `check` (브랜치·커밋 규칙 검사) |
| 최신 develop 반영 | 필수. PR이 BEHIND이면 "Update branch"를 누른다 |
| 관리자도 적용 | 예 |
| force push, 브랜치 삭제 | 막힘 |

PR 목록의 `MERGEABLE`은 "충돌 없음"이라는 뜻일 뿐, 머지할 수 있다는 뜻이 아니다. 실제로 머지할 수 있는지는 `mergeStateStatus`(BLOCKED, BEHIND, CLEAN)로 확인한다.

## 쓰지 않는 브랜치 자동 정리

`cleanup-stale-branches.yml`

| 실행 시점 | 확인 범위 |
|---|---|
| 이슈가 닫힐 때 | 그 이슈 번호의 브랜치 |
| 매주 월요일 09:00 (KST) | 전체 브랜치 |
| Actions 탭에서 수동 실행 | 전체 브랜치. 기본값은 dry run(지우지 않고 대상만 출력) |

아래를 **모두** 만족할 때만 지운다.

- develop에 없는 커밋이 하나도 없다. 그래서 지워도 코드가 사라지지 않는다.
- 그 브랜치에서 올린 열린 PR이 없다.
- 이름이 `type/번호-설명`이면 그 이슈가 닫혀 있다. 열린 이슈의 브랜치는 아직 작업 전일 수 있어서 남긴다.
- develop, main, `release/*`, `hotfix/*`가 아니다.

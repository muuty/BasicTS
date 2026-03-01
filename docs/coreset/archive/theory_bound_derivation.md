# Coreset Forecasting Bound Derivation (MAE, Expected Error)

## 0) Setup

\[
\mathcal{E}_{P}(f) := \mathbb{E}_{(X,Y)\sim P}\left[\ell(f(X),Y)\right], \quad
\ell(\hat y,y)=|\hat y-y|
\]

- \(\mathcal{E}_{P}(f)\): 분포 \(P\)에서의 expected MAE
- \(f\): 예측 모델 (여기서는 \(f_S\): coreset \(S\)로 학습된 모델)
- \(P\): \((X,Y)\) 결합분포
- \(\ell\): MAE loss (absolute error)

\[
\hat{\mathcal{E}}_{S}(f) := \frac{1}{|S|}\sum_{(x,y)\in S}\ell(f(x),y)
\]

- \(\hat{\mathcal{E}}_{S}(f)\): coreset \(S\)에서의 empirical MAE

분포 표기:
- \(P_{\text{train}}\): train 분포
- \(P_{\text{test}}\): test 분포
- \(P_{\text{core}}\): coreset 분포 (선택된 샘플이 유도하는 분포)
- \(P_{\text{train},X}, P_{\text{test},X}, P_{\text{core},X}\): 각 분포의 \(X\)-marginal

**Ground metric \(d_{ST}\)**:
입력 공간 \(\mathcal{X}\) 위의 거리 함수. Spatio-temporal forecasting에서 각 샘플
\(x = (\mathbf{x}^{(\text{in})}, \mathbf{x}^{(\text{out})}) \in \mathbb{R}^{T_{\text{in}} \times N \times F} \times \mathbb{R}^{T_{\text{out}} \times N \times F}\)
에 대해, 본 연구에서는 다음 네 가지 ground metric을 사용한다:

| distance_type | 정의 |
|---|---|
| euclidean | \(d_{ST}(x,x') = \|x_{\text{flat}} - x'_{\text{flat}}\|_2\) (전체 flatten 후 L2) |
| temporal | \(d_{ST}(x,x') = \|\bar{x}^{(t)} - \bar{x}'^{(t)}\|_2\) (node 평균 후 temporal feature의 L2) |
| spatial | \(d_{ST}(x,x') = \|\bar{x}^{(s)} - \bar{x}'^{(s)}\|_2\) (time 평균 후 spatial feature의 L2) |
| combined | \(d_{ST}(x,x') = \frac{1}{2}\tilde{d}_{\text{temporal}}(x,x') + \frac{1}{2}\tilde{d}_{\text{spatial}}(x,x')\) (정규화 후 평균) |

여기서 \(\tilde{d}\)는 \([0,1]\)로 정규화된 거리.
이 ground metric의 선택이 이후 Wasserstein 거리 \(W_1^{d_{ST}}\)와 Lipschitz 상수 \(L\)에 직접 영향을 미친다.

---

## 1) Telescoping 분해 (3항)

목표는 \(\mathcal{E}_{P_{\text{test}}}(f_S)\)를 \(\hat{\mathcal{E}}_S(f_S)\)로 upper bound하는 것.

**분해**: 중간 분포 \(P_{\text{train}}\), \(P_{\text{core}}\)를 경유하여 telescoping sum으로 분해한다.
이는 삼각부등식이 아니라, 등식으로 성립하는 대수적 분해이다:

\[
\mathcal{E}_{P_{\text{test}}}(f_S)-\hat{\mathcal{E}}_S(f_S)
=
\underbrace{\left(\mathcal{E}_{P_{\text{test}}}(f_S)-\mathcal{E}_{P_{\text{train}}}(f_S)\right)}_{\text{(A) test-train shift}}
+
\underbrace{\left(\mathcal{E}_{P_{\text{train}}}(f_S)-\mathcal{E}_{P_{\text{core}}}(f_S)\right)}_{\text{(B) train-coreset mismatch}}
+
\underbrace{\left(\mathcal{E}_{P_{\text{core}}}(f_S)-\hat{\mathcal{E}}_S(f_S)\right)}_{\text{(C) generalization on coreset}}
\]

이는 단순히 중간 항을 더했다 빼는 것 (\(+\mathcal{E}_{P_{\text{train}}} - \mathcal{E}_{P_{\text{train}}} + \mathcal{E}_{P_{\text{core}}} - \mathcal{E}_{P_{\text{core}}}\))으로 확인할 수 있다.

각 항의 절대값으로 upper bound:

\[
\mathcal{E}_{P_{\text{test}}}(f_S) - \hat{\mathcal{E}}_S(f_S)
\le |(A)| + |(B)| + |(C)|
\]

이 부등식은 실수의 성질 \(a + b + c \le |a| + |b| + |c|\)에서 나온다 (삼각부등식의 metric space 버전과는 구분).

따라서:

\[
\mathcal{E}_{P_{\text{test}}}(f_S)
\le
\hat{\mathcal{E}}_S(f_S)+|(A)|+|(B)|+|(C)|
\]

---

## 2) (C) Generalization on coreset

표준적인 uniform convergence bound를 적용한다.
\(f_S\)가 hypothesis class \(\mathcal{F}\)에 속하고, \(S\)가 \(P_{\text{core}}\)에서 i.i.d.로 추출된 \(|S|\)개 샘플이면:

\[
|(C)| = \left|\mathcal{E}_{P_{\text{core}}}(f_S)-\hat{\mathcal{E}}_S(f_S)\right|
\le \mathrm{Gen}(S,\delta)
\quad \text{with probability } \ge 1-\delta
\]

- \(\mathrm{Gen}(S,\delta)\): coreset에서의 일반화 항. 구체적 형태는 \(\mathcal{F}\)의 complexity에 의존.
  - Rademacher complexity 기반: \(\mathrm{Gen}(S,\delta) = 2\mathfrak{R}_{|S|}(\ell \circ \mathcal{F}) + \sqrt{\frac{\ln(2/\delta)}{2|S|}}\)
  - 이 항은 coreset 크기 \(|S|\)가 클수록 작아지며, coreset selection 방법과는 (직접적으로는) 무관하다.
- \(\delta\): failure probability (bound가 깨질 확률 상한)

**참고**: Coreset이 데이터-의존적으로 선택되므로, 엄밀하게는 \(S\)가 \(P_{\text{core}}\)에서 i.i.d.가 아닐 수 있다.
그러나 coreset 선택이 입력 \(X\)에만 의존하고 레이블 \(Y\)와 독립이면, 조건부 일반화 bound를 적용할 수 있다 (Sec. 6 참조).

---

## 3) (B) Train-coreset mismatch → \(L \cdot W_1 + \Delta_{\text{cond}}\)

\[
(B)=\mathcal{E}_{P_{\text{train}}}(f_S)-\mathcal{E}_{P_{\text{core}}}(f_S)
\]

**핵심 아이디어**: (B)는 두 분포의 marginal \(X\) 차이와 conditional \(Y|X\) 차이의 합으로 분해된다.

**Step 3a**: 조건부 기대값 함수 정의

\[
\phi_{\text{train}}(x):=\mathbb{E}_{Y\sim P_{\text{train}}(Y|X=x)}[\ell(f_S(x),Y)]
\]
\[
\phi_{\text{core}}(x):=\mathbb{E}_{Y\sim P_{\text{core}}(Y|X=x)}[\ell(f_S(x),Y)]
\]

이 함수들은 "주어진 입력 \(x\)에서의 expected loss"를 나타내며, conditional distribution에 의존한다.

**Step 3b**: Add-and-subtract trick

(B)를 같은 함수의 다른 분포 기대값 차이 + 다른 함수의 같은 분포 기대값 차이로 분해한다.
중간에 \(\mathbb{E}_{X \sim P_{\text{core},X}}[\phi_{\text{train}}(X)]\)를 더하고 빼면:

\[
(B) = \mathcal{E}_{P_{\text{train}}}(f_S) - \mathcal{E}_{P_{\text{core}}}(f_S)
\]
\[
= \mathbb{E}_{P_{\text{train},X}}[\phi_{\text{train}}(X)] - \mathbb{E}_{P_{\text{core},X}}[\phi_{\text{core}}(X)]
\]
\[
= \underbrace{\left(\mathbb{E}_{P_{\text{train},X}}[\phi_{\text{train}}(X)] - \mathbb{E}_{P_{\text{core},X}}[\phi_{\text{train}}(X)]\right)}_{\text{(B1) marginal shift: 같은 함수, 다른 분포}}
+ \underbrace{\left(\mathbb{E}_{P_{\text{core},X}}[\phi_{\text{train}}(X)] - \mathbb{E}_{P_{\text{core},X}}[\phi_{\text{core}}(X)]\right)}_{\text{(B2) conditional shift: 같은 분포, 다른 함수}}
\]

절대값으로 bound:

\[
|(B)|
\le
\underbrace{\left|\mathbb{E}_{X\sim P_{\text{train},X}}[\phi_{\text{train}}(X)]
-\mathbb{E}_{X\sim P_{\text{core},X}}[\phi_{\text{train}}(X)]\right|}_{\text{(B1)}}
+
\underbrace{\mathbb{E}_{X\sim P_{\text{core},X}}\left[|\phi_{\text{train}}(X)-\phi_{\text{core}}(X)|\right]}_{\text{(B2)}}
\]

**Step 3c**: (B1)에 Kantorovich-Rubinstein duality 적용

KR duality: 임의의 1-Lipschitz 함수 \(h\)에 대해
\[
\left|\mathbb{E}_{P}[h(X)] - \mathbb{E}_{Q}[h(X)]\right| \le W_1^{d}(P, Q)
\]

\(\phi_{\text{train}}\)이 ground metric \(d_{ST}\)에 대해 Lipschitz 상수 \(L\)을 가지면,
\(\phi_{\text{train}} / L\)은 1-Lipschitz이므로:

\[
\left|\mathbb{E}_{P_{\text{train},X}}\phi_{\text{train}}
-\mathbb{E}_{P_{\text{core},X}}\phi_{\text{train}}\right|
\le
L\,W_1^{d_{ST}}(P_{\text{train},X},P_{\text{core},X})
\]

**Step 3d**: (B2) = \(\Delta_{\text{cond}}\) 정의

\[
\Delta_{\text{cond}}(\text{train},\text{core};f_S)
:=
\mathbb{E}_{X\sim P_{\text{core},X}}\left[|\phi_{\text{train}}(X)-\phi_{\text{core}}(X)|\right]
\]

이 항은 **같은 입력 \(x\)에서 train 분포와 coreset 분포의 conditional \(Y|X\)가 다를 때** 발생한다.

**결론**:

\[
|(B)|
\le
L\,W_1^{d_{ST}}(P_{\text{train},X},P_{\text{core},X})
+\Delta_{\text{cond}}(\text{train},\text{core};f_S)
\]

---

## 3.1) Lipschitz 상수 \(L\)에 대하여

\(\phi_{\text{train}}(x) = \mathbb{E}_{Y|X=x}[|f_S(x) - Y|]\)의 Lipschitz 상수 \(L\)은 다음에 의존한다:

1. **모델 \(f_S\)의 smoothness**: \(f_S\)가 \(L_f\)-Lipschitz이면 \(\phi_{\text{train}}\)도 최소 \(L_f\)-Lipschitz이다.
   - 뉴럴 넷의 경우 \(L_f\)는 weight matrix의 spectral norm의 곱으로 upper bound 가능
   - Spectral normalization 등으로 \(L_f\)를 명시적으로 제어할 수 있다

2. **Conditional 분포 \(P(Y|X)\)의 smoothness**: \(X\)가 약간 변할 때 \(Y|X\)도 부드럽게 변하면 \(L\)이 작다.
   - Traffic forecasting에서는 인접 시간대의 교통 패턴이 부드럽게 변하므로 이 가정이 합리적이다.

3. **Ground metric \(d_{ST}\)의 선택**: metric을 rescale하면 \(L\)도 역으로 변한다 (trade-off).

실제로 \(L\)을 계산하는 것은 어렵지만, 이 bound에서 중요한 것은 \(L\)이 **coreset 선택 방법에 무관한 상수**라는 점이다. 따라서 서로 다른 coreset 방법을 비교할 때, \(W_1\) 항의 상대적 크기만으로 (B1)의 상대적 크기를 판단할 수 있다.

---

## 4) (A) Test-train shift도 동일 방식

Section 3과 동일한 add-and-subtract + KR duality를 적용:

\[
|(A)|
\le
L\,W_1^{d_{ST}}(P_{\text{test},X},P_{\text{train},X})
+\Delta_{\text{cond}}(\text{test},\text{train};f_S)
\]

**참고**: 이 항은 coreset 선택과 무관하게 고정된 값이다.
Train/test 분포 차이는 데이터셋의 시간적 분할 방식에 의해 결정되며, coreset 방법으로 줄일 수 없다.

---

## 5) 최종 일반 bound

Section 1-4를 결합하면 (확률 \(\ge 1-\delta\)로):

\[
\boxed{
\mathcal{E}_{P_{\text{test}}}(f_S)
\le
\hat{\mathcal{E}}_S(f_S)
+\mathrm{Gen}(S,\delta)
+L\,W_1(P_{\text{train},X},P_{\text{core},X})
+L\,W_1(P_{\text{test},X},P_{\text{train},X})
+\Delta_{\text{cond}}(\text{train},\text{core};f_S)
+\Delta_{\text{cond}}(\text{test},\text{train};f_S)
}
\]

(위 \(W_1\)는 모두 ground metric \(d_{ST}\) 기준)

각 항의 역할:
| 항 | 의미 | Coreset 선택으로 제어 가능? |
|---|---|---|
| \(\hat{\mathcal{E}}_S(f_S)\) | Coreset 위의 empirical loss | 간접적 (coreset 품질 → 학습 품질) |
| \(\mathrm{Gen}(S,\delta)\) | Coreset 위의 일반화 gap | \(|S|\)에만 의존 (크기가 같으면 동일) |
| \(L\,W_1(P_{\text{train},X},P_{\text{core},X})\) | Train-coreset 분포 mismatch | **직접 제어 가능** (핵심 항) |
| \(L\,W_1(P_{\text{test},X},P_{\text{train},X})\) | Test-train 분포 shift | 불가 (데이터셋에 의해 고정) |
| \(\Delta_{\text{cond}}(\text{train},\text{core})\) | Train-coreset conditional shift | Subset selection이면 ≈0 |
| \(\Delta_{\text{cond}}(\text{test},\text{train})\) | Test-train conditional shift | 불가 (데이터셋에 의해 고정) |

---

## 6) 케이스별 단순화

### Case 1: train/test 동일분포 (\(P_{\text{test}}=P_{\text{train}}\))

(A) 항이 정확히 0:

\[
\mathcal{E}_{P_{\text{test}}}(f_S)
\le
\hat{\mathcal{E}}_S(f_S)
+\mathrm{Gen}(S,\delta)
+L\,W_1(P_{\text{train},X},P_{\text{core},X})
+\Delta_{\text{cond}}(\text{train},\text{core};f_S)
\]

### \(\Delta_{\text{cond}} \approx 0\)이 성립하는 조건

\(\Delta_{\text{cond}}(\text{train},\text{core};f_S)\)는 같은 입력 \(x\)에서 train과 coreset의 \(Y|X\) 분포가 다를 때 나타난다.

**Subset selection**에서는 coreset이 train set의 부분집합이므로, coreset에 포함된 \(x\)에 대해
\(P_{\text{core}}(Y|X=x) = P_{\text{train}}(Y|X=x)\)가 정확히 성립한다.

그러나 coreset에 포함되지 않은 \(x\)에 대해서는 \(P_{\text{core}}(Y|X=x)\)가 정의되지 않으므로,
\(\Delta_{\text{cond}}\)는 사실상 **입력 공간에서 coreset이 대표하는 영역 vs 대표하지 못하는 영역**의
conditional 차이를 측정한다.

Traffic forecasting에서 \(\Delta_{\text{cond}} \approx 0\)이 합리적인 이유:

1. **Near-deterministic \(Y|X\)**: 교통 흐름은 물리적 법칙과 도로 네트워크 구조에 의해 강하게 제약된다.
   주어진 입력 패턴 \(X\) (과거 12시간 교통량)에 대해 미래 교통량 \(Y\)는 높은 확률로
   좁은 범위에 집중된다. 즉, \(\text{Var}(Y|X)\)가 작다.

2. **Subset이므로 bias 없음**: Coreset \(S \subset D_{\text{train}}\)이므로,
   coreset에 포함된 각 샘플의 \((x, y)\) 관계는 train 분포와 정확히 일치한다.
   Synthetic data 생성이나 reweighting과 달리, subset selection은 conditional distribution을 왜곡하지 않는다.

3. **시간 패턴의 연속성**: 교통 데이터는 시간에 대해 연속적이므로, coreset에 포함된 시점과
   가까운 시점의 conditional distribution은 유사하다.
   \(W_1\) 항이 작으면 (coreset이 train을 잘 대표하면) 대부분의 train 샘플은
   coreset의 어떤 샘플과 가까우므로, conditional shift도 자연스럽게 작아진다.

따라서:

\[
\mathcal{E}_{P_{\text{test}}}(f_S)
\lesssim
\hat{\mathcal{E}}_S(f_S)
+\mathrm{Gen}(S,\delta)
+L\,W_1(P_{\text{train},X},P_{\text{core},X})
\]

---

### Case 2: train/test 분포 다름 (\(P_{\text{test}}\neq P_{\text{train}}\))

\[
\mathcal{E}_{P_{\text{test}}}(f_S)
\le
\hat{\mathcal{E}}_S(f_S)
+\mathrm{Gen}(S,\delta)
+L\,W_1(P_{\text{train},X},P_{\text{core},X})
+L\,W_1(P_{\text{test},X},P_{\text{train},X})
+\Delta_{\text{cond}}(\text{train},\text{core};f_S)
+\Delta_{\text{cond}}(\text{test},\text{train};f_S)
\]

Traffic forecasting에서 train/test split은 시간 기준이므로 (e.g., 처음 70% = train, 나머지 30% = test),
비정상(non-stationary) 교통 패턴이 있으면 \(P_{\text{test}} \neq P_{\text{train}}\)이다.
이 경우 \(W_1(P_{\text{test},X}, P_{\text{train},X})\) 항이 0이 아니며,
coreset 선택으로 줄일 수 없는 고정된 domain gap이 존재한다.

---

## 7) k-medoids와의 연결 (핵심 인사이트)

### k-medoids 목적함수

k-medoids는 다음 목적함수를 최소화한다:

\[
\min_{S \subset D_{\text{train}},\, |S|=k}\;\frac{1}{n}\sum_{i=1}^{n}\min_{s\in S} d_{ST}(x_i, s)
\]

- \(x_i\): train 입력 샘플 (\(i = 1, \ldots, n\))
- \(S\): medoid 집합 (반드시 \(D_{\text{train}}\)의 부분집합)
- \(d_{ST}\): Section 0에서 정의한 ground metric

### 이 목적함수 = empirical \(W_1\) 최소화

위 목적함수는 **정확히** empirical train 분포 \(\hat{P}_{\text{train},X} = \frac{1}{n}\sum_i \delta_{x_i}\)와
coreset이 유도하는 empirical 분포 \(\hat{P}_{\text{core},X} = \frac{1}{k}\sum_{s \in S} w_s \cdot \delta_s\)
사이의 1-Wasserstein 거리와 대응한다.

구체적으로, 각 train 샘플 \(x_i\)를 가장 가까운 medoid \(s^*(x_i) = \arg\min_{s \in S} d_{ST}(x_i, s)\)로
배정하는 것은 discrete optimal transport plan \(\gamma\)를 정의하며:

\[
\frac{1}{n}\sum_{i=1}^{n}\min_{s\in S} d_{ST}(x_i,s)
= \sum_{i,j} \gamma_{ij}\, d_{ST}(x_i, s_j)
\ge W_1(\hat{P}_{\text{train},X},\hat{P}_{\text{core},X})
\]

(등호는 최적 transport plan이 nearest-assignment와 일치할 때 성립. 일반적으로 k-medoids의
nearest-assignment는 optimal transport plan의 good approximation이다.)

따라서 k-medoids 목적함수를 최소화하는 것은 bound의 핵심 항인
\(W_1(P_{\text{train},X}, P_{\text{core},X})\)를 직접적으로 줄이는 방향과 정렬된다.

### 왜 regression (특히 MAE forecasting)에서 특히 효과적인가?

MAE loss \(\ell(\hat{y}, y) = |\hat{y} - y|\)는 Lipschitz continuous (Lipschitz 상수 = 1 w.r.t. \(\hat{y}\)).
이 성질 덕분에:

1. **KR duality가 tight**: MAE의 Lipschitz 구조 덕분에 \(W_1\) 항이 실제 loss gap을 잘 반영한다.
   반면, 0-1 classification loss는 Lipschitz이 아니므로 같은 bound가 적용되지 않는다.

2. **밀도 비례 커버리지 = 최적 전략**: \(W_1\)을 최소화하려면 데이터가 밀집된 곳에
   더 많은 coreset 포인트를 배치해야 한다. 이는 k-medoids의 자연스러운 behavior이다.
   - 밀집 영역의 각 포인트는 작은 loss를 기여하지만, **포인트 수가 많으므로** 총 기여가 크다.
   - 희소 영역을 과도하게 대표하면 밀집 영역의 transport cost가 증가한다.

3. **Classification과의 차이**: 0-1 loss에서는 decision boundary 근처의 소수 샘플이 성능을 좌우한다.
   따라서 밀도 비례보다는 boundary-aware selection이 유리하며,
   k-medoids의 density-proportional 특성이 오히려 불리할 수 있다.

### 실험적 검증

우리의 실험에서:
- **k-medoids는 4개 distance type × 2개 ratio × 2개 모델의 거의 모든 setting에서 최저 MAE**를 달성
- **Within-setting 분석** (model, distance, ratio를 고정하고 method만 비교):
  ESS ratio (밀도 비례 커버리지 지표)가 MAE와 가장 강하게 상관 (cross-model Pearson r = -0.733)
- 이는 이론의 예측 — "density-proportional coverage가 \(W_1\)을 줄이고, 이것이 MAE를 줄인다" — 과 일치한다.

# 예측 과정

> ## Data Pre-Processing
> > ### 1. negative amount 처리
> > ### 2. store_id 별로 downsampling
> > ### 3. Target 값을 log 변환하여 정규화
> ## ARIMA 모델링

## Data Pre-Processing

### 1. negative amount 처리

- 하루 매출이 음수인 store들이 존재
> negative amount는 data의 variance를 증대시켜 결국 **prediction interval**이 커져 예측에 악영향을 끼친다.
> > prediction interval ; 새로운 관측에 대해 모델이 예측해야 하는 Target value의 범위.
>
> 따라서 negative인 Train example을 제거할 필요가 있음.

- 제거 방안
> Card_id를 이용하여 같은 id를 가진 example에 대해 **negative인 example보다 시간이 작으**면서 **가장 최근의 example** 중에서 <br>
> > 방안 1) 절댓값이 같은 양수 amount의 example을 제거. <br>
> > 방안 2) 절댓값이 더 큰 양수 amount의 example에서 음수 amount의 가격을 뻄. <br>
>
> 위의 두 방안을 기준으로 negative amount를 가진 example을 제거한다.

- 만약 **해당 card_id가 negative amount example 이전에 존재하지 않**거나 **방안 1, 2에 해당하는 경우가 없**을 경우 <br> 해당 example의 **실제 구매가 data 시작 시간보다 과거**이므로 **Training set에서 제외**한다.

### 2. store_id 별로 downsampling

- DownSampling 이란?
> 시계열 Data에서 downsampling은 **시간 간격을 재조정하여 datasample 수를 줄이는 것**으로 이해할 수 있다.

- Downsampling 진행 유무에 따른 차이
> - 통상적인 방법
> > - downsampling 없이 **ARIMA 모델**을 사용하여 각 일별 매출을 예측 -> 그렇게 산출된 100개의 매출 예측치를 총합하여 예측
>
> 단점 ; 예측 일수가 늘어날수록 **예측 불확실성이 증가**함.

> - downsampling 진행
> > - 100일 단위를 50일 기준으로 downsampling 하였을 때, **예측할 data가 100 -> 2 개로 줄어**들게 된다.
> > - 예측 대상이 줄어들어 불확실성을 줄이는 것에 도움이 되지만, **downsampling 기간이 너무 길면** 시계열에서 중요한 **seasonality**가 무시될 수 있음.

- resampling 기간
> - Sample이 충분하다면 최대한의 기간을 한 묶음으로 처리하여 최소로 필요한 기간만큼 판매 예측.
> - 요일에 따른 seannality가 존재 ; 최소 1주 단위로 묶어 seasonality를 없앨 필요가 있었음.
> > - seasonality ; 계절성, 주기적인 변동
> > - 아마 100일의 매출을 예측하는 것이므로 요일별 seasonality가 오히려 예측에 방해가 될 것으로 생각한 것 같음.
> - 년 단위로 존재하는 seasonality도 존재 ; 365.25로 나누어서 잘 떨어지는 수로 resampling 기간을 정할 필요가 있었음.
> > - 365.25를 나누어서 잘 나누어 떨어지는 숫자인 7, 14, 28일 단위로 downsampling 진행. (최소 10개의 sample이 생기지 않으면 더 적은 기간으로 downsampling함.)

- resampling 시에 기간에 따라 마지막 샘플에서 그 수가 다른 것들과 같이 않은 경우 최근의 data가 먼 과거의 data보다 중요하므로, **제일 앞쪽의 남는 sample들을 제거**

### 3. Target 값을 log 변환하여 정규화

- **QQ plot**을 통해 "amount"와 "log 변환한 amount"가 어느정도 **정규 분포**를 따르는 것을 확인.
- **boxplot** 통해 확인해보면 amount에 **outlier**가 많은 것을 확인할 수 있음.
- outlier의 영향을 줄이기 위해서 **log 변환한 amount를 Target으로 사용**
- 이 때, log 변환한 값이 **null 혹은 infinite** 값인 경우 해당 example을 제거.
> - null / infinite 값들이 몇 개인지 세어서 몇 퍼센트의 sample이 null / infinite 값인지 확인
> - **(최종 prediction) * (1 - probability of no sales)** 하여 판매가 없을 확률 만큼 매출을 discount

## ARIMA 모델링

### PACF(Partial AutoCorrelation Function / 편자기상관함수)

- PACF ; 시계열 Data 분석에서 model 선정의 기준이 되는 지표 중 하나.
> 어떤 것들에 대한 상관관계를 보려고 할 때, 그에 영향을 주는 다른 요소들을 제외하고 상관 관계를 보려는 방법.
- PACF를 통해 AR(0), AR(1), AR(2)를 확인했지만, AIC를 통해 ARIMA 모델의 파라미터인 p, q, r를 찾을 때 0~2의 값을 통해 구한 파라미터 보다 0~1 사이로 찾은 파리미터값이 더 정확한 예측을 함.

- AR 모형 ; 시계열 {X_t}를 종속변수로 하고 그 이전 시점의 시계열 {X_t-1, X_t-2, ... , X_t-p}를 독립변수로 가지는 회귀 모형.
> 과거 패턴이 지속되면 시계열 시계열 data과측치를 과거의 관측치와의 관계를 이용하여 예측할 수 있다는 생각에서 나옴.
- AR(0) ~ AR(p) 까지 구할 수 있는데, 이는 과거의 관측치를 몇 개까지 보고 분석할 것인지를 결정한다.

- 이론적으로 AR(p)의 PACF는 p 값 까지는 0이 아닌 값을 가지고 p 이후로는 0을 가진다. 즉, 지수적으로 감소하는 그래프를 가진다.
- 따라서 그러한 그래프를 가지는 AR 모델을 찾아서 적용.

### ARIMA 모델
- 파라미터
> - p ; 자기회귀(Autoregression), 이전 관측치와 현재 관측치와의 관계를 규정
> - d ; 시계열 모형을 stationary하게 만들기 위해 차분을 이용하는 것.
> - q ; 이전 관측치의 편차와의 관계를 규정
> > - 따라서 총 가능한 경우의 수는 (0, 0, 0) ~ (1, 1, 1) 까지 8가지가 된다.
> - 각각의 parameter 조합에 대해 AIC가 최소인 모델을 선택하여 예측한다.
> > - AIC ; 모델이 기존의 데이터를 얼마나 잘 설명하는지, 그리고 그 모델이 얼마나 간단한 지의 두 요소에 대한 평가치. (작을 수록 최적의 모형임) 

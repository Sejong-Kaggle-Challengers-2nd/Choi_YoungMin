# visualization

## 시도방안

1. 각 'datetime', 'days_of_week', 'holyday', 'installments'에 대한 'amount'를 plot

2. 각 요일별로 몇 회의 거래가 발생하였는지 histogram으로 제시

3. ~~공휴일 여부에 따라 몇 회의 거래가 발생하였는지 histogram으로 제시~~

4. 주중 / 주말 에 따라 몇 회의 거래가 발생하였는지 histogram으로 제시

## 개선방안

1. **공휴일 여부에 따른 거래량 histogram(시도방안 3번**)은 공휴일 보다 **공휴일이 아닌 날이 훨씰 많기 때문에 당연히 공휴일이 아닌 날의 거래량이 더 많게 나타남**. 따라서 좋은 시각화 정보가 아님.

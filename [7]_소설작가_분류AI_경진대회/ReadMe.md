# 전체적인 흐름

- XGboost의 Feature 들을 계산하여 dataframe에 추가
  - text를 분석하기 위한 feature를 생성하여 train과 test dataframe에 추가.
    - 단어의 개수, 단어의 평균 길이 등... 함수로 정의 (lamda 함수로 적용)
    - 작가별 등장인물의 list를 만들어서 text에 등장하는 인물과의 유사도 계산하여 feature로 추가
    - fasttext 패키지를 이용하여 text를 vector화 하여 추가
  - sklearn.feature_extraction.text의 Vectorizer를 이용하여 inference하고 각 클래스일 확률을 계산하여 dataframe에 추가
    - 다양한 모델을 사용하여 클래스 별 확률을 계산하고 해당 결과를 feature로써 dataframe에 추가.
      - Logistic Regression
      - SGDClassifier
      - RandomForestClassifier
      - MLPClassifier
      - DecisionTreeClassifier
      - Naive Bayes
    - 중간에 Turncated SVD를 이용하여 선형 차원 축소를 진행. (총 2회)
- XGboost를 이용하여 최종 inference

# 사용된 XGBoost Feature

- Meta Feature (문장 길이, Stop words 갯수, ..., Named Entity)
- FastText Embedding
- Naive Bayes
- Logistic Regression
- SGDClassifier
- RandomForestClassifier
- MLPClassifier
- DecisionTreeClassifier

# News

## textstat

- Textstat는 텍스트에서 통계를 계산하는 데 사용하기 쉬운 라이브러리입니다. 가독성, 복잡성 및 등급 수준을 결정하는 데 도움이 됩니다.

## fasttext

- 단어를 벡터로 만드는 또 다른 방법으로는 페이스북에서 개발한 FastText가 있습니다.
- Word2Vec 이후에 나온 것이기 때문에, 메커니즘 자체는 Word2Vec의 확장이라고 볼 수 있습니다. 
- Word2Vec와 FastText와의 가장 큰 차이점이라면 Word2Vec는 단어를 쪼개질 수 없는 단위로 생각한다면, FastText는 하나의 단어 안에도 여러 단어들이 존재하는 것으로 간주합니다. 즉 내부 단어(subword)를 고려하여 학습합니다.

## 1) flesch_reading_ease

- Flesch reading-ease test 에서는 점수가 높을수록 읽기 쉬운 재료를 나타내며 숫자가 낮을수록 읽기 어려운 구절을 표시합니다.
- Flesch Reading Ease Score를 반환합니다.
- 최대 점수는 121.22점이지만 점수가 얼마나 낮을 수 있는지에 대한 제한은 없습니다. 음수 점수가 유효합니다.

## 2) nltk

- 교육용으로 개발된 자연어 처리 및 문서 분석용 파이썬 패키지. 다양한 기능 및 예제를 가지고 있으며 실무 및 연구에서도 많이 사용됩니다.
- 말뭉치(corpus)는 자연어 분석 작업을 위해 만든 샘플 문서 집합을 말한다. 단순히 소설, 신문 등의 문서를 모아놓은 것도 있지만 품사. 형태소, 등의 보조적 의미를 추가하고 쉬운 분석을 위해 구조적인 형태로 정리해 놓은 것을 포함한다.
- 말뭉치 자료는 설치시에 제공되지 않고 download 명령으로 사용자가 다운로드 받아야 한다.
- nltk.download("book") 명령을 실행하면 NLTK 패키지 사용자 설명서에서 요구하는 대부분의 말뭉치를 다운로드 받아준다.

### 2-1) nltk.tokenize

- 자연어 문서를 분석하기 위해서는 우선 긴 문자열을 분석을 위한 작은 단위로 나누어야 한다. 이 문자열 단위를 토큰(token)이라고 하고 이렇게 문자열을 토큰으로 나누는 작업을 토큰 생성(tokenizing)이라고 한다. 영문의 경우에는 문장, 단어 등을 토큰으로 사용하거나 정규 표현식을 쓸 수 있다.
- 문자열을 토큰으로 분리하는 함수를 토큰 생성 함수(tokenizer)라고 한다. 토큰 생성 함수는 문자열을 입력받아 토큰 문자열의 리스트를 출력한다.
~~~python
from nltk.tokenize import word_tokenize
word_tokenize(emma_raw[50:100])
~~~
> ['Emma',
 'Woodhouse',
 ',',
 'handsome',
 ',',
 'clever',
 ',',
 'and',
 'rich',
 ',',
 'with',
 'a']
 
### 2-2) from nltk.tag import pos_tag (품사 부착)
 
 - 품사(POS, part-of-speech)는 낱말을 문법적인 기능이나 형태, 뜻에 따라 구분한 것이다. 품사의 구분은 언어마다 그리고 학자마다 다르다. 예를 들어 NLTK에서는 펜 트리뱅크 태그세트(Penn Treebank Tagset)라는 것을 이용한다. 다음은 펜 트리뱅크 태그세트에서 사용하는 품사의 예이다.
     - NNP : 단수 고유명사
     - VB : 동사
     - VBP : 동사 현재형
     - TO : to 전칳사
     - NN : 명사(단수형 혹은 집합형)
     - DT : 관형사
 - pos_tag 명령을 사용하면 단어 토큰에 품사를 부착하여 튜플로 출력한다. 다음 예문에서 refuse, permit이라는 같은 철자의 단어가 각각 동사와 명사로 다르게 품사 부착된 것을 볼 수 있다.
 ~~~python
from nltk.tag import pos_tag
sentence = "Emma refused to permit us to obtain the refuse permit"
tagged_list = pos_tag(word_tokenize(sentence))
tagged_list
~~~
> [('Emma', 'NNP'),
 ('refused', 'VBD'),
 ('to', 'TO'),
 ('permit', 'VB'),
 ('us', 'PRP'),
 ('to', 'TO'),
 ('obtain', 'VB'),
 ('the', 'DT'),
 ('refuse', 'NN'),
 ('permit', 'NN')]
 
 - Scikit-Learn 등에서 자연어 분석을 할 때는 같은 토큰이라도 품사가 다르면 다른 토큰으로 처리해야 하는 경우가 많은데 이 때는 원래의 토큰과 품사를 붙여서 새로운 토큰 이름을 만들어 사용하면 철자가 같고 품사가 다른 단어를 구분할 수 있다.
 
### 2-3) nltk.ne_chunk

- nltk 라이브러리 ne_chunk() 함수를 사용해서 개체명을 인식시킬 수 있다
- 개체명 인식을 사용하면 코퍼스로부터 어떤 단어가 사람, 장소, 조직 등을 의미하는 단어인지를 찾을 수 있습니다.
- 어떤 이름을 의미하는 단어를 보고는 그 단어가 어떤 유형인지를 인식하는 것을 말합니다.
    - "유정이는 2018년에 골드만삭스에 입사했다."
    - 유정 -> 사람 / 2018년 -> 시간 / 골드만삭스 -> 조직
- NLTK에서는 개체명 인식기(NER chunker)를 지원하고 있으므로, 별도 개체명 인식기를 구현할 필요없이 NLTK를 사용해서 개체명 인식을 수행할 수 있습니다.
- ne_chunk는 개체명을 태깅하기 위해서 앞서 품사 태깅(pos_tag)이 수행되어야 합니다. 위의 결과에서 James는 PERSON(사람), Disney는 조직(ORGANIZATION), London은 위치(GPE)라고 정상적으로 개체명 인식이 수행된 것을 볼 수 있습니다.
~~~python
from nltk import word_tokenize, pos_tag, ne_chunk
sentence = "James is working at Disney in London"
sentence=pos_tag(word_tokenize(sentence))
print(sentence) # 토큰화와 품사 태깅을 동시 수행
~~~
> [('James', 'NNP'), ('is', 'VBZ'), ('working', 'VBG'), ('at', 'IN'), ('Disney', 'NNP'), ('in', 'IN'), ('London', 'NNP')]
~~~python
sentence=ne_chunk(sentence)
print(sentence) # 개체명 인식
~~~
> (S
  (PERSON James/NNP)
  is/VBZ
  working/VBG
  at/IN
  (ORGANIZATION Disney/NNP)
  in/IN
  (GPE London/NNP))

### 2-4) from nltk.sentiment.vader import SentimentIntensityAnalyzer

- 문자열을 가져와서 네 가지 범주 각각에 대한 점수 dictionary를 반환합니다.
    - negative
    - neutral
    - positive
    - compound(computed by normalizing the scores above)
    
> a = 'This was a good movie.'
> sid.polarity_scores(a)

> OUTPUT-{'neg': 0.0, 'neu': 0.508, 'pos': 0.492, 'compound': 0.4404}

> a = 'This was the best, most awesome movie EVER MADE!!!'
> sid.polarity_scores(a)

> OUTPUT-{'neg': 0.0, 'neu': 0.425, 'pos': 0.575, 'compound': 0.8877}

### 2-5) from nltk.corpus import stopwords

- 갖고 있는 데이터에서 유의미한 단어 토큰만을 선별하기 위해서는 큰 의미가 없는 단어 토큰을 제거하는 작업이 필요합니다. 
- 여기서 큰 의미가 없다라는 것은 자주 등장하지만 분석을 하는 것에 있어서는 큰 도움이 되지 않는 단어들을 말합니다. 
- 예를 들면, I, my, me, over, 조사, 접미사 같은 단어들은 문장에서는 자주 등장하지만 실제 의미 분석을 하는데는 거의 기여하는 바가 없는 경우가 있습니다. 
- 이러한 단어들을 불용어(stopword)라고 하며, NLTK에서는 위와 같은 100여개 이상의 영어 단어들을 불용어로 패키지 내에서 미리 정의하고 있습니다.
~~~python
from nltk.corpus import stopwords  
stopwords.words('english')[:10]
~~~
> ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']  

## 생성한 feature

1. 각 문장에 포함된 **'단어의 개수'**
2. 각 문장에 포함된 **'단어의 평균 길이'**
3. 각 문장에 포함된 **'겹치지 않는 단어의 개수'**
4. 각 문장에 포함된 **'문자의 개수'**
5. 각 문장에 포함된 **'stopwards(불용어)의 개수'**
6. 각 문장에 포함된 **'문장부호의 개수'**
7. 각 문자에 포함된 단어 중 **'Upper case로 된 단어의 비율'**
8. 각 문자에 포함된 단어 중 **'title case(upper case + lower case)로 된 단어의 비율'**
9. 각 문장에 포함된 전체 문자의 개수 중 **'','로 구분되어진 chunk에 포함된 문자들의 평균 개수에 대한 비율'**
10. 각 문장에 포함된 전체 문자의 개수 중 **'ascii 문자나 숫자와 같은 symbol의 비율'**
11. 각 문장에 포함된 **'명사의 개수'**
12. 각 문장에 포함된 **'형용사의 개수'**
13. 각 문장에 포함된 **'동사의 개수'**
14. 각 문장의 **'SentimentIntensityAnalyzer의 compound 분석 값'**
15. 각 문장에서 **'단수 주어/주어/목적어 token이 포함된 갯수'**
16. 각 문장에서 **'복수 주어/주어/목적어 token이 포함된 갯수'**
17. 각 문장에 포함된 전체 문자의 개수에 대한 **'첫번째 문자 길이의 비율'**
18. 각 문장에 포함된 전체 문자의 개수에 대한 **'마지막 문자 길이의 비율'**
19. **첫번째 단어의 'symbol id를 구함'**
20. **마지막 단어의 'symbol id를 구함'**
21. **flesch_reading_ease score**를 계산

## fasttext

### festtext.train_unsupervised('data.txt')

- 파라미터로 모델 지정이 가능함.('skipgram', 'cbow')
- prameters
    - input : training file path **(required)**
    - model : unsupervised fasttext model / {cbow, skipgram} / default=skipgram
    - lr : learning rate / default=0.05
    - dim : size of word vectors / default=100
    - ws : size of the context window / default=5
    - epoch : number of epochs / default=5
    - minCount : minimal number of word occurences / default=5
    - minn : min length of char ngram / default=3
    - maxn : min length of char ngram / default=6
    - neg : number of negatives sampled / default=5
    - wordNgrams : max length of wrd ngram / default=1
    - loss : loss fuction / {ns, hs, softmax, ova} / default=ns
    - bucket : number of buckets / default=2,000,000
    - thread : number of treads / default=number of cpus
    - lrUpdateRate : change the rate of updates for the learning rate / default=100
    - t : sampling threshold / default=0.0001
    - verbose : verbose / default=2
- Model object fuctions
    - get_dimension           
        - Get the dimension (size) of a lookup vector (hidden layer).
        - This is equivalent to `dim` property.
    - get_input_vector        
        - Given an index, get the corresponding vector of the Input Matrix.
    - get_input_matrix        
        - Get a copy of the full input matrix of a Model.
    - get_labels              
        - Get the entire list of labels of the dictionary
        - This is equivalent to `labels` property.
    - get_line                
        - Split a line of text into words and labels.
    - get_output_matrix       
        - Get a copy of the full output matrix of a Model.
    - get_sentence_vector     
        - Given a string, get a single vector represenation. This function
        - assumes to be given a single line of text. We split words on
        - whitespace (space, newline, tab, vertical tab) and the control
        - characters carriage return, formfeed and the null character.
    - get_subword_id          
        - Given a subword, return the index (within input matrix) it hashes to.
    - get_subwords            
        - Given a word, get the subwords and their indicies.
    - get_word_id             
        - Given a word, get the word id within the dictionary.
    - get_word_vector         
        - Get the vector representation of word.
    - get_words               
        - Get the entire list of words of the dictionary
        - This is equivalent to `words` property.
    - is_quantized            
        - whether the model has been quantized
    - predict                 
        - Given a string, get a list of labels and a list of corresponding probabilities.
    - quantize                
        - Quantize the model reducing the size of the model and it's memory footprint.
    - save_model              
        - Save the model to the given path
    - test                    
        - Evaluate supervised model using file given by path
    - test_label              
        - Return the precision and recall score for each label.
        
### 실습
~~~python
import fasttext
model = fasttext.train_unsupervised('review.sorted.uniq.refined.tsv.text.tok',model='skipgram', epoch=5,lr = 0.1)

print(model['행사']) # get the vector of the word '행사'
~~~

## TruncatedSVD

- singular value decomposition(SVD)를 이용하여 linear dimensinality reduction(선형 차원 축소)를 수행함.
- PCA와는 다르게 이 estimator는 singular value decomposition 값을 계산하기 이전에 data를 center에 두지 않음.
    - 이는 희소 행렬에서 효율적으로 동작한다는 것을 의미한다.
- 특히, turncated SVD는 sklearn.feature_extraction.text의 벡터라이저가 반환한 term count/tf-idf 행렬에서 작용함.
    - 이 맥락에서 latent semantic analysis(LSA, 잠재 의미 분석)으로 알려져 있음.

## naive_bayes.MultinomialNB

- 다항식 모델을 위한 Naive bayes classifier
- 이는 이산형 특징을 가진 분류(ex: 텍스트 분류를 위한 단어 수 세기)에 적합함.
- 다항 분포에는 일반적으로 정수 피쳐 카운트가 필요함.
- 그러나 실제로는 tf-idf와 같은 부분 계수도 작동할 수 있음.



# lead
select event_type, value-value_2 as value
from 
    (select event_type
    , value
    , lead(value)over(partition by event_type order by time desc) value_2
    , row_number()over(partition by event_type order by time desc) num
    from events
    where 1=1 
    and event_type in 
        (SELECT event_type
        from events
        group by 1
        having count(*)>=2)
    ) a
where 1=1
and num=1
order by 1

# date_format
SELECT A.TITLE
,A.BOARD_ID
,B.REPLY_ID
,B.WRITER_ID
,B.CONTENTS
,DATE_FORMAT(B.CREATED_DATE,'%Y-%m-%d') CREATED_DATE
FROM USED_GOODS_BOARD A
INNER JOIN USED_GOODS_REPLY B ON A.BOARD_ID = B.BOARD_ID
WHERE 1=1
AND A.CREATED_DATE BETWEEN '2022-10-01' AND '2022-10-30'
ORDER BY B.CREATED_DATE, A.TITLE
;

# 오름차순, 내림차순 정렬 확인하기
SELECT USER_ID, PRODUCT_ID
FROM ONLINE_SALE
GROUP BY 1,2
HAVING COUNT(DISTINCT SALES_DATE)>=2
ORDER BY 1,2 DESC

# date_format 신경쓰기
SELECT A.*
FROM 
(SELECT DATE_FORMAT(SALES_DATE,'%Y-%m-%d') SALES_DATE, PRODUCT_ID, USER_ID, SALES_AMOUNT
FROM ONLINE_SALE
WHERE SALES_DATE BETWEEN '2022-03-01' AND '2022-03-31'
UNION 
SELECT DATE_FORMAT(SALES_DATE,'%Y-%m-%d') SALES_DATE, PRODUCT_ID, NULL AS USER_ID, SALES_AMOUNT
FROM OFFLINE_SALE
WHERE SALES_DATE BETWEEN '2022-03-01' AND '2022-03-31'
) A
ORDER BY SALES_DATE, PRODUCT_ID, USER_ID

# 값 필터링
SELECT ANIMAL_ID, NAME
FROM ANIMAL_INS
WHERE INTAKE_CONDITION = 'Sick'
ORDER BY 1

# 문제 제대로 읽기
SELECT ANIMAL_ID, NAME
FROM ANIMAL_INS
WHERE INTAKE_CONDITION != 'Aged'
ORDER BY 1

# CASE WHEN
SELECT BOARD_ID
, WRITER_ID
, TITLE
, PRICE
, CASE WHEN STATUS='SALE' THEN '판매중'
WHEN STATUS='RESERVED' THEN '예약중'
WHEN STATUS='DONE' THEN '거래완료'
ELSE NULL END AS STATUS
FROM USED_GOODS_BOARD
WHERE 1=1 
AND CREATED_DATE = '2022-10-05'
ORDER BY BOARD_ID DESC

# WHERE IN
SELECT ANIMAL_ID, NAME, SEX_UPON_INTAKE
FROM ANIMAL_INS
WHERE 1=1
AND NAME IN ('Lucy','Ella','Pickle','Rogan','Sabrina','Mitty')

# 아니 animal_type = dog 를 넣어야함
SELECT ANIMAL_ID, NAME
FROM ANIMAL_INS
WHERE 1=1
AND UPPER(NAME) LIKE '%EL%'
AND ANIMAL_TYPE = 'Dog'
ORDER BY NAME

# RLIKE
SELECT ANIMAL_ID, NAME
,CASE WHEN  SEX_UPON_INTAKE RLIKE 'Neutered|Spayed' THEN 'O'
ELSE 'X' END AS SEX_UPON_INTAKE
FROM ANIMAL_INS
ORDER BY ANIMAL_ID

# 경과일수는 DATEDIFF
SELECT A.ANIMAL_ID
, A.NAME
FROM ANIMAL_OUTS A
INNER JOIN ANIMAL_INS B ON A.ANIMAL_ID = B.ANIMAL_ID
ORDER BY DATEDIFF(A.DATETIME,B.DATETIME) DESC
LIMIT 2

# date_format
SELECT ANIMAL_ID, NAME, DATE_FORMAT(DATETIME,'%Y-%m-%d') DATETIME
FROM ANIMAL_INS
ORDER BY ANIMAL_ID

# CONCAT, 문제이해, 정렬
SELECT CONCAT('/home/grep/src/',A.BOARD_ID,'/',FILE_ID,FILE_NAME,FILE_EXT) FILE_PATH
FROM 
    (SELECT BOARD_ID
    FROM USED_GOODS_BOARD
    ORDER BY VIEWS DESC
    LIMIT 1) A
INNER JOIN USED_GOODS_FILE B ON A.BOARD_ID=B.BOARD_ID
ORDER BY 1 DESC

# CONCAT, SUBSTRING, 문제 잘 읽기 CITY
SELECT A.USER_ID
, A.NICKNAME
, CONCAT(CITY,' ',A.STREET_ADDRESS1,' ',A.STREET_ADDRESS2) 전체주소
, CONCAT(SUBSTRING(A.TLNO,1,3),'-', SUBSTRING(A.TLNO,4,4),'-',SUBSTRING(A.TLNO,8,4)) 전화번호
FROM USED_GOODS_USER A
INNER JOIN 
    (SELECT WRITER_ID
    FROM USED_GOODS_BOARD
    GROUP BY 1
    HAVING COUNT(BOARD_ID)>=3) B ON A.USER_ID = B.WRITER_ID
ORDER BY A.USER_ID DESC

# GROUP BY HAVING
SELECT A.USER_ID, A.NICKNAME, B.TOTAL_SALES
FROM USED_GOODS_USER A
INNER JOIN 
    (SELECT WRITER_ID, SUM(PRICE) TOTAL_SALES
    FROM USED_GOODS_BOARD
    WHERE 1=1 
    AND STATUS = 'DONE'
    GROUP BY 1
    HAVING SUM(PRICE)>=700000) B ON A.USER_ID = B.WRITER_ID
ORDER BY B.TOTAL_SALES

# 조회
SELECT ANIMAL_TYPE, COUNT(DISTINCT ANIMAL_ID) COUNT
FROM ANIMAL_INS
GROUP BY 1
ORDER BY 1

# LIKE
SELECT *
FROM CAR_RENTAL_COMPANY_CAR
WHERE 1=1
AND OPTIONS LIKE '%네비게이션%'
ORDER BY CAR_ID DESC


# 가장 오래보관한 3위
SELECT A.NAME, A.DATETIME
FROM ANIMAL_INS A
LEFT JOIN ANIMAL_OUTS B ON A.ANIMAL_ID = B.ANIMAL_ID
WHERE B.ANIMAL_ID IS NULL
ORDER BY A.DATETIME ASC
LIMIT 3

# 연도 추출 YEAR
SELECT BOOK_ID, DATE_FORMAT(PUBLISHED_DATE,'%Y-%m-%d') PUBLISHED_DATE
FROM BOOK
WHERE 1=1
AND CATEGORY = '인문'
AND YEAR(PUBLISHED_DATE) = 2021
ORDER BY 2

# IFNULL, CASE WHEN 실수체크
SELECT WAREHOUSE_ID
, WAREHOUSE_NAME
, ADDRESS
, IFNULL(FREEZER_YN,'N') FREEZER_YN
FROM FOOD_WAREHOUSE
WHERE 1=1
AND ADDRESS LIKE '%경기도%'
ORDER BY 1

# is null 로 case when 설정
SELECT ORDER_ID
, PRODUCT_ID
, DATE_FORMAT(OUT_DATE,'%Y-%m-%d') OUT_DATE
, CASE WHEN OUT_DATE IS NULL THEN '출고미정'
WHEN OUT_DATE <= '2022-05-01' THEN '출고완료'
ELSE '출고대기' END AS 출고여부
FROM FOOD_ORDER
ORDER BY 1

# year
SELECT COUNT(DISTINCT USER_ID) USERS
FROM USER_INFO
WHERE 1=1
AND YEAR(JOINED) = 2021
AND AGE BETWEEN 20 AND 29

# 날짜를 하드코딩하지 않기 10월은 31일까지 있음
SELECT MONTH(START_DATE) MONTH
,A.CAR_ID
,COUNT(*) RECORDS
FROM CAR_RENTAL_COMPANY_RENTAL_HISTORY A
INNER JOIN 
    (SELECT CAR_ID
    FROM CAR_RENTAL_COMPANY_RENTAL_HISTORY
    WHERE MONTH(START_DATE) BETWEEN 8 AND 10
    GROUP BY 1
    HAVING COUNT(*) >=5
    ) B ON A.CAR_ID = B.CAR_ID
WHERE MONTH(A.START_DATE) BETWEEN 8 AND 10
GROUP BY 1,2
HAVING COUNT(*)>=1
ORDER BY 1,2 DESC


# RILIE 여러개 중 1개
SELECT CAR_TYPE, COUNT(DISTINCT CAR_ID) CARS
FROM CAR_RENTAL_COMPANY_CAR
WHERE OPTIONS RLIKE '통풍시트|열선시트|가죽시트'
GROUP BY 1
ORDER BY 1

# JOIN이 여러개
SELECT B.AUTHOR_ID, C.AUTHOR_NAME, B.CATEGORY, SUM(B.PRICE*A.SALES) TOTAL_SALES
FROM BOOK_SALES A
INNER JOIN BOOK B ON A.BOOK_ID = B.BOOK_ID
INNER JOIN AUTHOR C ON B.AUTHOR_ID = C.AUTHOR_ID
WHERE 1=1
AND YEAR(A.SALES_DATE) = 2022
AND MONTH(A.SALES_DATE) = 1
GROUP BY 1,2,3
ORDER BY 1,3 DESC

# 왜 서울 RLIKE '%%' 안돼! AVERAGE 하면 GROUP BY 꼭 체크
SELECT A.REST_ID
, A.REST_NAME
, A.FOOD_TYPE
, A.FAVORITES
, A.ADDRESS
, ROUND(AVG(B.REVIEW_SCORE),2) SCORE
FROM REST_INFO A
INNER JOIN REST_REVIEW B ON A.REST_ID = B.REST_ID
WHERE 1=1
AND A.ADDRESS LIKE '서울%'
GROUP BY 1,2,3,4,5
ORDER BY 6 DESC, FAVORITES DESC

# JOIN할때 테이블 잘 체크하기
SELECT A.ANIMAL_ID, A.ANIMAL_TYPE, A.NAME
FROM ANIMAL_INS A
INNER JOIN ANIMAL_OUTS B ON A.ANIMAL_ID = B.ANIMAL_ID
WHERE 1=1
AND A.SEX_UPON_INTAKE LIKE 'Intact%'
AND B.SEX_UPON_OUTCOME NOT LIKE 'Intact%'
ORDER BY 1

# 모든 시간에 대한 동물수 COUNT(*)은 row수를 세고 count(animal_id)는 animal_id가 있는 경우만 count
WITH RECURSIVE cte AS (
    SELECT 0 AS hour
    UNION ALL
    SELECT hour +1 FROM cte WHERE hour < 23
)
SELECT b.hour, count(animal_id)
FROM ANIMAL_OUTS a
right join cte b on HOUR(a.DATETIME)  = b.hour
group by 1
order by 1

# GROUP BY를 잘 하고, GROUP BY 가 안되면 레코드가 1개나옴
SELECT A.PRODUCT_ID, B.PRODUCT_NAME, SUM(A.AMOUNT*B.PRICE)
FROM FOOD_ORDER A
INNER JOIN FOOD_PRODUCT B ON A.PRODUCT_ID = B.PRODUCT_ID
WHERE 1=1
AND DATE_FORMAT(A.PRODUCE_DATE,'%Y-%m') = '2022-05'
GROUP BY 1,2
ORDER BY 3 DESC, 1

# 2개조건
select FLAVOR
from
(SELECT A.FLAVOR, SUM(A.TOTAL_ORDER) TOTAL_ORDER
FROM FIRST_HALF A
INNER JOIN ICECREAM_INFO B ON A.FLAVOR = B.FLAVOR
WHERE 1=1
AND B.INGREDIENT_TYPE = 'fruit_based'
group by 1
having SUM(A.TOTAL_ORDER) >=3000
) c
order by TOTAL_ORDER desc

# 최종 출력물 OUTPUT과 비교해보기, 불필요한 컬럼 추출

# 시간 제약이 있으면 걸기
SELECT HOUR(DATETIME) HOUR
,COUNT(DISTINCT ANIMAL_ID)
FROM ANIMAL_OUTS
WHERE 1=1
AND HOUR(DATETIME) BETWEEN 9 AND 19
GROUP BY 1
ORDER BY 1

# 복잡한 내용일때는 결과 출력물을 잘 참고하자
WITH SALES AS (
    SELECT DISTINCT A.USER_ID
    ,YEAR(B.SALES_DATE) YEAR
    ,MONTH(B.SALES_DATE) MONTH
    FROM USER_INFO A
    LEFT JOIN ONLINE_SALE B ON A.USER_ID = B.USER_ID
    WHERE YEAR(A.JOINED)=2021
)
SELECT A.YEAR
,A.MONTH
,COUNT(DISTINCT A.USER_ID) PUCHASED_USERS
,ROUND(COUNT(DISTINCT A.USER_ID)/USER_CNT,1) PUCHASED_RATIO
FROM SALES A
INNER JOIN (SELECT COUNT(DISTINCT USER_ID) USER_CNT FROM SALES) B ON 1=1
WHERE YEAR IS NOT NULL
GROUP BY 1,2

# 컬럼명도 같게한다면
SELECT YEAR(YM) YEAR
,ROUND(AVG(PM_VAL1),2) PM10
,ROUND(AVG(PM_VAL2),2) `PM2.5`
FROM AIR_POLLUTION
WHERE LOCATION2 = '수원'
GROUP BY 1
ORDER BY 1

# ORDERING할때 숫자가 아닌것이 되었다면, 숫자 기준으로 ORDERING하고 문제 제대로 읽기
SELECT ROUTE
, CONCAT(ROUND(SUM(D_BETWEEN_DIST),1),'km') TOTAL_DISTANCE
, CONCAT(ROUND(AVG(D_BETWEEN_DIST),2),'km') AVERAGE_DISTANCE
FROM SUBWAY_DISTANCE
GROUP BY 1
ORDER BY ROUND(SUM(D_CUMULATIVE),1) DESC

# 소숫점 몇째짜리까지 보는지 제대로 확인하기
SELECT A.DEPT_ID
, A.DEPT_NAME_EN
, ROUND(AVG(B.SAL),0) AVG_SAL
FROM HR_DEPARTMENT A
INNER JOIN HR_EMPLOYEES B ON A.DEPT_ID = B.DEPT_ID
GROUP BY 1,2
ORDER BY 3 DESC

# 정보 잘읽기
SELECT EMP_NO
,EMP_NAME
,GRADE
,SAL*BONUS_RATE BONUS
FROM 
    (SELECT A.*
    ,CASE WHEN SCORE>=96 THEN 'S'
    WHEN SCORE>=90 THEN 'A'
    WHEN SCORE>=80 THEN 'B'
    ELSE 'C' END AS GRADE
    ,CASE WHEN SCORE>=96 THEN 0.2
    WHEN SCORE>=90 THEN 0.15
    WHEN SCORE>=80 THEN 0.1
    ELSE 0 END AS BONUS_RATE
    FROM
        (SELECT A.EMP_NO
        ,A.EMP_NAME
        ,A.SAL
        ,AVG(B.SCORE) SCORE
        FROM HR_EMPLOYEES A
        INNER JOIN HR_GRADE B ON A.EMP_NO=B.EMP_NO
        GROUP BY 1,2,3
        ) A
    ) B
ORDER BY 1 

# 정보 잘 읽기 점수가 상/하반기임
SELECT A.SCORE
,A.EMP_NO
,B.EMP_NAME
,B.POSITION
,B.EMAIL
FROM
    (SELECT EMP_NO
    ,SUM(SCORE) SCORE
    FROM HR_GRADE
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 1) A
INNER JOIN HR_EMPLOYEES B ON A.EMP_NO = B.EMP_NO

# parent가 가장 시초고 그 다음이 업그레이드임
SELECT DISTINCT B.ITEM_ID
,C.ITEM_NAME
,C.RARITY
FROM
    (SELECT *
    FROM ITEM_INFO
    WHERE RARITY = 'RARE') A
INNER JOIN ITEM_TREE B ON A.ITEM_ID = B.PARENT_ITEM_ID AND B.ITEM_ID IS NOT NULL
INNER JOIN ITEM_INFO C ON B.ITEM_ID = C.ITEM_ID
ORDER BY 1 DESC

# 문제 잘 이해하기 PARENT_ID > ITEM_ID 로 업그레이드 이며 PARENT_ID에 없으면 더이상 업그레이드 못함
SELECT A.ITEM_ID
,B.ITEM_NAME
,B.RARITY
FROM ITEM_TREE A
INNER JOIN ITEM_INFO B ON A.ITEM_ID = B.ITEM_ID
WHERE 1=1
AND A.ITEM_ID NOT IN (SELECT PARENT_ITEM_ID FROM ITEM_TREE WHERE PARENT_ITEM_ID IS NOT NULL)
ORDER BY 1 DESC


# 문제를 잘 읽고 할인은 1-이고 없는경우 IFNULL 이고
SELECT HISTORY_ID
# D.*
#,IFNULL(E.DISCOUNT_RATE,0) DISCOUNT_RATE
,TRUNCATE(DAYS*DAILY_FEE*(1-IFNULL(E.DISCOUNT_RATE,0)/100),0) FEE
#, TRUNCATE(SUM(DAYS*DAILY_FEE*(1-IFNULL(E.DISCOUNT_RATE,0)/100)),0) FEE
FROM 
    (SELECT C.*
    ,CASE WHEN DAYS>=90 THEN '90일 이상'
    WHEN DAYS>=30 THEN '30일 이상'
    WHEN DAYS>=7 THEN '7일 이상'
    ELSE '7일 미만' END AS DURATION_TYPE
    FROM 
        (SELECT A.HISTORY_ID
        ,B.CAR_TYPE
        ,A.CAR_ID
        ,(DATEDIFF(A.END_DATE,A.START_DATE)+1) DAYS
        ,B.DAILY_FEE
        FROM CAR_RENTAL_COMPANY_RENTAL_HISTORY A
        INNER JOIN CAR_RENTAL_COMPANY_CAR B ON A.CAR_ID = B.CAR_ID AND B.CAR_TYPE = '트럭'
        ) C
    ) D
LEFT JOIN CAR_RENTAL_COMPANY_DISCOUNT_PLAN E ON D.CAR_TYPE = E.CAR_TYPE 
                                            AND D.DURATION_TYPE = E.DURATION_TYPE 
ORDER BY 2 DESC, 1 DESC


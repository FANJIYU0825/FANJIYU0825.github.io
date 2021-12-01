# FANJIYU0825.github.io
個人網站
# 推薦系統論文應用
## 統整文
[CSDN](https://blog.csdn.net/SmartLab307/article/details/115368550?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161924726116780269819550%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=161924726116780269819550&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-8-115368550.first_rank_v2_pc_rank_v29&utm_term=%E6%8E%A8%E8%96%A6%E7%B3%BB%E7%B5%B1)
[部落格](https://zhuanlan.zhihu.com/p/270918998)
## echo效應 of RS
[ECHO所造成的影響](https://arxiv.org/pdf/2007.02474.pdf)
### data collection
:::success
我們把資料分成 三個部份別是
1. browser log: page view
2. click log :user’s click behaviors
3. purchase
:::
## 階層 [HRNN](https://arxiv.org/pdf/1706.04148.pdf) 
Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks
1. 使用 無縫是系統
![](https://i.imgur.com/XRItOit.png)


## [阿里巴巴](https://arxiv.org/pdf/1902.00851.pdf) (應用思考)
[官方code](https://github.com/rec-agent/rec-rl/blob/master/examples/rec_es/rec_input_fn_local.py)
1. 最後最終理想訂閱制公式  
2. [編碼閱讀](/KHCDlbfHTSObuIu_iouUQQ)
:::spoiler
``` =pyhon
```
:::
![](https://i.imgur.com/InePsH3.png)
我們要依據不同的點擊行為來為後面當作分數的輔助
:::info
agent
期望reward = Year~price~*(低ARPU X續訂率1+中ARPU X續訂率2 + 高ARPUX 續訂率3)*P[ 有無按下續訂閱按鈕 ]
:::
![](https://i.imgur.com/txu0e3S.png)
:::danger
目前卡在訂閱率的參數有問題
:::
:::success


2. ARPU分層 :
    a. 定義ARPU高中低的標準
    b. Clustering algorithms為每一群ARPU族群定義行為特徵內容
    c. 每次動作都優化周續訂率
    d. 什麼東西影響周續訂率?
:::
:::success


3. 第一階段 : 分群演算法 –
為每一群ARPU族群定義行為特徵內容(kMEANS)
4. 第二階段 : 先驗演算法 -
每一個分群的喜好物品用先驗演算法計算互相關聯計算、物品組合(RIO計算投報率)來獲得topk list
5. 第三階段 : Reward極大化強化學習( offline agent) value base
:::
# 時間序列演匴法
## Study Cat(時間序列資料演算選擇與應用)
[視覺化分析](https://github.com/FANJIYU0825/data-analyst--pratice/blob/master/ML_pre/SCORING_MODEL/ML%20Experience.ipynb)
:::info

第一次開發偏向助手
做的是用現行的資料做統計學習分析
:::
:::success


1. 證明遊戲有更好的成效在學習語言
先驗演算法 -
運用時間序列的分析:互相關聯計算、物品組合用
:::
:::success


2. 對學生學習效率跟後面所要達到的分數達到預期與否 提醒老師學生狀態
:::
# 製成
## 製成在強化學習的可行性研究
:::spoiler
PVT (process, voltage, temperature)
PPA (performance,power, area)
:::
![](https://i.imgur.com/is7ajox.png)
運用經驗來去設定 環境以及agent
![](https://i.imgur.com/UV1Bf1y.png)
希望讓agent 可以去學習這個flow
![](https://i.imgur.com/9UVfZIR.png)
1. X is a set of finite sizing variables to be searched,
2. each has a non-empty domain Di
, namely the design space,and bk is the possible values.
3. C is a set of constraints. A
    constraint is a pair consists of a constraint scope tj and a
    relation rj over the variables in the scope, limiting feasible
    permutations of assignments.
![](https://i.imgur.com/GO86YtC.png)
為了找到OPT 我們需要花費多少
4. 現行方不同製成方案做出比較分析: 
     1. 設定
     2. openmp
     3. from 45>22
     4. find best way of PVT
     5. case
        
5. 結論:雖然我們都可以發現ai有大大優於人力(可以達到人類的程度) BUT TOP的選擇上還是需要花費大量的人工經驗所以還是需要花很多時間去克服的目前機器無法做到

## 兒童英文學習使用
論文尚未推出
[網站](https://solepera.github.io/pubs/)
## 新技術探詢
[lowcode bulid sysytem](https://dl.acm.org/doi/pdf/10.1145/3417990.3420202)
## 先驗演算法
:::info


![](https://i.imgur.com/RRVHojh.png)
這邊我們會使用相關 套件
``` =python
#support rate cacutlate 
from mlxtend.frequent_patterns import apriori
# trans data to 
from mlxtend.preprocessing import TransactionEncoder
```
:::
:::success
因為 這邊是需要使用 
計算支持度會先需要使用
會需要把資料轉會成為 
numpy 格式
``` =python
te = TransactionEncoder()
te_ary1 = te.fit(userfeaturesdata_cat1['product']).transform(userfeaturesdata_cat1['product'])
```


:::

# DRL survey of Recommend
## tech
### (problem setting)
:::success
RL is learning how to map situation to actions
1. formulate situation
2. learn mapping
:::
:::info
1. formulate situation 
    1. MAB 連續動作
        * a^*
    2. MDP離散動作
        * Multi agnet setting 可為單一並非合作也非競爭 可以是independ or depend 
### policy learning

:::
## RECSYS 2021 
-[介紹 CF](https://in.ncu.edu.tw/~hhchen/academic_works/chen19_diff_reg_weight.pdf)
[搜尋論文](https://dl.acm.org/doi/proceedings/10.1145/3460231)
- [ML套件](http://rasbt.github.io/mlxtend/)
- [DQN 學習](/2YfL27R7Qza58KxJmFds7A)
- [推薦gym](/bvrQHPZBSBW8MWSmjOixuQ)
- [RL](/MKRQI0jZRou-6HOVQDOQTA)
- [Deep Reinforcement Learning for Search,
Recommendation, and Online Advertising: A](https://arxiv.org/pdf/1812.07127.pdf)

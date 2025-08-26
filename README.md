# 本專案分兩個部分 前半數據分析 後半機器學習
# 筆電數據分析
主要目標是分析一個關於筆記型電腦價格的資料集，並透過資料視覺化來探索各種因素如何影響筆電的價格。以下我將詳細說明這個專案的目的、流程以及想要分析的內容：

專案目的：

資料探索與清理： 專案首先要讀取並檢視原始資料，了解資料的結構、型態，並進行必要的資料清理，例如處理缺失值、異常值等，確保後續分析的準確性。

價格影響因素分析： 專案旨在找出哪些因素（例如：品牌、RAM 大小、作業系統、CPU、GPU、儲存類型等）對筆記型電腦的價格有顯著影響。

資料視覺化呈現： 利用圖表（如長條圖）將分析結果視覺化呈現，使得複雜的資料關係更容易理解，並協助讀者快速掌握重點。

從歐元轉換為台幣： 專案會將原始資料集中的價格從歐元轉換為台幣，以更符合台灣地區的消費習慣。

篩選資料範圍： 專案會將不需要的資料如一些不常用的作業系統的筆電篩選掉，使分析結果更精確。

專案流程與分析重點：

導入必要的函式庫： 專案開始會導入 pandas（用於資料處理）、 numpy（用於數值計算）、 seaborn 和 matplotlib（用於資料視覺化）。

讀取資料： 使用 pd.read_csv() 讀取名為 laptop_prices.csv 的資料集。

資料概覽：

df.info(): 輸出資料集的資訊，包括欄位名稱、型態、以及是否有缺失值等。

df.describe(): 輸出數值型欄位的統計摘要，如平均值、標準差、最小值、最大值等。

新增台幣價格欄位：

建立一個名為 Price_twd 的新欄位，將原有的歐元價格 (Price_euros) 乘以 35 (匯率)。

資料篩選：

使用 df['OS'].unique()找出作業系統有哪些類別，並篩選掉不需要分析的作業系統 No OS、 Mac OS X、 Android、Windows 10 S，將篩選後的資料存為 df_OS_clear。

這部分也示範了兩種寫法，一種使用 & 結合多個條件，另一種使用 ~df['OS'].isin([...]) 來排除指定清單的資料。

資料匯出：

將篩選後的資料框 df_OS_clear 存為 CSV 檔 laptop_OS_clear.csv。

視覺化設定：

設定 matplotlib 的字型和大小，確保圖表中的中文能正確顯示。

主要視覺化分析：

筆電品牌分佈： 使用 df['Company'].value_counts().plot(kind='bar', color='blue') 和 df_OS_clear['Company'].value_counts().plot(kind='bar', color='red') 繪製長條圖，顯示資料集中各品牌筆電的數量，以了解哪些品牌在資料集中較為常見。（篩選前後各畫一張圖）

作業系統分佈： 使用 df_OS_clear['OS'].value_counts().plot(kind='bar',color='lightgreen')繪製長條圖，顯示資料集中各作業系統的筆電數量，以了解主要筆電的作業系統。

RAM 大小與平均價格關係： 使用 sns.barplot() 繪製長條圖，展示不同 RAM 大小下筆電的平均價格。這部分使用了兩種方式取得資料，並用不同的資料呈現。

RAM 大小與總價格關係： 使用 sns.barplot() 繪製長條圖，展示不同 RAM 大小下筆電的總價格。

筆電品牌與平均價格關係： 使用 sns.barplot() 繪製長條圖，展示不同品牌筆電的平均價格。

筆電品牌與總價格關係： 使用 sns.barplot() 繪製長條圖，展示不同品牌筆電的總價格。

想要分析出的內容：

筆電市場分佈： 了解各品牌在市場上的佔有率，以及各種作業系統的使用情況。

RAM 大小對價格的影響： 找出 RAM 大小與價格之間的關係，例如 RAM 容量增加是否會導致價格顯著上升。

品牌與價格的關聯： 判斷不同品牌之間的價格差異，以及哪些品牌通常提供較高或較低價格的產品。

其他因素的影響： 藉由資料探索，是否能發現其他可能影響價格的因素。


# 筆記型電腦規格與價格預測模型

本專案旨在使用機器學習演算法，根據筆記型電腦的規格因子預測其價格（Price_twd）。我們使用了包含多種筆電規格資訊的資料集，並嘗試了三種不同的迴歸模型來完成這項預測任務。

## 資料集

包含多種筆記型電腦的規格資訊，例如：品牌、產品名稱、類型、螢幕尺寸、記憶體大小、作業系統、重量、價格（歐元和新台幣）、螢幕解析度等等。

## 方法與過程

以下是數據處理和模型訓練的詳細步驟：

1.  **導入函式庫：**
    *   導入 `pandas` 用於資料處理。
    *   導入 `scikit-learn` 用於機器學習模型和評估。

2.  **讀取資料：**
    *   使用 `pandas` 的 `read_csv` 函數讀取 `laptop_OS_clear.csv` 資料集。

3.  **數據預處理：**

    *   **類別型特徵編碼：**
        *   使用 `LabelEncoder` 將類別型欄位（如：品牌、產品名稱、作業系統等）轉換為數值型，方便模型使用。
        ```python
        categorical_features = ['Company', 'Product', 'TypeName', 'OS', 'Screen', 
                                 'ScreenW', 'ScreenH', 'Touchscreen', 'IPSpanel', 'RetinaDisplay',
                                 'CPU_company', 'CPU_model', 'PrimaryStorageType',
                                 'SecondaryStorageType', 'GPU_company', 'GPU_model']
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        ```
    *   **數值型特徵處理與標準化：**
        *   使用 `pd.to_numeric` 將數值型欄位轉換成數值。
        *   使用 `fillna` 將缺失值填補為該欄位的平均值。
        *   使用 `StandardScaler` 對數值型欄位進行標準化，使數據具有零均值和單位方差。
        ```python
        numerical_features = ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage', 'SecondaryStorage']
        for col in numerical_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].mean(), inplace=True)

        scaler = StandardScaler()
        numerical_features_scaled = scaler.fit_transform(df[numerical_features])
        numerical_features_scaled = pd.DataFrame(numerical_features_scaled, columns=numerical_features)
        df[numerical_features] = numerical_features_scaled
        ```
    *   **特徵選擇：**
        *   將目標變數 `Price_twd` 從特徵欄位中分離出來。

4.  **分割資料集：**

    *   使用 `train_test_split` 將資料集分割為訓練集和測試集，訓練集佔 80%，測試集佔 20%。同時設定 `random_state` 確保結果可以重複。
        ```python
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        ```

5.  **建立並訓練模型：**

    *   **線性迴歸模型：**
        *   建立 `LinearRegression` 模型並使用訓練集資料進行訓練。
            ```python
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            ```

    *   **隨機森林迴歸模型：**
        *   建立 `RandomForestRegressor` 模型，設定 `n_estimators` 為 100，並使用訓練集資料進行訓練。
            ```python
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            ```

    *   **梯度提升機迴歸模型：**
        *   建立 `GradientBoostingRegressor` 模型，設定 `n_estimators` 為 100， `learning_rate` 為 0.1，`max_depth` 為 3，並使用訓練集資料進行訓練。
            ```python
            gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            gb_model.fit(X_train, y_train)
            ```

6.  **模型評估：**

    *   使用測試集對各模型進行預測。
    *   使用 `mean_squared_error` 計算均方誤差（MSE）。
    *   使用 `r2_score` 計算 R2 分數。
        *   MSE 越小，表示模型預測值與真實值之間的誤差越小。
        *   R2 分數越接近 1，表示模型能夠解釋的變異程度越高。

## 模型選擇與結果分析

嘗試了以下三個迴歸模型：

1.  **線性迴歸模型：**
    *   結果：
        *   MSE: 157754706.49
        *   R2 : 0.75
    *   說明：線性迴歸是一個簡單的基線模型，由於假設特徵與目標變數之間存在線性關係，所以解釋能力較差，預測效果也相對較差。

2.  **隨機森林迴歸模型：**
    *   結果：
        *   MSE: 83984442.25
        *   R2 : 0.86
    *   說明：隨機森林能夠處理非線性關係，並透過集成學習提高模型的穩定性，因此相較於線性迴歸有更好的表現。

3.  **梯度提升機迴歸模型：**
    *   結果：
        *   MSE: 92708804.56
        *   R2 : 0.85
    *   說明：梯度提升機的性能雖然與隨機森林相當，但由於在我們的實驗中，隨機森林的參數稍微調優較好，因此隨機森林的評估結果較好。

根據評估結果，**選擇隨機森林迴歸模型作為最佳模型**，原因如下：

*   **效能：** 隨機森林的 R2 分數最高 (0.86)，表示它能夠解釋更多價格變異，而 MSE 也相對較小，表示預測誤差較低。
*  **整體表現：** 隨機森林迴歸模型不僅在準確度上勝出，其訓練速度和超參數調整的難易度也較為平衡。

儘管梯度提升機在其他問題上可能表現更佳，但在此資料集中，隨機森林模型取得了較好的效果。未來可以嘗試針對隨機森林的超參數做更細緻的調整，以及嘗試其他機器學習模型。

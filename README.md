# Waveguide Crossing Tutorial (Ultra-compact dual-mode)
此Tutorial主要針對 **Ultra-compact dual-mode Waveguide Crossing (兼容超緊湊及雙模態的交叉型波導)** 的設計與實作。
### 內容簡介
* **`Crossing_Tutorial.ipynb`**：這是一份 Jupyter Notebook Tutorial，裡面說明了波導交叉的幾何設計原理，並利用圖片（如八分之一對稱結構）輔助解釋如何簡化設計變數與建立結構。
* **`Waveguide_Crossing_2.4x2.4.py`**：隨附的 Python 程式碼，包含了建立 $2.4\mu m \times 2.4\mu m$ 尺寸波導交叉所需的參數與函數，用於實際要用來跑設計模擬。
* **factorization machine**要跟Crossing_Tutorial.ipynb或是Waveguide_Crossing_2.4x2.4.py放在同一資料夾才可以執行(例如，mpirun -n 31 python Waveguide_Crossing.py > test.log 2>&1)
* 執行時，在Terminal終端輸入```mpirun -n '核心數' python <py檔名>.py > <log檔名>.log 2>&1```

歡迎直接打開 `Crossing_Tutorial.ipynb` 開始閱讀！


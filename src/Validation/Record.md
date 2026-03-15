# ERRORS RECORD
**5th Feb** 
获得s1, s2的csv，验证完成后便与s5（前端）对接

1. s2的csv并未按60秒切分好
2. 在前端实现后，发现地面站点和无人机的z轴都高于现实约几百米

**8th Mar**
要將s1-s4的代碼在同一台機器上運行

1. s2沒將依賴倉庫SAREnv傳上共享倉庫且未提交說明文檔導致無法正常運行s2的代碼
2. s2的`generate_dataset.py`腳本比原依賴倉庫多了一個屬性︰"sizes"導致無法運行
3. s3產生的`topology_links`和`routing_rules`直接生成於s3的文件夾，而S4則是從S3的`output`中的`links`,`rules`找對應的.csv和.json文件。建議S3修改輸出文件部分的代碼，更好使用及管理輸出文件。
4. s1,s2的腳本模擬的總時長不一致（10min&7min）
5. 優化s1,s2的output使s3順利讀取
6. 
@:start

:: 設定時間
@set /a StartMS=%time:~9,2%
@set /a StartS=%time:~6,2%
@set /a StartM=%time:~3,2%
@set /a StartH=%time:~0,2%


:: 顯示現在的時間們
:: echo 現在小時：%StartH%
:: echo 現在分鐘：%StartM%
:: echo 現在秒數：%StartS%
:: echo 現在微秒：%StartMS%
::echo 現在時間：%time% 

:: 如果時間到了就觸發go方法
if "%StartS%" == "0" goto go

:: 如果時間沒有到，休息一秒，再重新檢查是否時間到
@timeout /t 1


:: 回到開頭重新檢查
@goto start



:go

:: 顯示文字
echo 整數分鐘已到，啟動載入資料程序...

:: 延遲5秒，讓mql5爬取資料回csv
@timeout /t 5

:: 啟動預測的py檔
echo 載入資料完成，正在進行預測...

:: 顯示預測過程
:: start C:\Users\D\Desktop\DNNAutomatedTradingProject\propfecy_dnn_預測.py

:: 利用vbs隱藏預測過程
start C:\Users\D\Desktop\DNNAutomatedTradingProject\p.vbs

:: 延遲45秒，讓py檔可以做出預測
@timeout /t 45

echo 下單中...

:: 導回start檢查是否下一次時間到
goto start

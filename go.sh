#!/bin/bash

# スクリプトファイルの絶対パスを取得
SCRIPT_DIR=$(cd &(dirname $0); pwd)
# カレントディレクトリをスクリプトの位置に移動
cd $SCRIPT_DIR
#ログ出力フォルダを生成
mkdir -p output_data

#全CPUを100%で可動させる
sudo stress-ng -c 0&
# GPU高負荷実行　領域抽出繰り返し
/usr/bin/python3 python/AI.py &

#1[s]間隔で実行中のデータをログに吐き出す。
sudo tegrastats --interval 1000 --logfile output_data/data_$(date +"%m%d%H%M").txt &

sleep 10
#実行区間中にjtopを実行しモニタリングできるようにする。
#スクリプト実行後jtopが起動できなくなることがあるため表示させない。
#sudo jtop

#エンターキーの押下待ち
echo "Enterキーが押されるまで待つ"
read wait

#ログ保存の終了
sudo tegrastats --stop
sleep 1


#負荷試験終了
sudo pkill stress-ng
sudo pkill python3
sleep 1


#カレントディレクトリを元の位置に戻す
cd -

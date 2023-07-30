# （注：開発途中です）
このレポジトリにはC++/CUDAで自作しているAIのコードを置いています。  
CUDAコンパイラが無いとコンパイル出来ません。

# 重要
柔軟性を上げるためにアーキテクチャを改良する予定だが、画像処理エンジンと連携させたいので、  
今後は https://github.com/asahi-kojima/ImageProcessingEngine に作業を移行しようと考えている。

### 今後の開発予定（横棒は完了）
・~~BatchNorm層の高速化~~  
・~~TransposedConvolution層の実装~~  
・~~単体テストの追加~~  
・~~MaxPooling層の実装~~  
・OpenCVを使ってフィルター行列の可視化などを行う  
・リファクタリング  
・CPU計算をマルチスレッド化もしくはSIMD化  
・CPU計算の最適化  
・GPU計算の最適化（SharedMemoryなどの利用）  
・DLL化してC#/XAMLでUIを作る or DirectX12でGUIを作成。
